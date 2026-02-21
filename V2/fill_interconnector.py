#!/usr/bin/env python3
"""
Fill missing V-SA interconnector data for 2024-08 through 2025-12.

Strategy:
  1. Try MMSDM historical archive (same pattern as download_data.py).
     By Feb 2026 this covers up to ~Oct/Nov 2025.
  2. For months still missing, fall back to NEMWEB daily DispatchIS_Reports
     ZIP files (five-minute dispatch, same DISPATCHINTERCONNECTORRES table).
  3. Patch data/processed/sa_nem_2023_2025.csv in place.

Run:
  python fill_interconnector.py
(No API key needed — uses NEMWEB only.)
"""

import io
import time
import zipfile
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent
IC_RAW_DIR    = BASE_DIR / "data" / "raw" / "interconnector"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CSV    = PROCESSED_DIR / "sa_nem_2023_2025.csv"

# Months to fill: 2024-08 through 2025-12
FILL_START = date(2024, 8, 1)
FILL_END   = date(2025, 12, 31)

# MMSDM monthly archive URL — two formats:
#   v1 (pre-2024): PUBLIC_DVD_DISPATCHINTERCONNECTORRES_{year}{month:02d}010000.zip
#   v2 (2024+):    PUBLIC_ARCHIVE#DISPATCHINTERCONNECTORRES#FILE01#{year}{month:02d}010000.zip
MMSDM_URL_V1 = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}"
    "/MMSDM_Historical_Data_SQLLoader/DATA"
    "/PUBLIC_DVD_DISPATCHINTERCONNECTORRES_{year}{month:02d}010000.zip"
)
MMSDM_URL_V2 = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}"
    "/MMSDM_Historical_Data_SQLLoader/DATA"
    "/PUBLIC_ARCHIVE%23DISPATCHINTERCONNECTORRES%23FILE01%23{year}{month:02d}010000.zip"
)
# Keep old name for backward compatibility
MMSDM_URL = MMSDM_URL_V1

# DispatchIS_Reports daily archive (5-minute dispatch files)
# One ZIP per day, contains the full dispatch interval table
DAILY_URL = (
    "https://nemweb.com.au/Reports/Archive/DispatchIS_Reports"
    "/PUBLIC_DISPATCHIS_{year}{month:02d}{day:02d}.zip"
)

IC_RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_dispatchinterconnectorres(text: str, source: str) -> pd.DataFrame:
    """
    Parse the DISPATCHINTERCONNECTORRES table from a NEMWEB CSV blob.
    Returns a DataFrame with columns [SETTLEMENTDATE, MWFLOW] for V-SA rows.
    """
    rows = []
    header = None
    for line in text.splitlines():
        parts = line.split(",")
        if not parts:
            continue
        rec_type = parts[0].strip().upper()
        if rec_type == "I" and len(parts) > 4:
            tbl = parts[2].strip().upper()
            if "INTERCONNECTORRES" in tbl:
                header = [c.strip().strip('"') for c in parts[4:]]
        elif rec_type == "D" and header is not None and len(parts) > 4:
            tbl = parts[2].strip().upper()
            if "INTERCONNECTORRES" in tbl:
                row_data = [c.strip().strip('"') for c in parts[4:]]
                if len(row_data) == len(header):
                    rows.append(row_data)

    if not rows or header is None:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)
    if "INTERCONNECTORID" not in df.columns:
        return pd.DataFrame()

    df = df[df["INTERCONNECTORID"].str.strip() == "V-SA"].copy()
    if df.empty:
        return pd.DataFrame()

    df = df[["SETTLEMENTDATE", "MWFLOW"]].copy()
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
    df["MWFLOW"] = pd.to_numeric(df["MWFLOW"], errors="coerce")
    return df.dropna(subset=["SETTLEMENTDATE", "MWFLOW"])


def open_nemweb_zip(content: bytes, source: str) -> pd.DataFrame:
    """Extract all CSVs from a NEMWEB ZIP and parse DISPATCHINTERCONNECTORRES."""
    frames = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            # Some daily ZIPs contain nested ZIPs; handle both
            for name in zf.namelist():
                if name.upper().endswith(".ZIP"):
                    # Nested ZIP (common in DispatchIS_Reports)
                    inner_bytes = zf.read(name)
                    inner_frames = open_nemweb_zip(inner_bytes, name)
                    if not inner_frames.empty:
                        frames.append(inner_frames)
                elif name.upper().endswith(".CSV"):
                    text = zf.read(name).decode("utf-8", errors="replace")
                    df = parse_dispatchinterconnectorres(text, name)
                    if not df.empty:
                        frames.append(df)
    except zipfile.BadZipFile:
        return pd.DataFrame()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def try_mmsdm_month(year: int, month: int) -> pd.DataFrame:
    """Try MMSDM monthly archive (v1 then v2 URL format). Returns DataFrame or empty."""
    cache = IC_RAW_DIR / f"interconnector_{year}_{month:02d}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["SETTLEMENTDATE"])
        return df

    for url_template in (MMSDM_URL_V1, MMSDM_URL_V2):
        url = url_template.format(year=year, month=month)
        try:
            r = requests.get(url, timeout=180)
        except requests.RequestException as exc:
            print(f"    MMSDM request error {year}-{month:02d}: {exc}")
            continue

        if r.status_code != 200:
            continue  # Try next URL format

        df = open_nemweb_zip(r.content, url)
        if df.empty:
            print(f"    WARNING: empty MMSDM parse for {year}-{month:02d}")
            continue

        df.to_csv(cache, index=False)
        print(f"    MMSDM {year}-{month:02d}: {len(df):,} rows cached")
        return df

    return pd.DataFrame()  # Both formats failed


def try_daily_month(year: int, month: int) -> pd.DataFrame:
    """
    Fall back: download all daily DispatchIS_Reports ZIPs for the month.
    Returns aggregated DataFrame.
    """
    # Figure out all days in this month
    first = date(year, month, 1)
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    last = min(last, FILL_END)

    daily_frames = []
    day = first
    while day <= last:
        day_cache = IC_RAW_DIR / f"daily_{year}{month:02d}{day.day:02d}.csv"
        if day_cache.exists():
            df = pd.read_csv(day_cache, parse_dates=["SETTLEMENTDATE"])
            daily_frames.append(df)
            day += timedelta(days=1)
            continue

        url = DAILY_URL.format(year=year, month=month, day=day.day)
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException:
            day += timedelta(days=1)
            continue

        if r.status_code != 200:
            day += timedelta(days=1)
            continue

        df = open_nemweb_zip(r.content, url)
        if not df.empty:
            df.to_csv(day_cache, index=False)
            daily_frames.append(df)

        time.sleep(0.2)  # polite rate limiting
        day += timedelta(days=1)

    if not daily_frames:
        return pd.DataFrame()
    return pd.concat(daily_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def months_to_fill():
    """Yield (year, month) tuples from FILL_START to FILL_END."""
    cur = FILL_START.replace(day=1)
    while cur <= FILL_END:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)


def resample_to_30min(df: pd.DataFrame) -> pd.Series:
    """Resample 5-minute MWFLOW to 30-minute mean, indexed by settlement_datetime."""
    df = df.copy()
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df.dropna(subset=["SETTLEMENTDATE", "MWFLOW"]).set_index("SETTLEMENTDATE").sort_index()
    rs = df["MWFLOW"].resample("30min").mean()
    rs.index.name = "settlement_datetime"
    return rs


def main():
    print("=" * 60)
    print("Fill interconnector gap: 2024-08 to 2025-12")
    print("=" * 60)

    all_frames = []
    months = list(months_to_fill())
    print(f"Months to process: {len(months)}\n")

    for year, month in tqdm(months, desc="Months", unit="mo"):
        print(f"\n  {year}-{month:02d}:")

        # Try MMSDM first (covers up to ~Nov 2025 by Feb 2026)
        df = try_mmsdm_month(year, month)

        if df.empty:
            print(f"    MMSDM not available — trying daily DispatchIS_Reports ...")
            df = try_daily_month(year, month)

        if df.empty:
            print(f"    WARNING: no data found for {year}-{month:02d}")
        else:
            all_frames.append(df)

    if not all_frames:
        print("\nERROR: No new interconnector data downloaded. Exiting.")
        return

    new_ic = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal new rows: {len(new_ic):,}")

    # Resample to 30-minute
    ic_30 = resample_to_30min(new_ic).rename("interconnector_mw")

    # ---------------------------------------------------------------------------
    # Patch the processed CSV
    # ---------------------------------------------------------------------------
    print(f"\nLoading existing output: {OUTPUT_CSV}")
    master = pd.read_csv(OUTPUT_CSV, parse_dates=["settlement_datetime"])
    master = master.set_index("settlement_datetime")
    master.index = pd.to_datetime(master.index)

    n_before = master["interconnector_mw"].isna().sum()
    print(f"  NaN interconnector rows before patch: {n_before:,}")

    # Update only the NaN slots (preserve existing values)
    master["interconnector_mw"] = master["interconnector_mw"].combine_first(ic_30)

    n_after = master["interconnector_mw"].isna().sum()
    print(f"  NaN interconnector rows after  patch: {n_after:,}")
    print(f"  Filled: {n_before - n_after:,} rows")

    master = master.reset_index()
    master.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved patched CSV → {OUTPUT_CSV}")

    # Summary
    print("\n--- Summary ---")
    print(f"Total rows:             {len(master):,}")
    ic = master["interconnector_mw"]
    print(f"interconnector_mw:      {ic.notna().sum():,} non-NaN  "
          f"({100*ic.notna().mean():.1f}% coverage)")
    print(f"  mean={ic.mean():.1f} MW  min={ic.min():.1f}  max={ic.max():.1f}")


if __name__ == "__main__":
    main()
