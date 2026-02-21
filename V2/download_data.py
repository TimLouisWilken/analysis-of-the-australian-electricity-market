#!/usr/bin/env python3
"""
Download South Australian NEM data (2023–2025) for econometrics analysis.

Variables downloaded:
  - SA1 Regional Reference Price  (OpenElectricity API: /market/network/NEM)
  - SA1 demand                    (OpenElectricity API: /market/network/NEM)
  - VIC1 price                    (OpenElectricity API: /market/network/NEM)
  - SA1 wind power                (OpenElectricity API: /data/network/NEM, fueltech=wind)
  - SA1 solar_utility power       (OpenElectricity API: /data/network/NEM, fueltech=solar_utility)
  - SA1 solar_rooftop power       (OpenElectricity API: /data/network/NEM, fueltech=solar_rooftop)
  - V-SA interconnector flow      (NEMWEB MMSDM: DISPATCHINTERCONNECTORRES)

Uses requests directly (bypasses openelectricity Python client due to aiohttp
Brotli decompression bug in this environment).

Usage:
  export OE_API_KEY="your_key_here"
  python download_data.py

Output:
  data/processed/sa_nem_2023_2025.csv

NOTE on pre-2017 baseline: This dataset covers 2023-2025 only, so the
pre-Hornsdale BESS baseline is not available. H3 (battery dampening test)
must be framed as a within-period cross-sectional or time-series analysis
rather than a before/after comparison. Flag this limitation in the paper.
"""

import io
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OE_API_KEY", "")
if not API_KEY:
    sys.exit(
        "ERROR: OE_API_KEY environment variable not set.\n"
        "Register at https://platform.openelectricity.org.au and then:\n"
        "  export OE_API_KEY='your_key_here'"
    )

OE_BASE   = "https://api.openelectricity.org.au/v4"
OE_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    # Explicitly request gzip, not brotli (aiohttp brotli is broken here;
    # requests handles gzip natively)
    "Accept-Encoding": "gzip, deflate",
}

DATE_START = datetime(2023, 1, 1)
DATE_END   = datetime(2025, 12, 31, 23, 59, 59)

# OpenElectricity API 5-minute resolution; resample to 30-minute after
OE_INTERVAL   = "5m"
RESAMPLE_FREQ = "30min"

BASE_DIR      = Path(__file__).parent
RAW_DIR       = BASE_DIR / "data" / "raw"
IC_RAW_DIR    = RAW_DIR / "interconnector"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CSV    = PROCESSED_DIR / "sa_nem_2023_2025.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
IC_RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def month_range(start: datetime, end: datetime):
    """Yield (month_start, month_end) pairs covering [start, end]."""
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        next_month = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur, min(next_month - timedelta(seconds=1), end)
        cur = next_month


def week_range(start: datetime, end: datetime, chunk_days: int = 7):
    """Yield (chunk_start, chunk_end) pairs in chunk_days-day steps covering [start, end].

    The OE API limits 5m requests to 8 days maximum; we use 7-day chunks.
    """
    cur = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days) - timedelta(seconds=1), end)
        yield cur, chunk_end
        cur = cur + timedelta(days=chunk_days)


def oe_get(endpoint: str, params: dict, label: str) -> list[dict]:
    """Make an OpenElectricity API GET request and return the data list."""
    url = OE_BASE + endpoint
    r = requests.get(url, headers=OE_HEADERS, params=params, timeout=120)
    if r.status_code != 200:
        print(f"  WARNING: {label} → HTTP {r.status_code}: {r.text[:200]}")
        return []
    body = r.json()
    if not body.get("success", True):
        print(f"  WARNING: {label} → API error: {body.get('error')}")
        return []
    return body.get("data", [])


def parse_timeseries(data_list: list[dict], value_key: str) -> pd.DataFrame:
    """
    Parse the OpenElectricity API timeseries data list into a flat DataFrame.

    Each element in data_list corresponds to one metric series.
    Each series has a 'results' list; each result has a 'data' list of
    [timestamp, value] pairs and a 'columns' dict with region/fueltech.

    Returns a DataFrame with columns: interval, <any group cols>, <value_key>
    """
    records = []
    for series in data_list:
        metric = series.get("metric", value_key)
        for result in series.get("results", []):
            cols = result.get("columns", {})
            for point in result.get("data", []):
                ts, val = point[0], point[1]
                row = {"interval": ts, metric: val}
                row.update(cols)
                records.append(row)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["interval"] = pd.to_datetime(df["interval"], utc=True).dt.tz_convert("Australia/Sydney").dt.tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# Section 1: OpenElectricity market data (price + demand)
# ---------------------------------------------------------------------------

def fetch_market_chunk(region: str, chunk_start: datetime, chunk_end: datetime) -> pd.DataFrame:
    """Download price + demand for one region, one 7-day chunk."""
    tag = f"{region}_{chunk_start.strftime('%Y%m%d')}"
    cache_path = RAW_DIR / f"market_{tag}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["interval"])

    data = oe_get(
        "/market/network/NEM",
        {
            "metrics": ["price", "demand"],
            "interval": OE_INTERVAL,
            "date_start": chunk_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "date_end": chunk_end.strftime("%Y-%m-%dT%H:%M:%S"),
            "network_region": region,
        },
        f"market/{region} {chunk_start.strftime('%Y-%m-%d')}",
    )
    df = parse_timeseries(data, "price")
    if df.empty:
        return df
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Section 2: OpenElectricity network data (generation by fueltech)
# ---------------------------------------------------------------------------

def fetch_power_chunk(fueltech: str, chunk_start: datetime, chunk_end: datetime) -> pd.DataFrame:
    """Download power (MW) for one fueltech in SA1, one 7-day chunk."""
    tag = f"SA1_{fueltech}_{chunk_start.strftime('%Y%m%d')}"
    cache_path = RAW_DIR / f"power_{tag}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["interval"])

    data = oe_get(
        "/data/network/NEM",
        {
            "metrics": ["power"],
            "interval": OE_INTERVAL,
            "date_start": chunk_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "date_end": chunk_end.strftime("%Y-%m-%dT%H:%M:%S"),
            "network_region": "SA1",
            "fueltech": fueltech,
        },
        f"power/{fueltech} {chunk_start.strftime('%Y-%m-%d')}",
    )
    df = parse_timeseries(data, "power")
    if df.empty:
        return df
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Section 3: NEMWEB MMSDM interconnector flows (V-SA)
# ---------------------------------------------------------------------------

MMSDM_BASE = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}"
    "/MMSDM_Historical_Data_SQLLoader/DATA"
    "/PUBLIC_DVD_DISPATCHINTERCONNECTORRES_{year}{month:02d}010000.zip"
)


def fetch_interconnector_month(year: int, month: int) -> pd.DataFrame:
    """Download DISPATCHINTERCONNECTORRES for one month, filter V-SA."""
    cache_csv = IC_RAW_DIR / f"interconnector_{year}_{month:02d}.csv"
    if cache_csv.exists():
        return pd.read_csv(cache_csv, parse_dates=["SETTLEMENTDATE"])

    url = MMSDM_BASE.format(year=year, month=month)
    try:
        r = requests.get(url, timeout=180)
        if r.status_code != 200:
            print(f"  WARNING: HTTP {r.status_code} for {year}-{month:02d}: {url}")
            return pd.DataFrame()
        data = r.content
    except requests.RequestException as exc:
        print(f"  WARNING: request error for {year}-{month:02d}: {exc}")
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = next((n for n in zf.namelist() if n.upper().endswith(".CSV")), None)
            if csv_name is None:
                print(f"  WARNING: no CSV in ZIP for {year}-{month:02d}")
                return pd.DataFrame()
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        print(f"  WARNING: bad ZIP for {year}-{month:02d}: {exc}")
        return pd.DataFrame()

    rows = []
    header = None
    for line in raw_text.splitlines():
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
        print(f"  WARNING: could not parse DISPATCHINTERCONNECTORRES for {year}-{month:02d}")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)
    if "INTERCONNECTORID" not in df.columns:
        print(f"  WARNING: INTERCONNECTORID column missing; got: {list(df.columns[:10])}")
        return pd.DataFrame()

    df = df[df["INTERCONNECTORID"].str.strip() == "V-SA"].copy()
    if df.empty:
        print(f"  WARNING: no V-SA rows in {year}-{month:02d}")
        return pd.DataFrame()

    df = df[["SETTLEMENTDATE", "MWFLOW"]].copy()
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
    df["MWFLOW"] = pd.to_numeric(df["MWFLOW"], errors="coerce")
    df = df.dropna(subset=["SETTLEMENTDATE", "MWFLOW"])
    df.to_csv(cache_csv, index=False)
    print(f"    {len(df):,} rows saved for {year}-{month:02d}")
    return df


# ---------------------------------------------------------------------------
# Section 4: Aggregate and merge
# ---------------------------------------------------------------------------

def concat_monthly(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_30min(df: pd.DataFrame, time_col: str, value_col: str, agg: str = "mean") -> pd.Series:
    """Resample a fine-grained series to 30-minute, indexed by settlement_datetime."""
    df = df[[time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    if agg == "mean":
        rs = df[value_col].resample(RESAMPLE_FREQ).mean()
    else:
        rs = df[value_col].resample(RESAMPLE_FREQ).sum(min_count=1)
    rs.index.name = "settlement_datetime"
    return rs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SA NEM Data Download — 2023 to 2025")
    print("=" * 60)

    months = list(month_range(DATE_START, DATE_END))
    weeks  = list(week_range(DATE_START, DATE_END, chunk_days=7))
    print(f"Months to process: {len(months)}  |  7-day chunks (OE API): {len(weeks)}")

    # -----------------------------------------------------------------------
    # 1. SA1 market (price + demand)  — chunked by week for 5m interval
    # -----------------------------------------------------------------------
    print(f"\n[1/6] SA1 market data (price + demand) — {len(weeks)} chunks ...")
    sa1_market = concat_monthly([
        fetch_market_chunk("SA1", cs, ce)
        for cs, ce in tqdm(weeks, desc="SA1 market")
    ])
    print(f"  Total raw rows: {len(sa1_market):,}  cols: {list(sa1_market.columns)}")

    # -----------------------------------------------------------------------
    # 2. VIC1 market (price only)  — chunked by week
    # -----------------------------------------------------------------------
    print(f"\n[2/6] VIC1 market data (price) — {len(weeks)} chunks ...")
    vic1_market = concat_monthly([
        fetch_market_chunk("VIC1", cs, ce)
        for cs, ce in tqdm(weeks, desc="VIC1 market")
    ])
    print(f"  Total raw rows: {len(vic1_market):,}  cols: {list(vic1_market.columns)}")

    # -----------------------------------------------------------------------
    # 3. SA1 wind  — chunked by week
    # -----------------------------------------------------------------------
    print(f"\n[3/6] SA1 wind power — {len(weeks)} chunks ...")
    wind_raw = concat_monthly([
        fetch_power_chunk("wind", cs, ce)
        for cs, ce in tqdm(weeks, desc="SA1 wind")
    ])
    print(f"  Total raw rows: {len(wind_raw):,}  cols: {list(wind_raw.columns)}")

    # -----------------------------------------------------------------------
    # 4. SA1 solar_utility  — chunked by week
    # -----------------------------------------------------------------------
    print(f"\n[4/6] SA1 solar_utility power — {len(weeks)} chunks ...")
    su_raw = concat_monthly([
        fetch_power_chunk("solar_utility", cs, ce)
        for cs, ce in tqdm(weeks, desc="SA1 solar_utility")
    ])
    print(f"  Total raw rows: {len(su_raw):,}  cols: {list(su_raw.columns)}")

    # -----------------------------------------------------------------------
    # 5. SA1 solar_rooftop  — chunked by week
    # -----------------------------------------------------------------------
    print(f"\n[5/6] SA1 solar_rooftop power — {len(weeks)} chunks ...")
    sr_raw = concat_monthly([
        fetch_power_chunk("solar_rooftop", cs, ce)
        for cs, ce in tqdm(weeks, desc="SA1 solar_rooftop")
    ])
    print(f"  Total raw rows: {len(sr_raw):,}  cols: {list(sr_raw.columns)}")

    # -----------------------------------------------------------------------
    # 6. NEMWEB interconnector (V-SA)  — monthly MMSDM ZIPs
    # -----------------------------------------------------------------------
    print(f"\n[6/6] V-SA interconnector flows (NEMWEB MMSDM) — {len(months)} months ...")
    ic_raw = concat_monthly([
        fetch_interconnector_month(ms.year, ms.month)
        for ms, _ in tqdm(months, desc="V-SA interconnector")
    ])
    print(f"  Total raw rows: {len(ic_raw):,}  cols: {list(ic_raw.columns)}")

    # -----------------------------------------------------------------------
    # Resample to 30-minute and build master index
    # -----------------------------------------------------------------------
    print("\n[Merge] Resampling to 30-minute intervals ...")

    idx = pd.date_range(DATE_START, DATE_END, freq=RESAMPLE_FREQ, name="settlement_datetime")
    master = pd.DataFrame(index=idx)

    def attach(series_or_none, col_name: str):
        if series_or_none is None or series_or_none.empty:
            master[col_name] = float("nan")
        else:
            master[col_name] = series_or_none.reindex(master.index)

    # -- SA1 price
    if not sa1_market.empty and "price" in sa1_market.columns:
        grp = sa1_market.groupby("interval")["price"].mean().reset_index()
        attach(to_30min(grp, "interval", "price"), "sa_price")
    else:
        master["sa_price"] = float("nan")
        print("  WARNING: sa1_market has no 'price' column")

    # -- SA1 demand
    if not sa1_market.empty and "demand" in sa1_market.columns:
        grp = sa1_market.groupby("interval")["demand"].mean().reset_index()
        attach(to_30min(grp, "interval", "demand"), "demand_mw")
    else:
        master["demand_mw"] = float("nan")
        print("  WARNING: sa1_market has no 'demand' column")

    # -- VIC1 price
    if not vic1_market.empty and "price" in vic1_market.columns:
        grp = vic1_market.groupby("interval")["price"].mean().reset_index()
        attach(to_30min(grp, "interval", "price"), "vic_price")
    else:
        master["vic_price"] = float("nan")
        print("  WARNING: vic1_market has no 'price' column")

    # -- Wind
    if not wind_raw.empty and "power" in wind_raw.columns:
        grp = wind_raw.groupby("interval")["power"].sum(min_count=1).reset_index()
        attach(to_30min(grp, "interval", "power"), "wind_mw")
    else:
        master["wind_mw"] = float("nan")
        print("  WARNING: wind_raw has no 'power' column")

    # -- Solar utility
    if not su_raw.empty and "power" in su_raw.columns:
        grp = su_raw.groupby("interval")["power"].sum(min_count=1).reset_index()
        attach(to_30min(grp, "interval", "power"), "solar_utility_mw")
    else:
        master["solar_utility_mw"] = float("nan")
        print("  WARNING: su_raw has no 'power' column")

    # -- Solar rooftop
    if not sr_raw.empty and "power" in sr_raw.columns:
        grp = sr_raw.groupby("interval")["power"].sum(min_count=1).reset_index()
        attach(to_30min(grp, "interval", "power"), "solar_rooftop_mw")
    else:
        master["solar_rooftop_mw"] = float("nan")
        print("  WARNING: sr_raw has no 'power' column")

    # -- Interconnector
    if not ic_raw.empty and "MWFLOW" in ic_raw.columns:
        attach(to_30min(ic_raw, "SETTLEMENTDATE", "MWFLOW"), "interconnector_mw")
    else:
        master["interconnector_mw"] = float("nan")
        print("  WARNING: ic_raw has no 'MWFLOW' column")

    # -----------------------------------------------------------------------
    # Derived variables
    # -----------------------------------------------------------------------
    print("[Derive] Adding calculated columns ...")

    su = master["solar_utility_mw"].fillna(0)
    sr = master["solar_rooftop_mw"].fillna(0)
    master["solar_total_mw"] = su + sr
    both_nan = master["solar_utility_mw"].isna() & master["solar_rooftop_mw"].isna()
    master.loc[both_nan, "solar_total_mw"] = float("nan")

    denom = master["demand_mw"].replace(0, float("nan"))
    master["renewable_penetration"] = (
        master["wind_mw"].fillna(0) + master["solar_total_mw"].fillna(0)
    ) / denom
    master.loc[master["demand_mw"].isna(), "renewable_penetration"] = float("nan")

    master["hour"]  = master.index.hour
    master["month"] = master.index.month
    master["year"]  = master.index.year
    master["date"]  = master.index.date

    master["price_flag"] = (
        (master["sa_price"] <= -1000) | (master["sa_price"] >= 15000)
    ).astype(int)
    master.loc[master["sa_price"].isna(), "price_flag"] = 0

    master = master.reset_index()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\n[Save] Writing {OUTPUT_CSV} ...")
    master.to_csv(OUTPUT_CSV, index=False)

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Rows         : {len(master):,}")
    print(f"Date range   : {master['settlement_datetime'].min()} → {master['settlement_datetime'].max()}")

    num_cols = [
        "sa_price", "vic_price", "wind_mw", "solar_utility_mw",
        "solar_rooftop_mw", "demand_mw", "interconnector_mw",
        "solar_total_mw", "renewable_penetration",
    ]
    print(f"\nMissing values (%):")
    for col in num_cols:
        if col in master.columns:
            pct = master[col].isna().mean() * 100
            print(f"  {col:<25s}: {pct:.1f}%")

    print(f"\nSA price statistics ($/MWh):")
    if "sa_price" in master.columns and master["sa_price"].notna().any():
        s = master["sa_price"]
        print(f"  min  = {s.min():.2f}")
        print(f"  mean = {s.mean():.2f}")
        print(f"  max  = {s.max():.2f}")

    print(f"\nPrice regime split:")
    if "sa_price" in master.columns:
        print(f"  sa_price < 0   : {(master['sa_price'] < 0).sum():,} intervals")
        print(f"  sa_price == 0  : {(master['sa_price'] == 0).sum():,} intervals")
        print(f"  sa_price > 0   : {(master['sa_price'] > 0).sum():,} intervals")
        print(f"  price_flag = 1 : {master['price_flag'].sum():,} intervals (cap/floor events)")

    print("\nDone.")


if __name__ == "__main__":
    main()
