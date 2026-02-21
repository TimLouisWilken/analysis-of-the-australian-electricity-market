#!/usr/bin/env python3
"""
Patch sa_nem_2023_2025.csv with battery charging/discharging data.

Adds three columns:
  - battery_discharging_mw  : battery exporting to grid (positive = generation)
  - battery_charging_mw     : battery importing from grid (positive = load)
  - battery_net_mw          : discharging - charging (positive = net injection)

Data source: OpenElectricity API, fueltech=battery_discharging / battery_charging
Region: SA1  |  Interval: 5m  |  Resampled to: 30min (mean)

Usage:
  export OE_API_KEY="your_key_here"
  python add_battery_data.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration (mirrors download_data.py)
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OE_API_KEY", "")
if not API_KEY:
    sys.exit(
        "ERROR: OE_API_KEY environment variable not set.\n"
        "Register at https://platform.openelectricity.org.au and then:\n"
        "  export OE_API_KEY='your_key_here'"
    )

OE_BASE = "https://api.openelectricity.org.au/v4"
OE_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

DATE_START    = datetime(2023, 1, 1)
DATE_END      = datetime(2025, 12, 31, 23, 59, 59)
OE_INTERVAL   = "5m"
RESAMPLE_FREQ = "30min"

BASE_DIR      = Path(__file__).parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CSV    = PROCESSED_DIR / "sa_nem_2023_2025.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers (copied from download_data.py)
# ---------------------------------------------------------------------------

def week_range(start: datetime, end: datetime, chunk_days: int = 7):
    """Yield (chunk_start, chunk_end) pairs in chunk_days-day steps."""
    cur = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days) - timedelta(seconds=1), end)
        yield cur, chunk_end
        cur = cur + timedelta(days=chunk_days)


def oe_get(endpoint: str, params: dict, label: str) -> list:
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


def parse_timeseries(data_list: list, value_key: str) -> pd.DataFrame:
    """Parse OE API timeseries data list into a flat DataFrame."""
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
    df["interval"] = (
        pd.to_datetime(df["interval"], utc=True)
        .dt.tz_convert("Australia/Sydney")
        .dt.tz_localize(None)
    )
    return df


def fetch_power_chunk(fueltech: str, chunk_start: datetime, chunk_end: datetime) -> pd.DataFrame:
    """Download power (MW) for one fueltech in SA1, one 7-day chunk. Caches to RAW_DIR."""
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


def concat_chunks(frames: list) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_30min(df: pd.DataFrame, time_col: str, value_col: str) -> pd.Series:
    """Resample a fine-grained series to 30-minute mean, indexed by settlement_datetime."""
    df = df[[time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    rs = df[value_col].resample(RESAMPLE_FREQ).mean()
    rs.index.name = "settlement_datetime"
    return rs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Add Battery Data to SA NEM Dataset")
    print("=" * 60)

    if not OUTPUT_CSV.exists():
        sys.exit(f"ERROR: processed CSV not found: {OUTPUT_CSV}")

    weeks = list(week_range(DATE_START, DATE_END, chunk_days=7))
    print(f"7-day chunks: {len(weeks)}  (× 2 fueltechs = {len(weeks)*2} API calls max)")

    # -----------------------------------------------------------------------
    # 1. Download battery_discharging
    # -----------------------------------------------------------------------
    print(f"\n[1/2] SA1 battery_discharging — {len(weeks)} chunks ...")
    dis_raw = concat_chunks([
        fetch_power_chunk("battery_discharging", cs, ce)
        for cs, ce in tqdm(weeks, desc="battery_discharging")
    ])
    print(f"  Raw rows: {len(dis_raw):,}  cols: {list(dis_raw.columns)}")

    # -----------------------------------------------------------------------
    # 2. Download battery_charging
    # -----------------------------------------------------------------------
    print(f"\n[2/2] SA1 battery_charging — {len(weeks)} chunks ...")
    chg_raw = concat_chunks([
        fetch_power_chunk("battery_charging", cs, ce)
        for cs, ce in tqdm(weeks, desc="battery_charging")
    ])
    print(f"  Raw rows: {len(chg_raw):,}  cols: {list(chg_raw.columns)}")

    # -----------------------------------------------------------------------
    # 3. Resample to 30-minute
    # -----------------------------------------------------------------------
    print("\n[Resample] Aggregating to 30-minute means ...")

    if not dis_raw.empty and "power" in dis_raw.columns:
        grp_dis = dis_raw.groupby("interval")["power"].sum(min_count=1).reset_index()
        s_dis = to_30min(grp_dis, "interval", "power")
        s_dis.name = "battery_discharging_mw"
    else:
        s_dis = None
        print("  WARNING: no battery_discharging data")

    if not chg_raw.empty and "power" in chg_raw.columns:
        grp_chg = chg_raw.groupby("interval")["power"].sum(min_count=1).reset_index()
        s_chg = to_30min(grp_chg, "interval", "power")
        s_chg.name = "battery_charging_mw"
    else:
        s_chg = None
        print("  WARNING: no battery_charging data")

    # -----------------------------------------------------------------------
    # 4. Patch the processed CSV
    # -----------------------------------------------------------------------
    print(f"\n[Patch] Loading {OUTPUT_CSV} ...")
    master = pd.read_csv(OUTPUT_CSV, parse_dates=["settlement_datetime"])
    master = master.set_index("settlement_datetime")

    if s_dis is not None:
        master["battery_discharging_mw"] = s_dis.reindex(master.index)
    else:
        master["battery_discharging_mw"] = float("nan")

    if s_chg is not None:
        master["battery_charging_mw"] = s_chg.reindex(master.index)
    else:
        master["battery_charging_mw"] = float("nan")

    # Derived: net injection (positive = net export to grid)
    master["battery_net_mw"] = (
        master["battery_discharging_mw"].fillna(0)
        - master["battery_charging_mw"].fillna(0)
    )
    both_nan = master["battery_discharging_mw"].isna() & master["battery_charging_mw"].isna()
    master.loc[both_nan, "battery_net_mw"] = float("nan")

    master = master.reset_index()

    print(f"[Save] Writing {OUTPUT_CSV} ...")
    master.to_csv(OUTPUT_CSV, index=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Rows         : {len(master):,}")
    print(f"Columns      : {list(master.columns)}")

    for col in ["battery_discharging_mw", "battery_charging_mw", "battery_net_mw"]:
        if col in master.columns:
            s = master[col]
            pct_nan = s.isna().mean() * 100
            pct_nonzero = (s.fillna(0) != 0).mean() * 100
            print(
                f"\n  {col}:"
                f"\n    missing  : {pct_nan:.1f}%"
                f"\n    non-zero : {pct_nonzero:.1f}%"
                f"\n    min      : {s.min():.3f}"
                f"\n    max      : {s.max():.3f}"
                f"\n    mean     : {s.mean():.3f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
