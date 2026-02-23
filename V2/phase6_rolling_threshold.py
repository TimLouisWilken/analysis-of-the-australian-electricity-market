#!/usr/bin/env python3
"""
Phase 6: Time-Varying Threshold Analysis
=========================================
Tests whether the no-arbitrage band (γ_L, γ_U) in the SA-VIC price ECT
varies over time, and whether that variation correlates with the growth
of the SA battery storage fleet.

Approach:
  - Rolling sub-samples: annual (2023 / 2024 / 2025) and semi-annual (6 windows)
  - Within each window: SSR-minimising grid search over (γ_L, γ_U) pairs
    (NQ × NQ quantile grid, fixed β = 1.2103 and L = 4 for comparability)
  - Threshold-capacity OLS regression:
      γ_U ~ installed_MWh  (H: declining → batteries suppress spikes)
      γ_L ~ installed_MWh  (H: rising   → batteries reduce discount depth)
      bandwidth ~ installed_MWh  (H: narrowing → market integration)

Fixed from Phase 1 / Phase 2:
  β = 1.2103  (Johansen cointegrating vector)
  L = 4       (ECM lags, baseline specification)

Output files:
  results/phase6_rolling_thresholds.csv
  results/phase6_capacity_regression.csv
  results/phase6_summary.txt
"""

import warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from statsmodels.regression.linear_model import OLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BETA_SA   = 1.2103   # fixed Johansen cointegrating vector
ECM_LAGS  = 4        # fixed baseline lag order
HAC_LAGS  = 12

NQ        = 17       # grid points per dimension (NQ × NQ = 289 pairs per window)
MIN_R_PCT = 0.05     # minimum fraction of obs required in each extreme regime

CONTROLS = [
    "wind_mw",
    "solar_total_mw",
    "demand_mw",
    "battery_discharging_mw",
    "battery_charging_mw",
    "interconnector_mw",
]

# Battery facilities commissioned in SA — from BESS SA.xlsx
# (name, commissioning_date, capacity_MWh)
BATTERY_CAPACITY = [
    ("Hornsdale Power Reserve",        "2017-10-01",  129.0),
    ("Dalrymple North",                "2018-12-01",    8.0),
    ("Lake Bonney",                    "2019-12-01",   52.0),
    ("Hornsdale Power Reserve Ext.",   "2020-09-01",   64.5),
    ("Adelaide Desalination Plant",    "2021-05-01",   13.0),
    ("Bolivar",                        "2022-02-01",    5.0),
    ("Christies Beach",                "2022-03-01",    4.3),
    ("Happy Valley",                   "2022-06-01",    8.8),
    ("Torrens Island",                 "2023-08-01",  250.0),
    ("Tailem Bend 2",                  "2025-01-01",   84.0),
    ("Blyth",                          "2025-04-01",  477.0),
    ("Mannum",                         "2025-10-01",  200.0),
]

# Full-sample reference thresholds from Phase 1
REF_GAMMA_L = -45.8039
REF_GAMMA_U =  20.0215

DIVIDER = "=" * 74


# ---------------------------------------------------------------------------
# Capacity helper
# ---------------------------------------------------------------------------
def installed_capacity_at(date: pd.Timestamp) -> float:
    """Total MWh of SA batteries commissioned at or before `date`."""
    return sum(
        mwh for _, comm, mwh in BATTERY_CAPACITY
        if pd.Timestamp(comm) <= date
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime").sort_index()
    df = df[df["sa_price"].abs() < 15001]
    df = df[df["vic_price"].abs() < 15001]
    return df


# ---------------------------------------------------------------------------
# ECM sample construction for a single window
# ---------------------------------------------------------------------------
def build_window_sample(df_window):
    """
    Construct the ECM design matrix for a time-window slice.
    Uses diffs and lags computed within the window; first ECM_LAGS+1 rows
    are dropped via dropna().
    """
    sa   = df_window["sa_price"]
    vic  = df_window["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")
    ect  = (sa - BETA_SA * vic).rename("ECT")

    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}
    for lag in range(1, ECM_LAGS + 1):
        rows[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows[f"dvic_l{lag}"] = dvic.shift(lag)
    for c in CONTROLS:
        rows[c] = df_window[c] if c in df_window.columns else np.nan

    return pd.DataFrame(rows).dropna()


# ---------------------------------------------------------------------------
# Grid search for optimal (γ_L, γ_U)
# ---------------------------------------------------------------------------
def grid_search_thresholds(sample):
    """
    Minimise total OLS sum-of-squared-residuals (dsa equation) over a
    NQ × NQ grid of (γ_L, γ_U) quantile pairs.

    Constraints:
      - γ_U > γ_L + 1.0  (non-degenerate band)
      - Each extreme regime has at least max(5%, 10 obs) observations
      - Band regime has at least 20% of observations

    Returns
    -------
    gamma_L : float
    gamma_U : float
    sizes   : tuple(n_R1, n_R2, n_R3)
    min_ssr : float
    """
    ect_lag = sample["ECT_lag"].values
    n       = len(ect_lag)
    min_n   = max(int(MIN_R_PCT * n), 10)
    min_n2  = int(0.20 * n)

    # Build design matrix once (same for all threshold pairs)
    lag_names = ([f"dsa_l{i}" for i in range(1, ECM_LAGS + 1)] +
                 [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)])
    x_cols  = ["ECT_lag"] + lag_names + CONTROLS
    X_full  = np.column_stack(
        [np.ones(n)] + [sample[c].values for c in x_cols]
    )
    y_full  = sample["dsa"].values

    lo_vals = np.quantile(ect_lag, np.linspace(0.05, 0.40, NQ))
    hi_vals = np.quantile(ect_lag, np.linspace(0.60, 0.95, NQ))

    best_ssr   = np.inf
    best_gL    = np.quantile(ect_lag, 0.10)
    best_gU    = np.quantile(ect_lag, 0.90)
    best_sizes = (None, None, None)

    for gL in lo_vals:
        for gU in hi_vals:
            if gU <= gL + 1.0:
                continue

            m1 = ect_lag < gL
            m2 = (ect_lag >= gL) & (ect_lag <= gU)
            m3 = ect_lag > gU

            if m1.sum() < min_n or m3.sum() < min_n or m2.sum() < min_n2:
                continue

            ssr = 0.0
            ok  = True
            for mask in (m1, m2, m3):
                X_r = X_full[mask]
                y_r = y_full[mask]
                try:
                    beta  = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
                    resid = y_r - X_r @ beta
                    ssr  += float(resid @ resid)
                except np.linalg.LinAlgError:
                    ok = False
                    break

            if ok and ssr < best_ssr:
                best_ssr   = ssr
                best_gL    = gL
                best_gU    = gU
                best_sizes = (int(m1.sum()), int(m2.sum()), int(m3.sum()))

    return best_gL, best_gU, best_sizes, best_ssr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(DIVIDER)
    print("  PHASE 6: TIME-VARYING THRESHOLD ANALYSIS")
    print(f"  Fixed: β = {BETA_SA}   ECM lags = {ECM_LAGS}")
    print(f"  Grid: {NQ} × {NQ} = {NQ**2} threshold pairs per window")
    print(f"  Min regime fraction: {int(MIN_R_PCT*100)}%")
    print(DIVIDER)

    df = load_data()
    print(f"\n  Data loaded: {len(df):,} observations  "
          f"({df.index.min().date()} to {df.index.max().date()})")

    # ── Define windows ───────────────────────────────────────────────────
    windows = []
    for yr in [2023, 2024, 2025]:
        windows.append({"label": f"{yr}",    "type": "annual",
                         "start": f"{yr}-01-01", "end": f"{yr}-12-31 23:59"})
    for yr in [2023, 2024, 2025]:
        windows.append({"label": f"{yr} H1", "type": "semi",
                         "start": f"{yr}-01-01", "end": f"{yr}-06-30 23:59"})
        windows.append({"label": f"{yr} H2", "type": "semi",
                         "start": f"{yr}-07-01", "end": f"{yr}-12-31 23:59"})

    # ── Grid search for each window ──────────────────────────────────────
    print(f"\n  Running grid search for {len(windows)} windows ...\n")
    print(f"  {'Window':<10s}  {'N':>6s}  {'Cap (MWh)':>10s}  "
          f"{'γ_L':>8s}  {'γ_U':>8s}  {'BW':>8s}  "
          f"{'R1%':>6s}  {'R2%':>6s}  {'R3%':>6s}")
    print("  " + "─" * 80)

    rows = []
    for winfo in windows:
        start = pd.Timestamp(winfo["start"])
        end   = pd.Timestamp(winfo["end"])
        mid   = start + (end - start) / 2

        df_w   = df[(df.index >= start) & (df.index <= end)].copy()
        sample = build_window_sample(df_w)
        n      = len(sample)

        if n < 200:
            print(f"  {'SKIP '+winfo['label']:<10s}  only {n} obs after lag creation")
            continue

        cap    = installed_capacity_at(mid)
        gL, gU, sizes, ssr = grid_search_thresholds(sample)
        n1, n2, n3 = sizes if sizes[0] is not None else (0, 0, 0)
        bw = gU - gL

        print(f"  {winfo['label']:<10s}  {n:>6d}  {cap:>10.0f}  "
              f"{gL:>8.2f}  {gU:>8.2f}  {bw:>8.2f}  "
              f"{n1/n*100:>6.1f}%  {n2/n*100:>6.1f}%  {n3/n*100:>6.1f}%")

        rows.append({
            "label"    : winfo["label"],
            "type"     : winfo["type"],
            "start"    : str(start.date()),
            "end"      : str(end.date()),
            "n_sample" : n,
            "cap_mwh"  : cap,
            "gamma_L"  : gL,
            "gamma_U"  : gU,
            "bandwidth": bw,
            "n_R1"     : n1,
            "n_R2"     : n2,
            "n_R3"     : n3,
            "pct_R1"   : n1 / n * 100 if n > 0 else np.nan,
            "pct_R2"   : n2 / n * 100 if n > 0 else np.nan,
            "pct_R3"   : n3 / n * 100 if n > 0 else np.nan,
            "ssr"      : ssr,
        })

    if not rows:
        print("\n  ERROR: No valid windows found. Check data coverage.")
        return

    df_roll = pd.DataFrame(rows)
    df_roll.to_csv(RESULTS_DIR / "phase6_rolling_thresholds.csv", index=False)
    print(f"\n  Saved: results/phase6_rolling_thresholds.csv")

    # ── Annual evolution summary ─────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  ANNUAL THRESHOLD EVOLUTION")
    print(DIVIDER)
    print(f"\n  Full-sample (Phase 1):  γ_L = {REF_GAMMA_L:.2f}  "
          f"γ_U = {REF_GAMMA_U:.2f}  BW = {REF_GAMMA_U - REF_GAMMA_L:.2f}")
    print(f"\n  {'Year':<8s}  {'Cap (MWh)':>10s}  "
          f"{'γ_L':>8s}  {'γ_U':>8s}  {'BW':>8s}  "
          f"{'ΔBWΔ vs FS':>11s}  {'R1%':>6s}  {'R3%':>6s}")
    print("  " + "─" * 76)

    df_ann = df_roll[df_roll["type"] == "annual"].copy()
    ref_bw = REF_GAMMA_U - REF_GAMMA_L
    for _, row in df_ann.sort_values("start").iterrows():
        dbw = row["bandwidth"] - ref_bw
        print(f"  {row['label']:<8s}  {row['cap_mwh']:>10.0f}  "
              f"{row['gamma_L']:>8.2f}  {row['gamma_U']:>8.2f}  "
              f"{row['bandwidth']:>8.2f}  {dbw:>+11.2f}  "
              f"{row['pct_R1']:>6.1f}%  {row['pct_R3']:>6.1f}%")

    # ── Semi-annual evolution ────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  SEMI-ANNUAL THRESHOLD EVOLUTION")
    print(DIVIDER)
    print(f"\n  {'Window':<10s}  {'Cap (MWh)':>10s}  "
          f"{'γ_L':>8s}  {'γ_U':>8s}  {'BW':>8s}  "
          f"{'R1%':>6s}  {'R3%':>6s}")
    print("  " + "─" * 66)

    df_semi = df_roll[df_roll["type"] == "semi"].copy()
    for _, row in df_semi.sort_values("start").iterrows():
        print(f"  {row['label']:<10s}  {row['cap_mwh']:>10.0f}  "
              f"{row['gamma_L']:>8.2f}  {row['gamma_U']:>8.2f}  "
              f"{row['bandwidth']:>8.2f}  "
              f"{row['pct_R1']:>6.1f}%  {row['pct_R3']:>6.1f}%")

    # ── Threshold-capacity regression ────────────────────────────────────
    print("\n" + DIVIDER)
    print("  THRESHOLD-CAPACITY OLS REGRESSION")
    print("  H₀: slope = 0   (threshold independent of battery fleet size)")
    print("  H₁: γ_U declines, γ_L rises, bandwidth narrows with capacity")
    print(DIVIDER)

    def stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    cap_rows = []
    for dep_name, dep_col in [("gamma_U",   "gamma_U"),
                               ("gamma_L",   "gamma_L"),
                               ("bandwidth", "bandwidth")]:
        print(f"\n  {dep_name}:")
        for wtype, label in [("annual", "annual (N=3)"), ("semi", "semi-annual (N=6)")]:
            sub = df_roll[df_roll["type"] == wtype].dropna(subset=["cap_mwh", dep_col])
            n_w = len(sub)
            if n_w < 3:
                print(f"    {label}: too few windows (N={n_w})")
                continue

            x    = sub["cap_mwh"].values
            y    = sub[dep_col].values
            X_r  = np.column_stack([np.ones_like(x), x])
            res  = OLS(y, X_r).fit()
            b    = res.params[1]
            se   = res.bse[1]
            pv   = res.pvalues[1]
            corr = np.corrcoef(x, y)[0, 1]
            st   = stars(pv)

            print(f"    {label}: slope = {b:+.4f}  (se={se:.4f})  "
                  f"p={pv:.4f}{st}  r={corr:+.3f}  R²={res.rsquared:.3f}")

            cap_rows.append({
                "dep_var"     : dep_name,
                "window_type" : wtype,
                "n_windows"   : n_w,
                "intercept"   : res.params[0],
                "slope"       : b,
                "slope_se"    : se,
                "slope_p"     : pv,
                "slope_stars" : st,
                "pearson_r"   : corr,
                "rsquared"    : res.rsquared,
            })

    cap_df = pd.DataFrame(cap_rows)
    cap_df.to_csv(RESULTS_DIR / "phase6_capacity_regression.csv", index=False)
    print(f"\n  Saved: results/phase6_capacity_regression.csv")

    # ── Summary text ──────────────────────────────────────────────────────
    summary_lines = [
        "PHASE 6 — TIME-VARYING THRESHOLD ANALYSIS",
        "=" * 60,
        f"Fixed: beta={BETA_SA}  ECM lags={ECM_LAGS}",
        f"Grid: {NQ}×{NQ} quantile pairs  min_regime_pct={int(MIN_R_PCT*100)}%",
        f"Full-sample reference (Phase 1): "
        f"gamma_L={REF_GAMMA_L:.2f}  gamma_U={REF_GAMMA_U:.2f}  "
        f"BW={REF_GAMMA_U-REF_GAMMA_L:.2f}",
        "",
        "Rolling threshold estimates (sorted by date):",
    ]
    for _, row in df_roll.sort_values("start").iterrows():
        summary_lines.append(
            f"  {row['label']:<12s}  cap={row['cap_mwh']:.0f} MWh  "
            f"gamma_L={row['gamma_L']:.2f}  gamma_U={row['gamma_U']:.2f}  "
            f"BW={row['bandwidth']:.2f}  "
            f"R1={row['pct_R1']:.1f}%  R3={row['pct_R3']:.1f}%"
        )

    summary_lines += ["", "Threshold-capacity regression:"]
    for _, row in cap_df.iterrows():
        summary_lines.append(
            f"  {row['dep_var']} ~ cap ({row['window_type']}  N={row['n_windows']}): "
            f"slope={row['slope']:+.4f} (se={row['slope_se']:.4f}) "
            f"p={row['slope_p']:.4f}{row['slope_stars']}  "
            f"r={row['pearson_r']:+.3f}"
        )

    (RESULTS_DIR / "phase6_summary.txt").write_text("\n".join(summary_lines))
    print(f"  Saved: results/phase6_summary.txt")

    print("\n" + DIVIDER)
    print("  INTERPRETATION NOTES")
    print(DIVIDER)
    print(f"""
  Economic hypothesis: a growing battery fleet arbitrages inter-regional
  price gaps more aggressively, compressing the no-arbitrage band.
    → γ_U declining: fewer SA price spikes above the upper threshold
    → γ_L rising: fewer SA discount episodes below the lower threshold
    → Bandwidth narrowing: tighter SA-VIC price integration

  Grid search note: thresholds are SSR-minimising for the dsa equation
  within each window. They are not equivalent to Phase 1 Sup-F thresholds
  (different objective, no bootstrap p-values). Treat as descriptive evidence.

  Capacity regression caveat: with N = 3 (annual) or N = 6 (semi-annual)
  windows, p-values have low power and may not reach conventional thresholds.
  Report direction of slope (sign) and Pearson r as the primary evidence.
  Consistent sign across annual and semi-annual windows is stronger support.

  For the paper: use Phase 6 as a time-series complement to the Phase 3
  counterfactual (static welfare) and Phase 4 IV (causal identification).
  Capacity-threshold correlation provides structural support for the finding
  that batteries are integrating SA into the wider NEM.
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
