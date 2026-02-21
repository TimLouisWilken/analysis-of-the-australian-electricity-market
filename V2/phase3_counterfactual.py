#!/usr/bin/env python3
"""
Phase 3: Counterfactual Simulation — Battery Dampening Effect
=============================================================

Stage 1: Re-estimate TVECM with lagged battery variables
         (battery_dis_{t-1}, battery_chg_{t-1}) — predetermined at time of
         price formation, strictly exogenous.  Expected result: discharge
         coefficient turns negative in Regime 3 (SA premium).

Stage 2: Static counterfactual — remove cumulative battery contribution
         from the actual SA price path to recover the "no-battery" level.

         battery_contribution_t = β_dis_r(t) × battery_dis_{t-1}
                                 + β_chg_r(t) × battery_chg_{t-1}
         sa_CF_t = sa_actual_t − Σ_{s≤t} battery_contribution_s

         (r(t) is the actual regime at t, from Phase 2 thresholds on ECT_{t-1})

Outputs:
  results/phase3_lagged_dsa_coefs.csv
  results/phase3_lagged_dvic_coefs.csv
  results/phase3_counterfactual.csv
  results/phase3_summary.txt
  results/phase3_coef_comparison.csv
"""

import warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS

# ---------------------------------------------------------------------------
# Configuration — matches Phase 2 exactly
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GAMMA_L = -45.8039
GAMMA_U =  20.0215

ECM_LAGS = 4
HAC_LAGS = 12

CONTROLS = [
    "wind_mw",
    "solar_total_mw",
    "demand_mw",
    "battery_discharging_mw",
    "battery_charging_mw",
    "interconnector_mw",
]
BATTERY_COLS = {"battery_discharging_mw", "battery_charging_mw"}

DIVIDER = "=" * 74

# ---------------------------------------------------------------------------
# 1. Data loading and ECT  (unchanged from phase2_tvecm.py)
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime").sort_index()
    df = df[df["sa_price"].abs() < 15001]
    df = df[df["vic_price"].abs() < 15001]
    return df


def estimate_ect(df):
    prices = df[["sa_price", "vic_price"]].dropna()
    res  = coint_johansen(prices.values, det_order=0, k_ar_diff=ECM_LAGS)
    beta = res.evec[:, 0] / res.evec[0, 0]   # normalise [1, -β]
    ect  = prices.values @ beta
    ect  = pd.Series(ect, index=prices.index, name="ECT")
    return beta, ect


# ---------------------------------------------------------------------------
# 2. Build sample — battery cols lagged 1 period (Stage 1 modification)
# ---------------------------------------------------------------------------
def build_sample_lagged(df, ect):
    """
    Identical to phase2 build_sample() except that battery_discharging_mw
    and battery_charging_mw are shifted 1 period (predetermined regressors).
    Also retains sa_actual and demand_mw as pass-through columns for the
    counterfactual and welfare calculations.
    """
    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")

    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}

    for lag in range(1, ECM_LAGS + 1):
        rows[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows[f"dvic_l{lag}"] = dvic.shift(lag)

    # Battery cols lagged 1; all other controls contemporaneous
    for c in CONTROLS:
        if c in df.columns:
            rows[c] = df[c].shift(1) if c in BATTERY_COLS else df[c]
        else:
            rows[c] = np.nan
            print(f"  WARNING: control '{c}' not found — set to NaN")

    # Pass-through columns (not used as regressors)
    rows["sa_actual"] = df["sa_price"]

    sample = pd.DataFrame(rows).dropna()
    sample["regime"] = np.where(
        sample["ECT_lag"] < GAMMA_L, 1,
        np.where(sample["ECT_lag"] > GAMMA_U, 3, 2)
    )
    return sample


# ---------------------------------------------------------------------------
# 3. OLS helpers  (unchanged from phase2_tvecm.py)
# ---------------------------------------------------------------------------
def regressor_names():
    lag_names = (
        [f"dsa_l{i}"  for i in range(1, ECM_LAGS + 1)] +
        [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)]
    )
    return ["const", "ECT_lag"] + lag_names + CONTROLS


def build_X(sample_r: pd.DataFrame) -> np.ndarray:
    cols = regressor_names()
    X = sample_r[cols[1:]].copy()
    X.insert(0, "const", 1.0)
    return X.values.astype(float)


def fit_ols_hac(y: np.ndarray, X: np.ndarray):
    return OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )


def stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def estimate_equation(sample: pd.DataFrame, dep_var: str) -> dict:
    results = {}
    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r]
        y   = sub[dep_var].values.astype(float)
        X   = build_X(sub)
        results[r] = fit_ols_hac(y, X)
    return results


def build_interacted(sample: pd.DataFrame, dep_var: str):
    k = len(regressor_names())
    rows, y_all = [], []
    for r in [1, 2, 3]:
        sub  = sample[sample["regime"] == r]
        y_r  = sub[dep_var].values.astype(float)
        X_r  = build_X(sub)
        T_r  = len(y_r)
        X_full_r = np.zeros((T_r, 3 * k))
        X_full_r[:, (r - 1) * k : r * k] = X_r
        rows.append(X_full_r)
        y_all.append(y_r)
    return np.concatenate(y_all), np.vstack(rows), k


def wald_test_equal(result_full, k: int, param_idx: int):
    n_params = result_full.params.shape[0]
    R = np.zeros((2, n_params))
    R[0, 0 * k + param_idx] =  1.0
    R[0, 1 * k + param_idx] = -1.0
    R[1, 1 * k + param_idx] =  1.0
    R[1, 2 * k + param_idx] = -1.0
    wt = result_full.wald_test(R, use_f=True, scalar=True)
    return float(wt.statistic), float(wt.pvalue)


def save_coef_csv(ols_results: dict, var_names: list, dep_var: str, prefix: str):
    rows = []
    for r in [1, 2, 3]:
        res = ols_results[r]
        for i, vn in enumerate(var_names):
            rows.append({
                "equation": dep_var,
                "regime":   r,
                "variable": vn,
                "coef":     res.params[i],
                "hac_se":   res.bse[i],
                "t_stat":   res.tvalues[i],
                "p_value":  res.pvalues[i],
                "stars":    stars(res.pvalues[i]),
            })
    fname = RESULTS_DIR / f"{prefix}_{dep_var}_coefs.csv"
    pd.DataFrame(rows).to_csv(fname, index=False)
    return fname


# ---------------------------------------------------------------------------
# 4. Static counterfactual  (Stage 2)
# ---------------------------------------------------------------------------
def run_counterfactual(sample: pd.DataFrame, sa_results: dict,
                       var_names: list):
    """
    Compute per-period battery contribution to Δsa and recover the
    counterfactual SA price level.

    battery_dis and battery_chg in `sample` are already t-1 values
    (shifted in build_sample_lagged), so they are the correct predetermined
    regressors used in estimation.
    """
    batt_idx_dis = var_names.index("battery_discharging_mw")
    batt_idx_chg = var_names.index("battery_charging_mw")

    beta_dis = {r: sa_results[r].params[batt_idx_dis] for r in [1, 2, 3]}
    beta_chg = {r: sa_results[r].params[batt_idx_chg] for r in [1, 2, 3]}

    cf = pd.DataFrame({
        "regime":          sample["regime"],
        "sa_actual":       sample["sa_actual"],
        "dsa_actual":      sample["dsa"],
        "battery_dis_lag": sample["battery_discharging_mw"],
        "battery_chg_lag": sample["battery_charging_mw"],
        "demand_mw":       sample["demand_mw"],
    })

    # Vectorised contribution by regime
    bc = pd.Series(np.nan, index=cf.index)
    for r in [1, 2, 3]:
        mask    = cf["regime"] == r
        bc[mask] = (
            beta_dis[r] * cf.loc[mask, "battery_dis_lag"] +
            beta_chg[r] * cf.loc[mask, "battery_chg_lag"]
        )
    cf["battery_contribution"] = bc

    # Counterfactual price level: remove cumulative battery contribution
    cf["sa_CF"] = cf["sa_actual"] - cf["battery_contribution"].cumsum()

    return cf, beta_dis, beta_chg


# ---------------------------------------------------------------------------
# 5. Summary statistics
# ---------------------------------------------------------------------------
def build_summary(cf: pd.DataFrame) -> str:
    lines = []
    add   = lines.append

    add("PHASE 3 COUNTERFACTUAL SUMMARY")
    add("=" * 60)
    add(f"Estimation sample: {cf.index[0]}  to  {cf.index[-1]}")
    add(f"Total half-hour periods: {len(cf):,}")
    sample_years = (cf.index[-1] - cf.index[0]).total_seconds() / (365.25 * 24 * 3600)
    add(f"Sample length: {sample_years:.2f} years")
    add("")

    # 1. Mean dampening by regime
    add("1. MEAN BATTERY CONTRIBUTION TO Δsa ($/MWh) BY REGIME")
    add("   Negative = batteries dampen prices (lower Δsa → lower level over time)")
    add("-" * 60)
    for r in [1, 2, 3]:
        sub = cf[cf["regime"] == r]
        m   = sub["battery_contribution"].mean()
        s   = sub["battery_contribution"].std()
        n   = len(sub)
        add(f"   Regime {r} (N={n:,}): mean = {m:+.4f}  std = {s:.4f}  $/MWh")
    add(f"   Full sample:        mean = {cf['battery_contribution'].mean():+.4f}  $/MWh")
    add("")

    # 2. Spike reduction
    SPIKE = 300.0
    add(f"2. PRICE SPIKE REDUCTION  (threshold = ${SPIKE:.0f}/MWh)")
    add("-" * 60)
    T      = len(cf)
    n_act  = (cf["sa_actual"] > SPIKE).sum()
    # Per-period counterfactual: remove this period's marginal battery contribution
    # (interpretable as: how many extra spikes would have occurred each half-hour
    #  if batteries had been absent that one period, holding all else fixed)
    sa_marg = cf["sa_actual"] - cf["battery_contribution"]
    n_marg  = (sa_marg > SPIKE).sum()
    delta_m = n_marg - n_act
    add(f"   Actual spikes:                         {n_act:,}  ({n_act/T*100:.2f}%)")
    add(f"   Per-period counterfactual spikes:      {n_marg:,}  ({n_marg/T*100:.2f}%)")
    add(f"   Marginal spike reduction by batteries: {delta_m:+,}  periods ({delta_m/T*100:+.2f} pp)")
    # Cumulative level note
    cum_end = cf["battery_contribution"].sum()
    add(f"")
    add(f"   Note on cumulative counterfactual (sa_CF):")
    add(f"   The cumulative battery price reduction over the {sample_years:.1f}-year sample")
    add(f"   totals {cum_end:+.0f} $/MWh — i.e. SA prices would have been higher")
    add(f"   by this integrated amount in the absence of batteries from day 1.")
    add(f"   The per-period figure above is more interpretable for spike analysis.")
    add("")

    # 3. Volatility
    add("3. VOLATILITY  std(Δsa_actual) vs std(Δsa_CF)  [$/MWh]")
    add("-" * 60)
    dsa_cf = cf["dsa_actual"] - cf["battery_contribution"]
    add(f"   std(Δsa_actual): {cf['dsa_actual'].std():.4f}")
    add(f"   std(Δsa_CF):     {dsa_cf.std():.4f}")
    add(f"   Reduction:       {cf['dsa_actual'].std() - dsa_cf.std():+.4f}")
    add("")
    add("   By regime:")
    for r in [1, 2, 3]:
        sub   = cf[cf["regime"] == r]
        s_act = sub["dsa_actual"].std()
        s_cf  = (sub["dsa_actual"] - sub["battery_contribution"]).std()
        add(f"   Regime {r}: std(Δsa)={s_act:.4f}  std(Δsa_CF)={s_cf:.4f}")
    add("")

    # 4. Welfare estimate
    add("4. WELFARE ESTIMATE  (consumer surplus proxy)")
    add("-" * 60)
    # price_reduction > 0 means batteries lowered price → consumer benefit
    price_reduction     = -cf["battery_contribution"]
    welfare_per_period  = price_reduction * cf["demand_mw"] * 0.5  # $ per half-hour
    welfare_total_M     = welfare_per_period.sum() / 1e6
    welfare_annual_M    = welfare_total_M / sample_years
    avg_demand          = cf["demand_mw"].mean()
    add(f"   Average SA demand:               {avg_demand:.1f} MW")
    add(f"   Total welfare gain (sample):    ${welfare_total_M:+.2f}M")
    add(f"   Annualised welfare gain:        ${welfare_annual_M:+.2f}M / year")
    add(f"   Note: positive = batteries lowered prices (consumer surplus gain)")
    add(f"         negative = batteries raised prices  (net cost to consumers)")
    add("")

    # 5. Hourly intraday profile
    add("5. MEAN BATTERY CONTRIBUTION BY HOUR OF DAY ($/MWh)")
    add("-" * 60)
    cf_h = cf.copy()
    cf_h["hour"] = cf_h.index.hour
    hourly = cf_h.groupby("hour")["battery_contribution"].mean()
    add(f"   {'Hour':>5s}  {'Mean contrib ($/MWh)':>22s}  bar")
    for h, v in hourly.items():
        bar_len = min(int(abs(v) * 2), 30)
        bar = ("-" if v < 0 else "+") * bar_len
        add(f"   {h:>5d}  {v:>+22.4f}  {bar}")
    add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Coefficient comparison table  (Phase 2 contemporaneous vs Phase 3 lagged)
# ---------------------------------------------------------------------------
def build_coef_comparison(sa_results_p3: dict, var_names: list) -> pd.DataFrame:
    """Load Phase 2 CSV and merge with Phase 3 results side-by-side."""
    p2 = pd.read_csv(RESULTS_DIR / "phase2_dsa_coefs.csv")

    rows = []
    for r in [1, 2, 3]:
        for vname in ["battery_discharging_mw", "battery_charging_mw"]:
            idx = var_names.index(vname)

            p2_row = p2[(p2["regime"] == r) & (p2["variable"] == vname)]
            p2_c   = float(p2_row["coef"].iloc[0])  if len(p2_row) else np.nan
            p2_s   = str(p2_row["stars"].iloc[0])    if len(p2_row) else ""

            p3_c   = float(sa_results_p3[r].params[idx])
            p3_s   = stars(sa_results_p3[r].pvalues[idx])

            rows.append({
                "regime":        r,
                "variable":      vname,
                "phase2_coef":   p2_c,
                "phase2_stars":  p2_s,
                "phase3_coef":   p3_c,
                "phase3_stars":  p3_s,
                "sign_flipped":  (np.sign(p2_c) != np.sign(p3_c)),
                "comment":       (
                    "causal dampening identified"   if (vname == "battery_discharging_mw" and p3_c < 0 and r == 3)
                    else "simultaneity removed"     if np.sign(p2_c) != np.sign(p3_c)
                    else "sign unchanged after lag"
                ),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    print(DIVIDER)
    print("  PHASE 3: COUNTERFACTUAL SIMULATION — BATTERY DAMPENING EFFECT")
    print(f"  Thresholds: γ_L = {GAMMA_L:.4f},  γ_U = {GAMMA_U:.4f}  (ECT units $/MWh)")
    print(DIVIDER)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data and computing ECT ...")
    df        = load_data()
    beta, ect = estimate_ect(df)
    print(f"  Cointegrating vector: sa_price − {-beta[1]:.4f} × vic_price")
    print(f"  Obs (after price-cap filter): {len(df):,}")

    # ── 2. Build sample with lagged battery ──────────────────────────────
    print("\n[2] Building sample (battery_dis_{t-1}, battery_chg_{t-1}) ...")
    sample = build_sample_lagged(df, ect)
    T      = len(sample)
    print(f"  ECM sample (after lag creation + dropna): T = {T:,}")
    for r in [1, 2, 3]:
        n = (sample["regime"] == r).sum()
        print(f"    Regime {r}: N = {n:,}  ({n/T*100:.1f}%)")

    var_names = regressor_names()

    # ── 3. SA equation ───────────────────────────────────────────────────
    print("\n[3] Estimating SA equation (Δsa) with lagged battery ...")
    sa_results = estimate_equation(sample, "dsa")

    batt_idx_dis = var_names.index("battery_discharging_mw")
    batt_idx_chg = var_names.index("battery_charging_mw")
    ect_idx      = var_names.index("ECT_lag")

    print(f"\n  {'Variable':<30s}  {'Regime 1':>16s}  {'Regime 2':>16s}  {'Regime 3':>16s}")
    print("  " + "-" * 84)
    for idx, vname in [(ect_idx,      "ECT_lag (α)"),
                       (batt_idx_dis, "battery_discharging_mw"),
                       (batt_idx_chg, "battery_charging_mw")]:
        row = f"  {vname:<30s}"
        for r in [1, 2, 3]:
            c  = sa_results[r].params[idx]
            p  = sa_results[r].pvalues[idx]
            se = sa_results[r].bse[idx]
            row += f"  {c:>+8.4f}{stars(p):<3s}({se:.4f})"
        print(row)

    # Verification gate
    beta_dis_r3 = sa_results[3].params[batt_idx_dis]
    print(f"\n  VERIFICATION: battery_discharging_mw Regime 3 = {beta_dis_r3:+.4f}")
    if beta_dis_r3 < 0:
        print("  ✓  Negative — lagged discharge DAMPENS SA price spikes.")
        print("     Causal battery effect identified via predetermined regressor.")
    else:
        print("  ⚠  Still POSITIVE — possible market power / strategic discharging.")
        print("     Document as robustness concern; causal identification incomplete.")

    # ── 4. VIC equation ──────────────────────────────────────────────────
    print("\n[4] Estimating VIC equation (Δvic) ...")
    vic_results = estimate_equation(sample, "dvic")

    # ── 5. Wald tests ────────────────────────────────────────────────────
    print("\n[5] Wald tests — H0: equal coefficient across regimes (SA equation)")
    y_sa, X_int, k_int = build_interacted(sample, "dsa")
    res_int_sa = OLS(y_sa, X_int).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    print(f"  {'Variable':<34s}  {'F-stat':>8s}  {'p-value':>8s}  {'':>5s}")
    print("  " + "-" * 62)
    for vname in ["ECT_lag", "battery_discharging_mw", "battery_charging_mw"]:
        idx  = var_names.index(vname)
        F, p = wald_test_equal(res_int_sa, k_int, idx)
        print(f"  {vname:<34s}  {F:>8.4f}  {p:>8.4f}  {stars(p)}")

    # ── 6. Save coefficient CSVs ─────────────────────────────────────────
    print("\n[6] Saving coefficient CSVs ...")
    f1 = save_coef_csv(sa_results,  var_names, "dsa",  "phase3_lagged")
    f2 = save_coef_csv(vic_results, var_names, "dvic", "phase3_lagged")
    print(f"  Saved: {f1.name}")
    print(f"  Saved: {f2.name}")

    # ── 7. Static counterfactual ─────────────────────────────────────────
    print("\n[7] Running static counterfactual ...")
    cf, beta_dis, beta_chg = run_counterfactual(sample, sa_results, var_names)

    print("  Regime-specific battery coefficients applied:")
    print(f"  {'Regime':<10s}  {'β_dis (t-1)':>14s}  {'β_chg (t-1)':>14s}")
    print("  " + "-" * 44)
    for r in [1, 2, 3]:
        print(f"  Regime {r:<5d}  {beta_dis[r]:>+14.4f}  {beta_chg[r]:>+14.4f}")

    print("\n  Mean battery contribution by regime ($/MWh):")
    for r in [1, 2, 3]:
        sub = cf[cf["regime"] == r]
        m   = sub["battery_contribution"].mean()
        n   = len(sub)
        pct_neg = (sub["battery_contribution"] < 0).mean()
        print(f"    Regime {r} (N={n:,}): {m:+.4f}  ({pct_neg:.1%} negative)")
    print(f"    Overall:           {cf['battery_contribution'].mean():+.4f}")

    # Spot checks
    r3 = cf[cf["regime"] == 3]
    r1 = cf[cf["regime"] == 1]
    frac_neg_r3 = (r3["battery_contribution"] < 0).mean()
    frac_pos_r1 = (r1["battery_contribution"] > 0).mean()
    print(f"\n  Spot check — Regime 3: {frac_neg_r3:.1%} of contributions negative")
    print(f"               Regime 1: {frac_pos_r1:.1%} of contributions positive")
    print(f"  sa_CF > sa_actual in Regime 3: {(r3['sa_CF'] > r3['sa_actual']).mean():.1%}")

    # ── 8. Save counterfactual CSV ────────────────────────────────────────
    print("\n[8] Saving counterfactual time series ...")
    cf_out = cf[["regime", "sa_actual", "sa_CF",
                 "battery_contribution", "battery_dis_lag", "battery_chg_lag"]]
    out_path = RESULTS_DIR / "phase3_counterfactual.csv"
    cf_out.to_csv(out_path, index=True)
    print(f"  Saved: {out_path.name}  ({len(cf_out):,} rows)")

    # ── 9. Summary statistics ─────────────────────────────────────────────
    print("\n[9] Summary statistics ...")
    summary_text = build_summary(cf)
    (RESULTS_DIR / "phase3_summary.txt").write_text(summary_text)
    print(f"  Saved: results/phase3_summary.txt")
    print()
    print(summary_text)

    # ── 10. Coefficient comparison ────────────────────────────────────────
    print("\n[10] Coefficient comparison table (Phase 2 vs Phase 3) ...")
    comp_df = build_coef_comparison(sa_results, var_names)
    comp_df.to_csv(RESULTS_DIR / "phase3_coef_comparison.csv", index=False)
    print(f"  Saved: results/phase3_coef_comparison.csv")

    print()
    print(f"  {'Regime':<8s}  {'Variable':<28s}  {'Phase2 (contemp)':>18s}  "
          f"{'Phase3 (lagged)':>18s}  {'Flipped?':>10s}")
    print("  " + "-" * 90)
    for _, row in comp_df.iterrows():
        flip = "YES ←" if row["sign_flipped"] else "no"
        print(f"  {int(row['regime']):<8d}  {row['variable']:<28s}  "
              f"  {row['phase2_coef']:>+8.4f}{row['phase2_stars']:<3s}         "
              f"  {row['phase3_coef']:>+8.4f}{row['phase3_stars']:<3s}         "
              f"  {flip:>10s}")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  PHASE 3 COMPLETE")
    print(DIVIDER)
    welfare_annual = (
        (-cf["battery_contribution"] * cf["demand_mw"] * 0.5).sum()
        / 1e6
        / ((cf.index[-1] - cf.index[0]).total_seconds() / (365.25 * 24 * 3600))
    )
    print(f"""
  Key results (SA equation, lagged battery):
    Regime 1 β_dis: {beta_dis[1]:+.4f}   β_chg: {beta_chg[1]:+.4f}
    Regime 2 β_dis: {beta_dis[2]:+.4f}   β_chg: {beta_chg[2]:+.4f}
    Regime 3 β_dis: {beta_dis[3]:+.4f}   β_chg: {beta_chg[3]:+.4f}
      {"→ causal dampening confirmed" if beta_dis[3] < 0 else "→ sign unchanged — market power concern"}

  Mean battery contribution (all regimes): {cf['battery_contribution'].mean():+.4f} $/MWh
  Annualised consumer welfare:             ${welfare_annual:+.2f}M / year

  Output files:
    results/phase3_lagged_dsa_coefs.csv
    results/phase3_lagged_dvic_coefs.csv
    results/phase3_counterfactual.csv
    results/phase3_summary.txt
    results/phase3_coef_comparison.csv
""")


if __name__ == "__main__":
    main()
