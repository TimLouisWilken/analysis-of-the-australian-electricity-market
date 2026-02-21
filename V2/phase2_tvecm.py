#!/usr/bin/env python3
"""
Phase 2: Three-Regime TVECM Estimation
=======================================
Threshold parameters from Phase 1 (two-threshold band test, p = 0.040):
  γ_L = -45.80  (lower threshold on ECT = sa_price - 1.2103 × vic_price)
  γ_U = +20.02  (upper threshold)

Three regimes:
  Regime 1  ECT < γ_L  : SA price well below VIC — large import pressure
  Regime 2  γ_L ≤ ECT ≤ γ_U : near-equilibrium no-arbitrage band (80 % of obs)
  Regime 3  ECT > γ_U  : SA price premium over VIC — large export pressure

Model (estimated separately per regime via OLS):

  Δsa_t = μ_r + α_r · ECT_{t-1}
        + Σᵢ₌₁⁴ φᵢʳ · Δsa_{t-i}
        + Σᵢ₌₁⁴ ψᵢʳ · Δvic_{t-i}
        + β₁ʳ · wind_mw_t
        + β₂ʳ · solar_total_mw_t
        + β₃ʳ · demand_mw_t
        + β₄ʳ · battery_discharging_mw_t
        + β₅ʳ · battery_charging_mw_t
        + β₆ʳ · interconnector_mw_t
        + ε_t

Standard errors: HAC (Newey-West, automatic bandwidth).

Also estimated: Δvic_t equation (same structure) — price leadership test.

Wald tests (HAC-robust):
  H0: α_1 = α_2 = α_3       (equal adjustment speed across regimes)
  H0: β_dis_1 = β_dis_2 = β_dis_3  (equal battery discharge effect)
  H0: β_chg_1 = β_chg_2 = β_chg_3  (equal battery charge effect)

Output: results/phase2_*.csv  +  results/phase2_summary.txt
"""

import os, warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Thresholds from Phase 1 two-threshold band test (p = 0.040)
GAMMA_L = -45.8039
GAMMA_U =  20.0215

ECM_LAGS = 4          # lags in the VECM equations
HAC_LAGS = 12         # Newey-West bandwidth (≈ 6 h at 30-min)
TRIM_PCT  = 0.10      # for regime check (match Phase 1)

REGIME_LABELS = {
    1: f"Regime 1  (ECT < {GAMMA_L:.2f})",
    2: f"Regime 2  ({GAMMA_L:.2f} ≤ ECT ≤ {GAMMA_U:.2f})",
    3: f"Regime 3  (ECT > {GAMMA_U:.2f})",
}

CONTROLS = [
    "wind_mw",
    "solar_total_mw",
    "demand_mw",
    "battery_discharging_mw",
    "battery_charging_mw",
    "interconnector_mw",
]

DIVIDER = "=" * 74

# ---------------------------------------------------------------------------
# 1. Data loading, ECT, regime assignment
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime").sort_index()

    # Remove price-cap extremes for cleaner estimation
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


def build_sample(df, ect):
    """
    Construct the full aligned sample for the TVECM estimation.
    Returns a DataFrame with all regressors and regime assignments,
    dropping any rows with NaN in any column used for estimation.
    """
    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")

    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}

    # Lags of Δsa and Δvic
    for lag in range(1, ECM_LAGS + 1):
        rows[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows[f"dvic_l{lag}"] = dvic.shift(lag)

    # Contemporaneous controls
    for c in CONTROLS:
        if c in df.columns:
            rows[c] = df[c]
        else:
            rows[c] = np.nan
            print(f"  WARNING: control '{c}' not found in data — set to NaN")

    sample = pd.DataFrame(rows).dropna()
    sample["regime"] = np.where(
        sample["ECT_lag"] < GAMMA_L, 1,
        np.where(sample["ECT_lag"] > GAMMA_U, 3, 2)
    )
    return sample


# ---------------------------------------------------------------------------
# 2. OLS estimation helpers
# ---------------------------------------------------------------------------
def regressor_names():
    """Ordered list of regressor names (excl. dependent variable, regime)."""
    lag_names = (
        [f"dsa_l{i}"  for i in range(1, ECM_LAGS + 1)] +
        [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)]
    )
    return ["const", "ECT_lag"] + lag_names + CONTROLS


def build_X(sample_r: pd.DataFrame) -> np.ndarray:
    """Design matrix for regime r observations."""
    cols = regressor_names()
    X = sample_r[cols[1:]].copy()   # ECT_lag + lags + controls
    X.insert(0, "const", 1.0)
    return X.values.astype(float)


def fit_ols_hac(y: np.ndarray, X: np.ndarray, var_names: list):
    """OLS with Newey-West HAC standard errors."""
    res = OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    return res


def stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ---------------------------------------------------------------------------
# 3. Per-equation, per-regime estimation
# ---------------------------------------------------------------------------
def estimate_equation(sample: pd.DataFrame, dep_var: str):
    """
    Estimate the TVECM for one dependent variable (dep_var = 'dsa' or 'dvic')
    across all three regimes. Returns a list of statsmodels OLS result objects.
    """
    results = {}
    var_names = regressor_names()
    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r]
        y   = sub[dep_var].values.astype(float)
        X   = build_X(sub)
        res = fit_ols_hac(y, X, var_names)
        results[r] = res
    return results


# ---------------------------------------------------------------------------
# 4. Interacted model for Wald tests
# ---------------------------------------------------------------------------
def build_interacted(sample: pd.DataFrame, dep_var: str):
    """
    Stack all three regimes into one regression with interacted regressors:
      X_full = [D1⊗X, D2⊗X, D3⊗X]
    This allows Wald tests of cross-regime restrictions.
    """
    var_names = regressor_names()
    k = len(var_names)
    rows = []
    y_all = []
    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r]
        y_r = sub[dep_var].values.astype(float)
        X_r = build_X(sub)               # (N_r, k)
        T_r = len(y_r)

        # Interacted: zero-pad for the other two regime blocks
        X_full_r = np.zeros((T_r, 3 * k))
        X_full_r[:, (r - 1) * k : r * k] = X_r
        rows.append(X_full_r)
        y_all.append(y_r)

    X_full = np.vstack(rows)
    y_full = np.concatenate(y_all)
    return y_full, X_full, k


def wald_test_equal(result_full, k: int, param_idx: int):
    """
    Test H0: coefficient at position param_idx is equal across all 3 regimes.
    param_idx is the 0-based index within a single regime's k-vector.
    Returns (F_stat, p_value, df1, df2).
    """
    n_params = result_full.params.shape[0]   # 3*k
    R = np.zeros((2, n_params))
    # Row 0: β_r1[param_idx] - β_r2[param_idx] = 0
    R[0, 0 * k + param_idx] =  1.0
    R[0, 1 * k + param_idx] = -1.0
    # Row 1: β_r2[param_idx] - β_r3[param_idx] = 0
    R[1, 1 * k + param_idx] =  1.0
    R[1, 2 * k + param_idx] = -1.0

    wt = result_full.wald_test(R, use_f=True, scalar=True)
    return float(wt.statistic), float(wt.pvalue)


# ---------------------------------------------------------------------------
# 5. Pretty-print tables
# ---------------------------------------------------------------------------
def coef_table(ols_results: dict, var_names: list, dep_label: str,
               highlight: list = None):
    """
    Print a 3-column coefficient table (one column per regime).
    highlight: variable names to visually mark (battery vars, ECT).
    """
    if highlight is None:
        highlight = []

    print(f"\n  {'Variable':<28s}", end="")
    for r in [1, 2, 3]:
        n = ols_results[r].nobs
        print(f"  {'Regime '+str(r)+' (N='+str(int(n))+')':>22s}", end="")
    print()
    print("  " + "─" * 95)

    for i, vn in enumerate(var_names):
        mark = " ◄" if vn in highlight else ""
        print(f"  {vn+mark:<30s}", end="")
        for r in [1, 2, 3]:
            res = ols_results[r]
            c, se, p = res.params[i], res.bse[i], res.pvalues[i]
            print(f"  {c:>9.4f}{stars(p):<3s}({se:.4f})", end="")
        print()

    print("  " + "─" * 95)
    for lbl, fn in [("R²", lambda r: r.rsquared),
                    ("Adj R²", lambda r: r.rsquared_adj),
                    ("σ (resid)", lambda r: np.sqrt(r.mse_resid))]:
        print(f"  {lbl:<30s}", end="")
        for r in [1, 2, 3]:
            print(f"  {fn(ols_results[r]):>22.4f}", end="")
        print()


def wald_table(result_full, k, var_names, dep_label):
    """Print Wald test results for key parameters."""
    print(f"\n  {'Variable':<34s}  {'F-stat':>8s}  {'p-value':>8s}  {'Result':>14s}")
    print("  " + "─" * 72)

    tests = (
        [("const",                   "Equal intercepts (μ)")]
        + [("ECT_lag",               "Equal adj. speed (α)")]
        + [(v, f"Equal regime coef: {v}") for v in CONTROLS]
    )
    for vname, label in tests:
        if vname not in var_names:
            continue
        idx = var_names.index(vname)
        F, p = wald_test_equal(result_full, k, idx)
        res_txt = ("REJECT ***" if p < 0.01
                   else "REJECT **"  if p < 0.05
                   else "Reject *"   if p < 0.10
                   else "fail reject")
        print(f"  {label:<34s}  {F:>8.4f}  {p:>8.4f}  {res_txt:>14s}")


def regime_summary_table(sample: pd.DataFrame):
    print("\n" + DIVIDER)
    print("  REGIME DISTRIBUTION")
    print(DIVIDER)
    T = len(sample)
    for r, lbl in REGIME_LABELS.items():
        sub = sample[sample["regime"] == r]
        n   = len(sub)
        print(f"\n  {lbl}  (N={n:,}, {n/T*100:.1f}%)")
        print(f"    ECT    : mean={sub['ECT_lag'].mean():.2f}  "
              f"std={sub['ECT_lag'].std():.2f}  "
              f"min={sub['ECT_lag'].min():.2f}  max={sub['ECT_lag'].max():.2f}")
        print(f"    Δsa    : mean={sub['dsa'].mean():.3f}  "
              f"std={sub['dsa'].std():.3f}")
        print(f"    Batt.dis: mean={sub['battery_discharging_mw'].mean():.2f}  "
              f"max={sub['battery_discharging_mw'].max():.2f}")
        print(f"    Batt.chg: mean={sub['battery_charging_mw'].mean():.2f}  "
              f"max={sub['battery_charging_mw'].max():.2f}")


def ljung_box_check(ols_results: dict, dep_label: str, lags: int = 12):
    """Quick residual autocorrelation check per regime."""
    print(f"\n  Ljung-Box residual autocorrelation ({dep_label}, {lags} lags):")
    print(f"  {'Regime':<30s}  {'LB stat':>9s}  {'p-value':>8s}  {'AR?':>5s}")
    print("  " + "─" * 60)
    for r in [1, 2, 3]:
        resid = ols_results[r].resid
        lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
        stat = float(lb["lb_stat"].iloc[0])
        pval = float(lb["lb_pvalue"].iloc[0])
        flag = "YES" if pval < 0.05 else "no"
        print(f"  {REGIME_LABELS[r]:<30s}  {stat:>9.4f}  {pval:>8.4f}  {flag:>5s}")


# ---------------------------------------------------------------------------
# 6. Save results to CSV and text
# ---------------------------------------------------------------------------
def save_coef_csv(ols_results: dict, var_names: list, dep_var: str):
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
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"phase2_{dep_var}_coefs.csv", index=False)


def save_wald_csv(wald_rows: list):
    pd.DataFrame(wald_rows).to_csv(RESULTS_DIR / "phase2_wald_tests.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    print(DIVIDER)
    print("  PHASE 2: THREE-REGIME TVECM ESTIMATION")
    print(f"  Thresholds: γ_L = {GAMMA_L:.4f},  γ_U = {GAMMA_U:.4f}  (ECT units $/MWh)")
    print(f"  ECM lags: {ECM_LAGS}   |   HAC bandwidth: {HAC_LAGS}")
    print(DIVIDER)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data and computing ECT ...")
    df   = load_data()
    beta, ect = estimate_ect(df)
    print(f"  Cointegrating vector: sa_price − {-beta[1]:.4f} × vic_price")
    print(f"  Obs (after price-cap filter): {len(df):,}")

    sample = build_sample(df, ect)
    T      = len(sample)
    print(f"  ECM sample (after lag creation + dropna): T = {T:,}")

    regime_summary_table(sample)

    # ── 2. Variable names ─────────────────────────────────────────────────
    var_names = regressor_names()
    k         = len(var_names)
    print(f"\n  Regressors per regime: {k}  ({var_names})")

    # ── 3. SA equation (Δsa_price) ────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  SA PRICE EQUATION  (dependent variable: Δsa_price)")
    print(DIVIDER)

    sa_results = estimate_equation(sample, "dsa")

    highlight = ["ECT_lag", "battery_discharging_mw", "battery_charging_mw"]
    coef_table(sa_results, var_names, "Δsa_price", highlight=highlight)

    print("\n  Stars: *** p<0.01  ** p<0.05  * p<0.10   HAC (Newey-West, 12 lags)")

    # ── 4. VIC equation (Δvic_price) ──────────────────────────────────────
    print("\n" + DIVIDER)
    print("  VIC PRICE EQUATION  (dependent variable: Δvic_price)")
    print(DIVIDER)

    vic_results = estimate_equation(sample, "dvic")
    coef_table(vic_results, var_names, "Δvic_price", highlight=highlight)
    print("\n  Stars: *** p<0.01  ** p<0.05  * p<0.10   HAC (Newey-West, 12 lags)")

    # ── 5. Interacted model + Wald tests (SA equation) ───────────────────
    print("\n" + DIVIDER)
    print("  WALD TESTS (SA EQUATION) — H0: equal coefficient across regimes")
    print("  Using interacted OLS with HAC robust standard errors")
    print(DIVIDER)

    y_sa, X_int, k_int = build_interacted(sample, "dsa")
    res_int_sa = OLS(y_sa, X_int).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    wald_table(res_int_sa, k_int, var_names, "SA")

    # ── 6. Interacted model + Wald tests (VIC equation) ──────────────────
    print("\n" + DIVIDER)
    print("  WALD TESTS (VIC EQUATION) — H0: equal coefficient across regimes")
    print(DIVIDER)

    y_vic, X_int_v, _ = build_interacted(sample, "dvic")
    res_int_vic = OLS(y_vic, X_int_v).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    wald_table(res_int_vic, k_int, var_names, "VIC")

    # ── 7. Battery effects summary ────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  BATTERY EFFECTS SUMMARY")
    print(DIVIDER)

    batt_idx_dis = var_names.index("battery_discharging_mw")
    batt_idx_chg = var_names.index("battery_charging_mw")

    print(f"\n  {'':30s}  {'Regime 1':>14s}  {'Regime 2':>14s}  {'Regime 3':>14s}")
    print("  " + "─" * 78)
    for dep, results in [("Δsa_price", sa_results), ("Δvic_price", vic_results)]:
        for idx, vname in [(batt_idx_dis, "battery_discharging_mw"),
                           (batt_idx_chg, "battery_charging_mw")]:
            print(f"  {dep+': '+vname:<30s}", end="")
            for r in [1, 2, 3]:
                c = results[r].params[idx]
                p = results[r].pvalues[idx]
                print(f"  {c:>9.4f}{stars(p):<3s}      ", end="")
            print()

    print("\n  Wald tests (SA eq): H0 = equal battery effect across regimes")
    for idx, vname in [(batt_idx_dis, "battery_discharging_mw"),
                       (batt_idx_chg, "battery_charging_mw")]:
        F, p = wald_test_equal(res_int_sa, k_int, idx)
        print(f"    {vname:<30s}  F = {F:.4f}   p = {p:.4f}  {stars(p)}")

    # ── 8. Adjustment speed summary ───────────────────────────────────────
    print("\n" + DIVIDER)
    print("  ADJUSTMENT SPEED COMPARISON (α = ECT_{t-1} coefficient)")
    print(DIVIDER)

    ect_idx = var_names.index("ECT_lag")
    print(f"\n  {'Equation':<14s}  {'Regime 1 α':>12s}  {'Regime 2 α':>12s}  "
          f"{'Regime 3 α':>12s}  {'Half-life (R1)':>14s}  {'Half-life (R3)':>14s}")
    print("  " + "─" * 84)

    for dep, results, label in [("SA",  sa_results,  "Δsa_price"),
                                 ("VIC", vic_results, "Δvic_price")]:
        a = {r: results[r].params[ect_idx] for r in [1, 2, 3]}
        p = {r: results[r].pvalues[ect_idx] for r in [1, 2, 3]}

        # Half-life: hl = -log(2)/log(1+α) for |α| < 1 (in 30-min periods)
        def halflife(alpha):
            if alpha <= -1 or alpha >= 0:
                return np.nan
            return -np.log(2) / np.log(1 + alpha) * 0.5  # in hours

        hl1 = halflife(a[1])
        hl3 = halflife(a[3])
        hl1_str = f"{hl1:.1f} h" if np.isfinite(hl1) else "n/a"
        hl3_str = f"{hl3:.1f} h" if np.isfinite(hl3) else "n/a"

        print(f"  {label:<14s}  "
              f"{a[1]:>9.4f}{stars(p[1]):<3s}  "
              f"{a[2]:>9.4f}{stars(p[2]):<3s}  "
              f"{a[3]:>9.4f}{stars(p[3]):<3s}  "
              f"{hl1_str:>14s}  {hl3_str:>14s}")

    F_ect, p_ect = wald_test_equal(res_int_sa, k_int, ect_idx)
    print(f"\n  Wald H0: α_1 = α_2 = α_3  →  F = {F_ect:.4f},  p = {p_ect:.4f}  {stars(p_ect)}")
    if p_ect < 0.05:
        print("  → REJECT: adjustment speed differs significantly across regimes.")
        print("    This validates the three-regime TVECM over a linear VECM.")
    else:
        print("  → Cannot reject equal adjustment speeds at 5%.")

    # ── 9. Diagnostics ───────────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  RESIDUAL DIAGNOSTICS")
    print(DIVIDER)
    ljung_box_check(sa_results,  "Δsa_price")
    ljung_box_check(vic_results, "Δvic_price")

    # ── 10. Save ─────────────────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  SAVING RESULTS")
    print(DIVIDER)

    save_coef_csv(sa_results,  var_names, "dsa")
    save_coef_csv(vic_results, var_names, "dvic")
    print(f"  Saved: results/phase2_dsa_coefs.csv")
    print(f"  Saved: results/phase2_dvic_coefs.csv")

    # Collect all Wald tests
    wald_rows = []
    for dep, res_int in [("SA", res_int_sa), ("VIC", res_int_vic)]:
        for vname in var_names:
            idx = var_names.index(vname)
            F, p = wald_test_equal(res_int, k_int, idx)
            wald_rows.append({"equation": dep, "variable": vname,
                               "F_stat": F, "p_value": p, "stars": stars(p)})
    save_wald_csv(wald_rows)
    print(f"  Saved: results/phase2_wald_tests.csv")

    # Summary text
    ect_idx = var_names.index("ECT_lag")
    summary = "\n".join([
        "PHASE 2 TVECM SUMMARY",
        "=" * 50,
        f"Thresholds: gamma_L = {GAMMA_L:.4f}  gamma_U = {GAMMA_U:.4f}",
        f"ECM lags: {ECM_LAGS}   HAC lags: {HAC_LAGS}",
        f"Total obs: {T}",
        f"  Regime 1 (N={len(sample[sample.regime==1]):,})",
        f"  Regime 2 (N={len(sample[sample.regime==2]):,})",
        f"  Regime 3 (N={len(sample[sample.regime==3]):,})",
        "",
        "Adjustment speeds (SA equation, alpha = ECT_{t-1}):",
        f"  Regime 1: {sa_results[1].params[ect_idx]:.4f}  (p={sa_results[1].pvalues[ect_idx]:.4f})",
        f"  Regime 2: {sa_results[2].params[ect_idx]:.4f}  (p={sa_results[2].pvalues[ect_idx]:.4f})",
        f"  Regime 3: {sa_results[3].params[ect_idx]:.4f}  (p={sa_results[3].pvalues[ect_idx]:.4f})",
        f"  Wald test (H0: alpha_1=alpha_2=alpha_3): F={F_ect:.4f}  p={p_ect:.4f}",
        "",
        "Battery discharge (SA equation, beta_dis):",
        f"  Regime 1: {sa_results[1].params[batt_idx_dis]:.4f}  (p={sa_results[1].pvalues[batt_idx_dis]:.4f})",
        f"  Regime 2: {sa_results[2].params[batt_idx_dis]:.4f}  (p={sa_results[2].pvalues[batt_idx_dis]:.4f})",
        f"  Regime 3: {sa_results[3].params[batt_idx_dis]:.4f}  (p={sa_results[3].pvalues[batt_idx_dis]:.4f})",
        "",
        "Battery charging (SA equation, beta_chg):",
        f"  Regime 1: {sa_results[1].params[batt_idx_chg]:.4f}  (p={sa_results[1].pvalues[batt_idx_chg]:.4f})",
        f"  Regime 2: {sa_results[2].params[batt_idx_chg]:.4f}  (p={sa_results[2].pvalues[batt_idx_chg]:.4f})",
        f"  Regime 3: {sa_results[3].params[batt_idx_chg]:.4f}  (p={sa_results[3].pvalues[batt_idx_chg]:.4f})",
    ])
    (RESULTS_DIR / "phase2_summary.txt").write_text(summary)
    print(f"  Saved: results/phase2_summary.txt")

    print("\n" + DIVIDER)
    print("  INTERPRETATION GUIDE FOR THE PAPER")
    print(DIVIDER)
    print(f"""
  α (ECT adjustment speed):
    Negative α means the price corrects back toward long-run equilibrium.
    |α_outer| > |α_band| confirms the threshold: faster correction in
    extreme regimes, slow/no correction inside the no-arbitrage band.

  Battery discharging (β_dis):
    Negative in Regime 3 (SA premium) → batteries dampen price spikes.
    Coefficient size × mean discharging gives average $/MWh dampening.

  Battery charging (β_chg):
    Negative in Regime 1 (SA discount) → batteries absorb surplus supply,
    pushing price up (reducing negative price depth).
    Positive in Regime 3 (SA premium) would indicate pro-cyclical charging
    (unlikely during peak — check for significance).

  Wald test on α across regimes:
    Rejection confirms TVECM > linear VECM (supports Phase 1 Hansen test).
    Non-rejection would suggest the threshold is statistically identified
    but adjustment speeds are similar (still useful for battery analysis).

  Wald test on battery coefficients across regimes:
    Rejection means the battery effect is regime-dependent — the key
    contribution of the paper (batteries act asymmetrically by regime).
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
