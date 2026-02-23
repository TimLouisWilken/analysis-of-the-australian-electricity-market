#!/usr/bin/env python3
"""
Phase 5: Lag Robustness Analysis for Outer Regimes
====================================================
Ljung-Box diagnostics in Phase 2 show significant residual autocorrelation
in Regime 1 (discount) and Regime 3 (spike) at the baseline 4-lag spec.

This script tests sensitivity of key estimates to lag order L ∈ {4,6,8,10,12}.

Strategy:
  - Build the sample once at MAX_LAGS=12 (all windows start at the same t₀)
    so that N is identical across lag orders within each regime.
  - For each (regime, L): fit OLS+HAC, compute AIC/BIC, Ljung-Box(12),
    α_SA (adjustment speed) and β_dis_SA (battery discharge effect).
  - Identify optimal L per regime via AIC and BIC.

Fixed from Phase 1 / Phase 2:
  β = 1.2103  (Johansen cointegrating vector)
  γ_L = -45.8039  γ_U = +20.0215  (two-threshold band test, p = 0.040)

Output files:
  results/phase5_lag_ic.csv       — AIC, BIC per (regime, lag)
  results/phase5_lag_ljungbox.csv — Ljung-Box per (regime, lag)
  results/phase5_lag_coefs.csv    — α, β_dis, β_chg per (regime, lag)
  results/phase5_summary.txt      — narrative summary
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
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------------------------------------------------------------------
# Configuration — all threshold/cointegration parameters fixed from Phase 1
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BETA_SA  = 1.2103      # Johansen cointegrating vector: ECT = sa - 1.2103*vic
GAMMA_L  = -45.8039    # lower threshold
GAMMA_U  =  20.0215    # upper threshold

MAX_LAGS   = 12
LAG_ORDERS = [4, 6, 8, 10, 12]
HAC_LAGS   = 12        # Newey-West bandwidth (≈ 6 h at 30-min)

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
# Data loading and sample construction
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime").sort_index()
    df = df[df["sa_price"].abs() < 15001]
    df = df[df["vic_price"].abs() < 15001]
    return df


def build_max_sample(df):
    """
    Build the estimation sample at MAX_LAGS=12.
    All lag-order comparisons use this same sample, ensuring N is constant.
    Uses fixed beta=1.2103 for ECT (rather than re-estimating via Johansen)
    so that only the lag order varies.
    """
    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")
    ect  = (sa - BETA_SA * vic).rename("ECT")

    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}

    for lag in range(1, MAX_LAGS + 1):
        rows[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows[f"dvic_l{lag}"] = dvic.shift(lag)

    for c in CONTROLS:
        rows[c] = df[c] if c in df.columns else np.nan

    sample = pd.DataFrame(rows).dropna()
    sample["regime"] = np.where(
        sample["ECT_lag"] < GAMMA_L, 1,
        np.where(sample["ECT_lag"] > GAMMA_U, 3, 2)
    )
    return sample


# ---------------------------------------------------------------------------
# Per-(regime, lag) estimation
# ---------------------------------------------------------------------------
def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fit_regime_lag(sub, L, dep_var="dsa"):
    """
    Fit OLS+HAC with lag order L on a regime sub-sample.

    Parameters
    ----------
    sub     : pd.DataFrame — rows for a single regime from the max-lag sample
    L       : int          — lag order to use (cols dsa_l1..lL, dvic_l1..lL)
    dep_var : str          — "dsa" (SA eq) or "dvic" (VIC eq)

    Returns dict with AIC, BIC, Ljung-Box, and key coefficient estimates.
    """
    lag_names = ([f"dsa_l{i}"  for i in range(1, L + 1)] +
                 [f"dvic_l{i}" for i in range(1, L + 1)])
    var_names = ["const", "ECT_lag"] + lag_names + CONTROLS

    X = sub[var_names[1:]].copy()
    X.insert(0, "const", 1.0)
    y    = sub[dep_var].values.astype(float)
    X_np = X.values.astype(float)

    res = OLS(y, X_np).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )

    n   = int(res.nobs)
    k   = len(var_names)
    ssr = float(np.sum(res.resid ** 2))
    aic = n * np.log(ssr / n) + 2 * k
    bic = n * np.log(ssr / n) + k * np.log(n)

    lb      = acorr_ljungbox(res.resid, lags=[12], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pval = float(lb["lb_pvalue"].iloc[0])

    ect_idx = var_names.index("ECT_lag")
    dis_idx = var_names.index("battery_discharging_mw")
    chg_idx = var_names.index("battery_charging_mw")

    return {
        "n"          : n,
        "k"          : k,
        "aic"        : aic,
        "bic"        : bic,
        "lb_stat"    : lb_stat,
        "lb_pval"    : lb_pval,
        "alpha"      : res.params[ect_idx],
        "alpha_se"   : res.bse[ect_idx],
        "alpha_p"    : res.pvalues[ect_idx],
        "beta_dis"   : res.params[dis_idx],
        "beta_dis_se": res.bse[dis_idx],
        "beta_dis_p" : res.pvalues[dis_idx],
        "beta_chg"   : res.params[chg_idx],
        "beta_chg_se": res.bse[chg_idx],
        "beta_chg_p" : res.pvalues[chg_idx],
        "rsq"        : res.rsquared,
        "rsq_adj"    : res.rsquared_adj,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(DIVIDER)
    print("  PHASE 5: LAG ROBUSTNESS ANALYSIS")
    print(f"  Fixed: β = {BETA_SA}   γ_L = {GAMMA_L:.2f}   γ_U = {GAMMA_U:.2f}")
    print(f"  Lag orders tested: {LAG_ORDERS}")
    print(f"  Max-lag sample (L={MAX_LAGS}) ensures N is identical across lag orders")
    print(DIVIDER)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df     = load_data()
    sample = build_max_sample(df)
    T      = len(sample)

    print(f"  Max-lag-12 ECM sample: T = {T:,}")
    for r in [1, 2, 3]:
        n = (sample["regime"] == r).sum()
        print(f"    Regime {r}: N = {n:,}  ({n / T * 100:.1f}%)")

    # ── 2. Estimate all (regime × lag) combinations ──────────────────────
    print(f"\n[2] Estimating {3 * len(LAG_ORDERS)} specifications ...")
    results   = {}
    ic_rows   = []
    lb_rows   = []
    coef_rows = []

    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r]
        for L in LAG_ORDERS:
            res = fit_regime_lag(sub, L, dep_var="dsa")
            results[(r, L)] = res

            ic_rows.append({
                "regime": r, "lags": L,
                "n": res["n"], "k": res["k"],
                "aic": res["aic"], "bic": res["bic"],
            })
            lb_rows.append({
                "regime": r, "lags": L,
                "lb_stat": res["lb_stat"], "lb_pval": res["lb_pval"],
                "ar_significant": "YES" if res["lb_pval"] < 0.05 else "no",
            })
            coef_rows.append({
                "regime"         : r,
                "lags"           : L,
                "alpha"          : res["alpha"],
                "alpha_se"       : res["alpha_se"],
                "alpha_stars"    : stars(res["alpha_p"]),
                "beta_dis"       : res["beta_dis"],
                "beta_dis_se"    : res["beta_dis_se"],
                "beta_dis_stars" : stars(res["beta_dis_p"]),
                "beta_chg"       : res["beta_chg"],
                "beta_chg_se"    : res["beta_chg_se"],
                "beta_chg_stars" : stars(res["beta_chg_p"]),
                "rsq"            : res["rsq"],
                "rsq_adj"        : res["rsq_adj"],
            })

    print(f"  Done.")

    # ── 3. AIC/BIC + Ljung-Box table ────────────────────────────────────
    print("\n" + DIVIDER)
    print("  AIC / BIC / LJUNG-BOX   —   SA equation (Δsa_price)")
    print(DIVIDER)
    hdr = (f"  {'R':<2s}  {'L':>4s}  {'N':>6s}  {'k':>3s}  "
           f"{'AIC':>12s}  {'ΔAIC':>8s}  {'BIC':>12s}  {'ΔBIC':>8s}  "
           f"{'LB(12)':>8s}  {'LB p':>7s}  AR?")
    print(hdr)
    print("  " + "─" * 92)

    for r in [1, 2, 3]:
        base_aic = results[(r, 4)]["aic"]
        base_bic = results[(r, 4)]["bic"]
        for L in LAG_ORDERS:
            res  = results[(r, L)]
            daic = res["aic"] - base_aic
            dbic = res["bic"] - base_bic
            ar   = "YES ◄" if res["lb_pval"] < 0.05 else "no"
            bl   = " (baseline)" if L == 4 else ""
            print(f"  {r:<2d}  {L:>4d}  {res['n']:>6d}  {res['k']:>3d}  "
                  f"{res['aic']:>12.1f}  {daic:>+8.1f}  "
                  f"{res['bic']:>12.1f}  {dbic:>+8.1f}  "
                  f"{res['lb_stat']:>8.3f}  {res['lb_pval']:>7.4f}  {ar}{bl}")
        print()

    # ── 4. Optimal lag identification ─────────────────────────────────────
    print(DIVIDER)
    print("  OPTIMAL LAG ORDER BY AIC AND BIC   (SA equation)")
    print(DIVIDER)
    print(f"\n  {'Regime':<8s}  {'Opt-AIC':>8s}  {'Opt-BIC':>8s}  "
          f"{'Baseline LB p':>14s}  {'AR resolved at opt L?':>22s}")
    print("  " + "─" * 68)

    opt_per_regime = {}
    for r in [1, 2, 3]:
        aics    = {L: results[(r, L)]["aic"] for L in LAG_ORDERS}
        bics    = {L: results[(r, L)]["bic"] for L in LAG_ORDERS}
        opt_aic = min(aics, key=aics.get)
        opt_bic = min(bics, key=bics.get)

        base_lb_p  = results[(r, 4)]["lb_pval"]
        opt_lb_p   = results[(r, opt_aic)]["lb_pval"]
        ar_resolve = "YES" if (base_lb_p < 0.05 and opt_lb_p >= 0.05) else (
                     "no" if base_lb_p >= 0.05 else
                     f"still AR (p={opt_lb_p:.3f})")
        opt_per_regime[r] = opt_aic
        print(f"  {r:<8d}  {opt_aic:>8d}  {opt_bic:>8d}  "
              f"{base_lb_p:>14.4f}  {ar_resolve:>22s}")

    # ── 5. Key coefficient table ──────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  KEY COEFFICIENTS BY (REGIME, LAG ORDER)   —   SA equation")
    print("  α = ECT adjustment speed   β_dis = battery discharge effect")
    print(DIVIDER)
    print(f"\n  {'R':<2s}  {'L':>4s}  {'α_SA':>10s}  {'(se)':>8s}  {'':3s}  "
          f"{'β_dis':>10s}  {'(se)':>8s}  {'':3s}  {'β_chg':>10s}  {'':3s}  "
          f"{'Adj R²':>7s}")
    print("  " + "─" * 92)

    for r in [1, 2, 3]:
        for L in LAG_ORDERS:
            res  = results[(r, L)]
            mark = " ◄" if L == 4 else ("  ★ AIC-opt" if L == opt_per_regime[r] else "")
            print(f"  {r:<2d}  {L:>4d}  "
                  f"{res['alpha']:>10.4f}  ({res['alpha_se']:.4f})  "
                  f"{stars(res['alpha_p']):<3s}  "
                  f"{res['beta_dis']:>10.4f}  ({res['beta_dis_se']:.4f})  "
                  f"{stars(res['beta_dis_p']):<3s}  "
                  f"{res['beta_chg']:>10.4f}  "
                  f"{stars(res['beta_chg_p']):<3s}  "
                  f"{res['rsq_adj']:>7.4f}{mark}")
        print()

    # ── 6. Stability summary ─────────────────────────────────────────────
    print(DIVIDER)
    print("  COEFFICIENT STABILITY CHECK")
    print("  Range of α and β_dis across L ∈ {4,6,8,10,12}")
    print(DIVIDER)
    print(f"\n  {'Regime':<8s}  {'α min':>9s}  {'α max':>9s}  {'α range':>9s}  "
          f"  {'β_dis min':>10s}  {'β_dis max':>10s}  {'β_dis range':>12s}")
    print("  " + "─" * 76)
    for r in [1, 2, 3]:
        alphas = [results[(r, L)]["alpha"]   for L in LAG_ORDERS]
        betas  = [results[(r, L)]["beta_dis"] for L in LAG_ORDERS]
        print(f"  {r:<8d}  "
              f"{min(alphas):>9.4f}  {max(alphas):>9.4f}  {max(alphas)-min(alphas):>9.4f}  "
              f"  {min(betas):>10.4f}  {max(betas):>10.4f}  {max(betas)-min(betas):>12.4f}")

    # ── 7. Save results ──────────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  SAVING RESULTS")
    print(DIVIDER)

    pd.DataFrame(ic_rows).to_csv(  RESULTS_DIR / "phase5_lag_ic.csv",       index=False)
    pd.DataFrame(lb_rows).to_csv(  RESULTS_DIR / "phase5_lag_ljungbox.csv",  index=False)
    pd.DataFrame(coef_rows).to_csv(RESULTS_DIR / "phase5_lag_coefs.csv",     index=False)
    print("  Saved: results/phase5_lag_ic.csv")
    print("  Saved: results/phase5_lag_ljungbox.csv")
    print("  Saved: results/phase5_lag_coefs.csv")

    # Summary text
    lines = [
        "PHASE 5 — LAG ROBUSTNESS ANALYSIS",
        "=" * 60,
        f"Fixed: beta={BETA_SA}  gamma_L={GAMMA_L:.4f}  gamma_U={GAMMA_U:.4f}",
        f"Max-lag sample: T={T}   Lag orders tested: {LAG_ORDERS}",
        f"HAC bandwidth: {HAC_LAGS}   Ljung-Box test lags: 12",
        "",
    ]
    for r in [1, 2, 3]:
        aics    = {L: results[(r, L)]["aic"] for L in LAG_ORDERS}
        bics    = {L: results[(r, L)]["bic"] for L in LAG_ORDERS}
        opt_aic = min(aics, key=aics.get)
        opt_bic = min(bics, key=bics.get)
        lines.append(f"Regime {r} (N={results[(r,4)]['n']}):")
        lines.append(f"  Optimal L by AIC: {opt_aic}   by BIC: {opt_bic}")
        lines.append(f"  Baseline (L=4) LB(12) p = {results[(r,4)]['lb_pval']:.4f}  "
                     f"({'significant AR' if results[(r,4)]['lb_pval'] < 0.05 else 'no significant AR'})")
        for L in LAG_ORDERS:
            res = results[(r, L)]
            lines.append(
                f"  L={L:2d}: alpha={res['alpha']:+.4f}{stars(res['alpha_p']):<3s}  "
                f"beta_dis={res['beta_dis']:+.4f}{stars(res['beta_dis_p']):<3s}  "
                f"AIC={res['aic']:.1f}  BIC={res['bic']:.1f}  "
                f"LB(12)p={res['lb_pval']:.4f}"
            )
        lines.append("")

    (RESULTS_DIR / "phase5_summary.txt").write_text("\n".join(lines))
    print("  Saved: results/phase5_summary.txt")

    print("\n" + DIVIDER)
    print("  INTERPRETATION")
    print(DIVIDER)
    print("""
  AIC: selects best predictive fit (rewards log-likelihood, penalises k by 2).
  BIC: penalises k by log(n) — prefers more parsimonious models.

  Stable α and β_dis across L → baseline (L=4) results are robust to lag
  truncation.  Large variation → Phase 2/3/4 point estimates are sensitive
  to specification and should be reported with caution.

  If LB(12) remains significant even at the AIC-optimal L, this suggests the
  autocorrelation is not purely due to lag truncation — likely GARCH-type
  volatility clustering in the extreme price regimes, which HAC-robust SEs
  partially address.

  Stars: *** p<0.01  ** p<0.05  * p<0.10  (HAC, Newey-West, 12 lags)
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
