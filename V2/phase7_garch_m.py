#!/usr/bin/env python3
"""
Phase 7: TVECM-GARCH-in-mean (GARCH-M) Extension
==================================================
Motivation: Phase 5 lag robustness confirms persistent ARCH-type residual
autocorrelation in Regimes 1 and 3 across all lag orders L ∈ {4..12}.
This rules out lag truncation; the source is GARCH-type volatility clustering
in extreme-price episodes (price spikes / deep discounts).

GARCH-M specification (per regime, SA equation):

  Δsa_t = μ_r + α_r · ECT_{t-1}
         + Σᵢ₌₁⁴ φᵢ · Δsa_{t-i}  +  Σᵢ₌₁⁴ ψᵢ · Δvic_{t-i}
         + controls
         + δ_r · σ_t        ← risk premium: conditional vol enters mean
         + ε_t

  ε_t = σ_t · z_t,   z_t ~ i.i.d. Student-t(ν)
  σ²_t = ω + α_G · ε²_{t-1} + β_G · σ²_{t-1}   [GARCH(1,1)]

δ_r is the risk premium: higher conditional volatility → higher expected
price change.  For Regime 3 (SA spike), δ > 0 means uncertain spike periods
carry an upward price premium independent of fundamentals.

Estimation: iterated two-step feasible GARCH-M (Engle 1982):
  1. OLS of mean equation → residuals ε̂_t
  2. GARCH(1,1)-t fit on ε̂_t  → σ̂_t (conditional std dev, arch lib)
  3. OLS + HAC of Δsa_t on [X_t, σ̂_t] → updated β, δ, new ε̂_t
  4. Repeat until |δ̂_new − δ̂_old| < 10⁻⁵  (max 10 iterations)

Output files:
  results/phase7_garch_params.csv   — GARCH(1,1)-t parameters per regime
  results/phase7_mean_coefs.csv     — mean equation coefficients (post GARCH-M)
  results/phase7_diagnostics.csv    — ARCH-LM and Ljung-Box tests
  results/phase7_summary.txt        — narrative summary
"""

import warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model as _arch_model

# ---------------------------------------------------------------------------
# Configuration — fixed from Phase 1 / Phase 2
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BETA_SA  = 1.2103
GAMMA_L  = -45.8039
GAMMA_U  =  20.0215

ECM_LAGS = 4        # baseline (Phase 5: BIC-optimal R2/R3; robustly stable R1)
HAC_LAGS = 12       # Newey-West bandwidth
MAX_ITER = 10       # max GARCH-M iterations
CONV_TOL = 1e-5     # convergence threshold on δ

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
# Data loading and ECM sample (identical to Phase 2)
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime").sort_index()
    df = df[df["sa_price"].abs() < 15001]
    df = df[df["vic_price"].abs() < 15001]
    return df


def build_sample(df):
    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")
    ect  = (sa - BETA_SA * vic).rename("ECT")

    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}
    for lag in range(1, ECM_LAGS + 1):
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


def base_var_names():
    lag_names = ([f"dsa_l{i}"  for i in range(1, ECM_LAGS + 1)] +
                 [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)])
    return ["const", "ECT_lag"] + lag_names + CONTROLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stars(p):
    if not np.isfinite(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def arch_lm_test(resid, nlags=12):
    """
    ARCH-LM test (Engle 1982): regress ε²_t on ε²_{t-1},...,ε²_{t-nlags}.
    Returns (LM_stat, p_value).  H₀: no ARCH effects.
    """
    e2  = resid ** 2
    n   = len(e2)
    if n <= nlags + 5:
        return np.nan, np.nan
    try:
        X_lm = np.column_stack(
            [np.ones(n - nlags)] +
            [e2[nlags - i : n - i] for i in range(1, nlags + 1)]
        )
        y_lm  = e2[nlags:]
        res   = OLS(y_lm, X_lm).fit()
        lm    = float((n - nlags) * res.rsquared)
        pval  = float(1.0 - sp_stats.chi2.cdf(lm, nlags))
        return lm, pval
    except Exception:
        return np.nan, np.nan


def lb_stat_p(series, lag=12):
    """Ljung-Box Q(lag) statistic and p-value."""
    try:
        lb = acorr_ljungbox(series, lags=[lag], return_df=True)
        return float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Core: per-regime iterated two-step GARCH-M estimator
# ---------------------------------------------------------------------------
def fit_garch_m(sub, dep_var="dsa"):
    """
    Iterated two-step GARCH(1,1)-t GARCH-M estimator for a single regime.

    Returns a flat dict with:
      - Mean equation: alpha, beta_dis, beta_chg, delta (+ se, p)
      - GARCH equation: omega, alpha_g, beta_g, nu, persistence
      - Diagnostics: ARCH-LM pre/post, LB on raw/standardised residuals
      - Fit info: n, n_iter, converged, garch_ok
    """
    vn_base = base_var_names()

    X_base  = sub[vn_base[1:]].copy()
    X_base.insert(0, "const", 1.0)
    y       = sub[dep_var].values.astype(float)
    X_np    = X_base.values.astype(float)
    n       = len(y)

    # ── Baseline OLS (no GARCH-M) ─────────────────────────────────────────
    res_ols = OLS(y, X_np).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    resid   = res_ols.resid.copy()

    # Pre-GARCH-M diagnostics
    arch_pre_stat, arch_pre_p = arch_lm_test(resid)
    lb_raw_pre_stat, lb_raw_pre_p = lb_stat_p(resid)

    # ── Iterated GARCH-M ──────────────────────────────────────────────────
    sigma_t   = np.full(n, float(np.std(resid)))   # warm-start σ̂
    delta_old = 0.0
    garch_res = None
    res_gm    = None
    converged = False
    garch_ok  = True
    n_iter    = 0

    for it in range(1, MAX_ITER + 1):
        n_iter = it

        # Step 1: fit GARCH(1,1)-t to current residuals
        try:
            am       = _arch_model(resid, mean="Zero", vol="GARCH",
                                   p=1, q=1, dist="StudentsT", rescale=True)
            gr       = am.fit(disp="off", show_warning=False,
                              options={"maxiter": 500, "ftol": 1e-10})
            sig_new  = gr.conditional_volatility

            if np.any(np.isnan(sig_new)) or np.any(sig_new <= 0):
                garch_ok = False
                break

            sigma_t   = sig_new
            garch_res = gr

        except Exception as exc:
            garch_ok = False
            break

        # Step 2: augment X with σ̂_t, re-estimate mean equation (HAC)
        X_aug = np.column_stack([X_np, sigma_t])
        try:
            res_gm = OLS(y, X_aug).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
            )
        except Exception:
            garch_ok = False
            break

        resid     = res_gm.resid.copy()
        delta_new = float(res_gm.params[-1])   # σ_t is last column

        if abs(delta_new - delta_old) < CONV_TOL:
            converged = True
            break
        delta_old = delta_new

    # ── Post-GARCH-M diagnostics ──────────────────────────────────────────
    if garch_ok and garch_res is not None:
        arch_post_stat, arch_post_p = arch_lm_test(resid)
        lb_raw_post_stat, lb_raw_post_p = lb_stat_p(resid)

        std_r  = resid / np.where(sigma_t > 0, sigma_t, 1e-9)
        lb_std_stat,  lb_std_p  = lb_stat_p(std_r)
        lb_sq_stat,   lb_sq_p   = lb_stat_p(std_r ** 2)
    else:
        arch_post_stat = arch_post_p = np.nan
        lb_raw_post_stat = lb_raw_post_p = np.nan
        lb_std_stat = lb_std_p = np.nan
        lb_sq_stat  = lb_sq_p  = np.nan
        res_gm = None

    # ── Extract GARCH parameters ──────────────────────────────────────────
    if garch_res is not None:
        gp   = garch_res.params
        gse  = garch_res.std_err
        gpv  = garch_res.pvalues

        def gget(key):
            return (float(gp.get(key, np.nan)),
                    float(gse.get(key, np.nan)),
                    float(gpv.get(key, np.nan)))

        omega_v,   omega_se,   omega_p   = gget("omega")
        alpha_g_v, alpha_g_se, alpha_g_p = gget("alpha[1]")
        beta_g_v,  beta_g_se,  beta_g_p  = gget("beta[1]")
        nu_v,      nu_se,      nu_p      = gget("nu")
        persistence = alpha_g_v + beta_g_v
        uncon_vol   = (omega_v / max(1.0 - persistence, 1e-8)) ** 0.5 \
                      if persistence < 1.0 else np.inf
    else:
        omega_v = alpha_g_v = beta_g_v = nu_v = np.nan
        omega_se = alpha_g_se = beta_g_se = nu_se = np.nan
        omega_p = alpha_g_p = beta_g_p = nu_p = np.nan
        persistence = uncon_vol = np.nan

    # ── Mean equation coefficients ────────────────────────────────────────
    res_use  = res_gm if res_gm is not None else res_ols
    vn_use   = vn_base + ["sigma_t"] if res_gm is not None else vn_base

    ect_idx  = vn_use.index("ECT_lag")
    dis_idx  = vn_use.index("battery_discharging_mw")
    chg_idx  = vn_use.index("battery_charging_mw")
    has_gm   = res_gm is not None

    return {
        # Fit info
        "n"            : n,
        "n_iter"       : n_iter,
        "converged"    : converged,
        "garch_ok"     : garch_ok,
        # Mean — ECT adjustment speed
        "alpha"        : float(res_use.params[ect_idx]),
        "alpha_se"     : float(res_use.bse[ect_idx]),
        "alpha_p"      : float(res_use.pvalues[ect_idx]),
        # Mean — battery discharge
        "beta_dis"     : float(res_use.params[dis_idx]),
        "beta_dis_se"  : float(res_use.bse[dis_idx]),
        "beta_dis_p"   : float(res_use.pvalues[dis_idx]),
        # Mean — battery charge
        "beta_chg"     : float(res_use.params[chg_idx]),
        "beta_chg_se"  : float(res_use.bse[chg_idx]),
        "beta_chg_p"   : float(res_use.pvalues[chg_idx]),
        # GARCH-M risk premium δ
        "delta"        : float(res_use.params[-1])   if has_gm else np.nan,
        "delta_se"     : float(res_use.bse[-1])      if has_gm else np.nan,
        "delta_p"      : float(res_use.pvalues[-1])  if has_gm else np.nan,
        # GARCH variance equation
        "omega"        : omega_v,    "omega_se"    : omega_se,    "omega_p"    : omega_p,
        "alpha_g"      : alpha_g_v,  "alpha_g_se"  : alpha_g_se,  "alpha_g_p"  : alpha_g_p,
        "beta_g"       : beta_g_v,   "beta_g_se"   : beta_g_se,   "beta_g_p"   : beta_g_p,
        "nu"           : nu_v,        "nu_se"       : nu_se,
        "persistence"  : persistence,
        "uncon_vol"    : uncon_vol if np.isfinite(uncon_vol) else np.nan,
        # Diagnostics
        "arch_pre"     : arch_pre_stat,  "arch_pre_p"  : arch_pre_p,
        "arch_post"    : arch_post_stat, "arch_post_p" : arch_post_p,
        "lb_raw_pre"   : lb_raw_pre_stat,  "lb_raw_pre_p"  : lb_raw_pre_p,
        "lb_raw_post"  : lb_raw_post_stat, "lb_raw_post_p" : lb_raw_post_p,
        "lb_std"       : lb_std_stat,   "lb_std_p"    : lb_std_p,
        "lb_sq"        : lb_sq_stat,    "lb_sq_p"     : lb_sq_p,
        # Fit quality
        "rsq"          : float(res_use.rsquared),
        "rsq_adj"      : float(res_use.rsquared_adj),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(DIVIDER)
    print("  PHASE 7: TVECM-GARCH-in-mean (GARCH-M) EXTENSION")
    print(f"  Spec: GARCH(1,1)-t  |  ECM lags={ECM_LAGS}  |  HAC={HAC_LAGS}  "
          f"|  max_iter={MAX_ITER}  |  tol={CONV_TOL:.0e}")
    print(f"  Motivation: Phase 5 confirmed persistent ARCH in R1 & R3 (all lag orders)")
    print(DIVIDER)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df     = load_data()
    sample = build_sample(df)
    T      = len(sample)
    print(f"  ECM sample: T = {T:,}  (same specification as Phase 2)")
    for r in [1, 2, 3]:
        n = (sample["regime"] == r).sum()
        print(f"    Regime {r}: N = {n:,}  ({n/T*100:.1f}%)")

    # ── 2. Load Phase 2 baseline for comparison ───────────────────────────
    p2_coef = {}
    try:
        p2 = pd.read_csv(RESULTS_DIR / "phase2_dsa_coefs.csv")
        for _, row in p2.iterrows():
            p2_coef[(int(row["regime"]), str(row["variable"]))] = (
                float(row["coef"]), float(row["p_value"])
            )
        have_p2 = True
    except Exception:
        have_p2 = False

    # ── 3. Estimate GARCH-M per regime ────────────────────────────────────
    print("\n[2] Estimating GARCH-M per regime (SA equation) ...")
    results = {}
    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r]
        print(f"  Regime {r}  (N = {len(sub):,}) ...", end="  ", flush=True)
        res = fit_garch_m(sub, dep_var="dsa")
        results[r] = res
        if not res["garch_ok"]:
            status = "GARCH FAILED — showing OLS only"
        elif res["converged"]:
            status = f"converged ({res['n_iter']} iterations)"
        else:
            status = f"max iterations reached ({res['n_iter']})"
        print(status)

    # ── 4. ARCH-LM pre/post table ─────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  ARCH-LM TEST (12 lags)  —  BEFORE vs AFTER GARCH-M")
    print("  H₀: no ARCH effects in residuals")
    print(DIVIDER)
    hdr = (f"  {'Regime':<8}  {'N':>6}  "
           f"{'ARCH pre':>10}  {'p pre':>7}  {'Signif?':>8}    "
           f"{'ARCH post':>10}  {'p post':>7}  {'Resolved?':>10}")
    print(hdr)
    print("  " + "─" * 78)
    for r in [1, 2, 3]:
        res = results[r]
        pre = (f"{res['arch_pre']:>10.3f}  {res['arch_pre_p']:>7.4f}  "
               f"{'YES ***' if res['arch_pre_p'] < 0.001 else 'YES' if res['arch_pre_p'] < 0.05 else 'no':>8}")
        if np.isfinite(res["arch_post_p"]):
            resolved = "YES ✓" if res["arch_post_p"] >= 0.05 else "partial"
            post = (f"{res['arch_post']:>10.3f}  {res['arch_post_p']:>7.4f}  {resolved:>10}")
        else:
            post = f"{'—':>10}  {'—':>7}  {'(failed)':>10}"
        print(f"  {r:<8}  {res['n']:>6}  {pre}    {post}")

    # ── 5. GARCH parameter table ──────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  GARCH(1,1)-t VARIANCE EQUATION PARAMETERS")
    print("  σ²_t = ω + α_G·ε²_{t-1} + β_G·σ²_{t-1}   [Student-t(ν) errors]")
    print(DIVIDER)
    print(f"\n  {'Regime':<8}  {'ω':>10}     {'α_G (ARCH)':>12}     "
          f"{'β_G (GARCH)':>12}     {'α_G+β_G':>8}  {'ν (dof)':>9}  {'Uncon.σ':>10}")
    print("  " + "─" * 90)
    for r in [1, 2, 3]:
        res = results[r]
        if not res["garch_ok"]:
            print(f"  {r:<8}  {'GARCH fit failed':>60}")
            continue
        uv  = f"{res['uncon_vol']:>10.3f}" if np.isfinite(res["uncon_vol"]) else " near unit-root"
        print(f"  {r:<8}  "
              f"{res['omega']:>10.4f}     "
              f"{res['alpha_g']:>8.4f}{stars(res['alpha_g_p']):<4}     "
              f"{res['beta_g']:>8.4f}{stars(res['beta_g_p']):<4}     "
              f"{res['persistence']:>8.4f}  "
              f"{res['nu']:>9.3f}  "
              f"{uv}")
    print("\n  Stars: *** p<0.01  ** p<0.05  * p<0.10  (GARCH MLE Bollerslev-Wooldridge QML t-stats)")
    print("  α_G + β_G: GARCH persistence.  Near 1.0 → long-memory volatility clustering.")
    print("  ν: Student-t degrees of freedom.  Small ν (<5) → heavy tails.")
    print("  Uncon. σ = √(ω / (1−α_G−β_G))  [conditional std dev at unconditional mean]")

    # ── 6. Risk premium (δ) table ─────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  RISK PREMIUM COEFFICIENT  δ_r  (GARCH-M: δ_r · σ̂_t in mean equation)")
    print("  Interpretation: 1 $/MWh increase in conditional volatility → δ $/MWh change in Δsa_t")
    print(DIVIDER)
    print(f"\n  {'Regime':<8}  {'δ':>10}  {'se':>8}  {'':3}  {'Economic reading':>45}")
    print("  " + "─" * 80)
    interp = {
        1: {True:  "positive premium: uncertainty deepens discount episodes",
            False: "vol. dampens discount (unusual)"},
        2: {True:  "positive premium in near-equilibrium band (expect ≈ 0)",
            False: "negative premium in band"},
        3: {True:  "positive risk premium in spike regime ✓ (uncertainty → higher prices)",
            False: "negative: vol. dampens spike expectations"},
    }
    for r in [1, 2, 3]:
        res = results[r]
        d, ds, dp = res["delta"], res["delta_se"], res["delta_p"]
        if not np.isfinite(d):
            print(f"  {r:<8}  {'—':>10}  {'—':>8}  {'':3}  GARCH fit failed — no estimate")
        else:
            reading = interp[r][d > 0]
            print(f"  {r:<8}  {d:>10.4f}  {ds:>8.4f}  {stars(dp):<3}  {reading}")

    # ── 7. Coefficient comparison: Phase 2 OLS → Phase 7 GARCH-M ─────────
    print("\n" + DIVIDER)
    print("  COEFFICIENT COMPARISON: Phase 2 (OLS) → Phase 7 (GARCH-M)")
    print("  SA equation  |  α = ECT adjustment speed  |  β_dis = battery discharge")
    print(DIVIDER)
    print(f"\n  {'Spec':<12}  "
          f"{'R1 α':>8}  {'':3}  {'R1 β_dis':>9}  {'':3}  "
          f"{'R2 α':>8}  {'':3}  {'R2 β_dis':>9}  {'':3}  "
          f"{'R3 α':>8}  {'':3}  {'R3 β_dis':>9}  {'':3}")
    print("  " + "─" * 97)

    if have_p2:
        row_p2 = f"  {'Phase 2 OLS':<12}"
        for r in [1, 2, 3]:
            a, ap = p2_coef.get((r, "ECT_lag"), (np.nan, 1.0))
            b, bp = p2_coef.get((r, "battery_discharging_mw"), (np.nan, 1.0))
            row_p2 += f"  {a:>8.4f}  {stars(ap):<3}  {b:>9.4f}  {stars(bp):<3}"
        print(row_p2)

    row_gm = f"  {'GARCH-M':<12}"
    for r in [1, 2, 3]:
        res = results[r]
        row_gm += (f"  {res['alpha']:>8.4f}  {stars(res['alpha_p']):<3}  "
                   f"{res['beta_dis']:>9.4f}  {stars(res['beta_dis_p']):<3}")
    print(row_gm)
    print("\n  Stars: *** p<0.01  ** p<0.05  * p<0.10  (HAC Newey-West, 12 lags)")

    # ── 8. Residual diagnostics table ────────────────────────────────────
    print("\n" + DIVIDER)
    print("  RESIDUAL DIAGNOSTICS  (Ljung-Box Q(12)  and  ARCH-LM(12))")
    print("  ε̃_t = ε̂_t / σ̂_t  (standardised residuals)  —  should be i.i.d.")
    print(DIVIDER)
    print(f"\n  {'Regime':<8}  "
          f"{'LB raw OLS':>12}  {'':8}  "
          f"{'LB raw GM':>12}  {'':8}  "
          f"{'LB std. res':>12}  {'':8}  "
          f"{'LB std²':>10}  {'':8}")
    print(f"  {'':8}  "
          f"{'Q(12)':>8}  {'p':>8}    "
          f"{'Q(12)':>8}  {'p':>8}    "
          f"{'Q(12)':>8}  {'p':>8}    "
          f"{'Q(12)':>8}  {'p':>8}")
    print("  " + "─" * 98)

    def fmtlb(stat, pval, flag_sig=True):
        if not np.isfinite(stat):
            return f"{'—':>8}  {'—':>8}"
        flag = " ←" if (flag_sig and pval < 0.05) else "  "
        return f"{stat:>8.3f}  {pval:>8.4f}{flag}"

    for r in [1, 2, 3]:
        res = results[r]
        print(f"  {r:<8}  "
              f"{fmtlb(res['lb_raw_pre'],  res['lb_raw_pre_p'])}    "
              f"{fmtlb(res['lb_raw_post'], res['lb_raw_post_p'])}    "
              f"{fmtlb(res['lb_std'],      res['lb_std_p'])}    "
              f"{fmtlb(res['lb_sq'],       res['lb_sq_p'])}")

    # ── 9. Save results ───────────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  SAVING RESULTS")
    print(DIVIDER)

    garch_rows, coef_rows, diag_rows = [], [], []
    for r in [1, 2, 3]:
        res = results[r]
        garch_rows.append({
            "regime":      r,  "n": res["n"],
            "omega":       res["omega"],    "omega_se":    res["omega_se"],    "omega_p":    res["omega_p"],
            "alpha_g":     res["alpha_g"],  "alpha_g_se":  res["alpha_g_se"],  "alpha_g_p":  res["alpha_g_p"],
            "beta_g":      res["beta_g"],   "beta_g_se":   res["beta_g_se"],   "beta_g_p":   res["beta_g_p"],
            "persistence": res["persistence"],
            "nu":          res["nu"],       "nu_se":       res["nu_se"],
            "uncon_vol":   res["uncon_vol"],
            "garch_ok":    res["garch_ok"], "converged":   res["converged"],  "n_iter": res["n_iter"],
        })
        coef_rows.append({
            "regime":       r,
            "alpha":        res["alpha"],   "alpha_se":    res["alpha_se"],    "alpha_stars":    stars(res["alpha_p"]),
            "beta_dis":     res["beta_dis"],"beta_dis_se": res["beta_dis_se"], "beta_dis_stars": stars(res["beta_dis_p"]),
            "beta_chg":     res["beta_chg"],"beta_chg_se": res["beta_chg_se"], "beta_chg_stars": stars(res["beta_chg_p"]),
            "delta":        res["delta"],   "delta_se":    res["delta_se"],    "delta_stars":    stars(res["delta_p"]),
            "rsq":          res["rsq"],     "rsq_adj":     res["rsq_adj"],
        })
        diag_rows.append({
            "regime":         r,
            "arch_pre":       res["arch_pre"],      "arch_pre_p":    res["arch_pre_p"],
            "arch_post":      res["arch_post"],     "arch_post_p":   res["arch_post_p"],
            "lb_raw_pre":     res["lb_raw_pre"],    "lb_raw_pre_p":  res["lb_raw_pre_p"],
            "lb_raw_post":    res["lb_raw_post"],   "lb_raw_post_p": res["lb_raw_post_p"],
            "lb_std":         res["lb_std"],        "lb_std_p":      res["lb_std_p"],
            "lb_sq":          res["lb_sq"],         "lb_sq_p":       res["lb_sq_p"],
        })

    pd.DataFrame(garch_rows).to_csv(RESULTS_DIR / "phase7_garch_params.csv", index=False)
    pd.DataFrame(coef_rows).to_csv( RESULTS_DIR / "phase7_mean_coefs.csv",   index=False)
    pd.DataFrame(diag_rows).to_csv( RESULTS_DIR / "phase7_diagnostics.csv",  index=False)
    print("  Saved: results/phase7_garch_params.csv")
    print("  Saved: results/phase7_mean_coefs.csv")
    print("  Saved: results/phase7_diagnostics.csv")

    # Summary text
    lines = [
        "PHASE 7 — TVECM-GARCH-M SUMMARY",
        "=" * 60,
        f"Spec: GARCH(1,1)-t  ECM_lags={ECM_LAGS}  HAC={HAC_LAGS}  max_iter={MAX_ITER}",
        "",
        "GARCH parameters:",
    ]
    for r in [1, 2, 3]:
        res = results[r]
        if not res["garch_ok"]:
            lines.append(f"  Regime {r}: GARCH fit FAILED")
        else:
            lines.append(
                f"  Regime {r}: omega={res['omega']:.4f}  alpha_G={res['alpha_g']:.4f}"
                f"{stars(res['alpha_g_p'])}  beta_G={res['beta_g']:.4f}"
                f"{stars(res['beta_g_p'])}  persistence={res['persistence']:.4f}"
                f"  nu={res['nu']:.2f}  converged={res['converged']}"
            )
    lines += ["", "Risk premium delta:"]
    for r in [1, 2, 3]:
        res = results[r]
        lines.append(
            f"  Regime {r}: delta={res['delta']:.4f}  se={res['delta_se']:.4f}"
            f"  p={res['delta_p']:.4f}{stars(res['delta_p'])}"
        )
    lines += ["", "Battery discharge (Phase 2 OLS → Phase 7 GARCH-M):"]
    for r in [1, 2, 3]:
        res = results[r]
        if have_p2:
            p2v, p2p = p2_coef.get((r, "battery_discharging_mw"), (np.nan, 1.0))
            lines.append(
                f"  Regime {r}: OLS={p2v:.4f}{stars(p2p)}"
                f"  →  GARCH-M={res['beta_dis']:.4f}{stars(res['beta_dis_p'])}"
            )
        else:
            lines.append(f"  Regime {r}: GARCH-M={res['beta_dis']:.4f}{stars(res['beta_dis_p'])}")
    lines += ["", "Diagnostics (p-values):"]
    for r in [1, 2, 3]:
        res = results[r]
        lines.append(
            f"  Regime {r}: ARCH_pre={res['arch_pre_p']:.4f}"
            f"  ARCH_post={res['arch_post_p']:.4f}"
            f"  LB_std={res['lb_std_p']:.4f}"
            f"  LB_sq={res['lb_sq_p']:.4f}"
        )
    (RESULTS_DIR / "phase7_summary.txt").write_text("\n".join(lines))
    print("  Saved: results/phase7_summary.txt")

    # ── 10. Interpretation guide ──────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  INTERPRETATION GUIDE")
    print(DIVIDER)
    print("""
  ARCH-LM pre: Significant → volatility clustering confirmed in baseline OLS
    residuals.  Expected in R1 and R3 (Phase 5 finding).  R2 should be clean.

  GARCH persistence (α_G + β_G):
    Near 1.0 → near-integrated / long-memory volatility (IGARCH territory).
    Common in electricity spot prices — extreme events create persistent
    volatility clusters that decay slowly.
    ν (Student-t dof): small ν (< 5) confirms heavy tails beyond normal.

  Risk premium δ:
    Regime 3 (spike): positive δ → uncertainty itself drives prices higher.
    Mechanism: during spike episodes, market participants require compensation
    for price uncertainty → upward bias in Δsa_t beyond fundamentals.
    Policy implication: batteries reduce σ_t → indirect dampening via δ·σ_t
    even without changing the direct β_dis channel.

  Battery coefficient stability (β_dis Phase 2 → Phase 7):
    Stable sign and magnitude → OLS estimate was not contaminated by
    heteroscedasticity; Phase 2 inference is robust.
    Large change → GARCH-M provides materially different (more efficient)
    estimate of the battery causal channel.

  LB on standardised residuals ε̃_t = ε̂_t / σ̂_t:
    If LB(12) p < 0.05 remains significant → residual serial structure
    beyond GARCH (possible jump component or regime-switching volatility).
    LB on ε̃²_t: should be near 1.0 if GARCH variance is correctly specified.

  Stars: *** p<0.01  ** p<0.05  * p<0.10  (HAC for mean eq.; MLE QML for GARCH)
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
