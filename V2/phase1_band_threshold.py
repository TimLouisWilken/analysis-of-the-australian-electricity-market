#!/usr/bin/env python3
"""
Phase 1 Extension: Band Threshold Tests for TVECM
==================================================
Motivation: The V-SA interconnector creates a natural no-arbitrage band.
Within ±γ of long-run equilibrium, flows don't activate (transaction costs,
congestion limits). Outside the band, arbitrage corrects the spread.
This maps to a BAND-TVECM structure, not a simple one-sided threshold.

Tests
-----
1. Symmetric Band (Balke & Fomby 1997)
      H0: linear VECM (one regime)
      H1: inner |ECT| ≤ γ   vs   outer |ECT| > γ
      One parameter γ > 0; economically: symmetric no-arbitrage band.

2. Two-Threshold / General Band (3-regime TVECM)
      H0: linear VECM
      H1: low ECT < γ_L  |  band γ_L ≤ ECT ≤ γ_U  |  high ECT > γ_U
      Two parameters; allows asymmetric import/export thresholds.

Both use Rademacher wild-bootstrap (vectorised — all draws processed per
grid-point simultaneously via numpy BLAS matmul).

Comparison table at the end ranks all three Hansen tests (single-threshold
from phase1_diagnostics.py + both band variants) by AIC and bootstrap p.
"""

import os, warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import time
from pathlib import Path

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ECM_LAGS    = 4
SEED        = 42
TRIM        = 0.15    # minimum fraction per regime (single + symmetric)
TRIM_3R     = 0.10    # minimum fraction per regime (3-regime test)

# Grid / bootstrap settings
N_GAMMA_SYM = 200     # grid for symmetric band (1 parameter)
N_BOOT_SYM  = 799     # bootstrap reps — symmetric band

N_GAMMA_2T  = 40      # grid per axis for two-threshold (40×40/2 ≈ 780 valid pairs)
N_BOOT_2T   = 399     # bootstrap reps — two-threshold

DIVIDER = "=" * 72
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Data loading + ECT (self-contained)
# ---------------------------------------------------------------------------
def load_and_prepare():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime")
    prices = df[["sa_price", "vic_price"]].dropna()
    prices = prices[prices["sa_price"].abs() < 15001]
    prices = prices[prices["vic_price"].abs() < 15001].copy()

    # Johansen cointegration → cointegrating vector + ECT
    res  = coint_johansen(prices[["sa_price", "vic_price"]].values,
                          det_order=0, k_ar_diff=ECM_LAGS)
    beta = res.evec[:, 0]
    beta = beta / beta[0]          # normalise [1, -β]

    ect  = prices[["sa_price", "vic_price"]].values @ beta
    ect  = pd.Series(ect, index=prices.index, name="ECT")

    return prices, beta, ect


def build_ecm(prices, ect, n_lags=ECM_LAGS):
    sa   = prices["sa_price"]
    vic  = prices["vic_price"]
    dsa  = sa.diff();  dvic = vic.diff()
    start = prices.index[n_lags + 1]

    y = dsa.loc[start:].values.astype(float)
    q = ect.shift(1).loc[start:].values.astype(float)

    cols = [np.ones(len(y)), q]
    for lag in range(1, n_lags + 1):
        cols.append(dsa.shift(lag).loc[start:].values.astype(float))
        cols.append(dvic.shift(lag).loc[start:].values.astype(float))
    X = np.column_stack(cols)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(q)
    return y[mask], X[mask], q[mask]


# ---------------------------------------------------------------------------
# 2. Core vectorised helpers
# ---------------------------------------------------------------------------
def _ssr_batch(Y_mat: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    OLS SSR for every column of Y_mat simultaneously.
    Y_mat : (T_regime, B+1),  X : (T_regime, k)
    Returns (B+1,) SSR vector.
    """
    Q, _ = np.linalg.qr(X, mode="reduced")   # (T_regime, k)
    fitted = Q @ (Q.T @ Y_mat)                # (T_regime, B+1)  ← BLAS dgemm
    resid  = Y_mat - fitted
    return np.einsum("ij,ij->j", resid, resid)


def _bootstrap_yall(y, X, n_boot, rng):
    """Generate Y_all = [y_obs | B bootstrap samples] using wild Rademacher."""
    T = len(y)
    b_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid     = y - X @ b_ols
    xi_mat    = rng.choice([-1.0, 1.0], size=(T, n_boot))
    Y_all     = np.empty((T, n_boot + 1))
    Y_all[:, 0]  = y
    Y_all[:, 1:] = (X @ b_ols)[:, None] + resid[:, None] * xi_mat
    return Y_all, b_ols, resid


def _restricted_ssr_all(Y_all, X):
    """Restricted (linear) SSR for every column of Y_all."""
    Q, _ = np.linalg.qr(X, mode="reduced")
    proj  = Q @ (Q.T @ Y_all)
    return (np.einsum("ij,ij->j", Y_all, Y_all)
          - np.einsum("ij,ij->j", proj,  proj))


def _f_stat_from_ssr(SSR_R_all, SSR_U_min, T):
    return (SSR_R_all - SSR_U_min) / SSR_R_all * T


def _pvalue(F_obs, F_boot, n_boot):
    return float((np.sum(F_boot >= F_obs) + 1) / (n_boot + 1))


def _model_aic_bic(ssr, T, k_total):
    """AIC and BIC for a regime-switching OLS model."""
    sigma2 = ssr / T
    ll     = -0.5 * T * (np.log(2 * np.pi * sigma2) + 1)
    aic    = -2 * ll + 2 * k_total
    bic    = -2 * ll + np.log(T) * k_total
    return aic, bic


# ---------------------------------------------------------------------------
# 3. Symmetric Band Test
# ---------------------------------------------------------------------------
def symmetric_band_test(y, X, q,
                        trim=TRIM, n_gamma=N_GAMMA_SYM, n_boot=N_BOOT_SYM):
    """
    Balke-Fomby (1997) symmetric band test.

    Two regimes:
      Inner : |ECT_{t-1}| ≤ γ  (near equilibrium, slow adjustment)
      Outer : |ECT_{t-1}| > γ  (large deviation, fast adjustment)

    Grid: γ ∈ [trim-quantile, (1-trim)-quantile] of |ECT|
    Bootstrap: Rademacher wild bootstrap (vectorised across all draws per γ)
    """
    T, k = X.shape
    rng   = np.random.default_rng(SEED)
    abs_q = np.abs(q)

    # gamma grid: search within the |ECT| distribution
    g_lo, g_hi = np.quantile(abs_q, [trim, 1 - trim])
    gamma_grid  = np.linspace(g_lo, g_hi, n_gamma)
    print(f"    |ECT| grid: [{g_lo:.3f}, {g_hi:.3f}]  ({n_gamma} pts)")

    Y_all, b_ols, resid = _bootstrap_yall(y, X, n_boot, rng)
    SSR_R_all = _restricted_ssr_all(Y_all, X)

    # sweep
    SSR_U_mat = np.full((n_gamma, n_boot + 1), np.inf)
    for i, gamma in enumerate(gamma_grid):
        m_in  = abs_q <= gamma
        m_out = ~m_in
        if m_in.sum() <= k or m_out.sum() <= k:
            continue
        SSR_U_mat[i] = (_ssr_batch(Y_all[m_in],  X[m_in])
                      + _ssr_batch(Y_all[m_out], X[m_out]))

    SSR_U_min = np.nanmin(SSR_U_mat, axis=0)          # (n_boot+1,)
    F_all     = _f_stat_from_ssr(SSR_R_all, SSR_U_min, T)
    F_obs     = float(F_all[0])
    p_val     = _pvalue(F_obs, F_all[1:], n_boot)

    best_i    = int(np.nanargmin(SSR_U_mat[:, 0]))
    gamma_hat = float(gamma_grid[best_i])
    ssr_best  = float(SSR_U_mat[best_i, 0])
    ssr_R     = float(SSR_R_all[0])

    m_in_hat  = abs_q <= gamma_hat
    m_out_hat = ~m_in_hat

    # regime-level AIC/BIC (total k_total = 2k params for 2 regimes)
    aic, bic = _model_aic_bic(ssr_best, T, k_total=2 * k)

    boot_cvs = {pct: float(np.percentile(F_all[1:], pct))
                for pct in [80, 90, 95, 99]}

    return dict(
        test_name      = "Symmetric Band (|ECT| ≤ γ vs outer)",
        sup_F          = F_obs,
        gamma_hat      = gamma_hat,        # the single threshold on |ECT|
        p_value        = p_val,
        ssr_linear     = ssr_R,
        ssr_threshold  = ssr_best,
        var_red_pct    = (ssr_R - ssr_best) / ssr_R * 100,
        n_inner        = int(m_in_hat.sum()),
        n_outer        = int(m_out_hat.sum()),
        T              = T,
        k              = k,
        n_boot         = n_boot,
        aic            = aic,
        bic            = bic,
        boot_cvs       = boot_cvs,
        F_profile      = _f_stat_from_ssr(SSR_R_all[0], SSR_U_mat[:, 0], T),
        gamma_grid     = gamma_grid,
    )


# ---------------------------------------------------------------------------
# 4. Two-Threshold (General Band) Test
# ---------------------------------------------------------------------------
def two_threshold_test(y, X, q,
                       trim_per_regime=TRIM_3R,
                       n_gamma=N_GAMMA_2T, n_boot=N_BOOT_2T):
    """
    Two-threshold (3-regime) TVECM test.

    Three regimes:
      Low  : ECT < γ_L          (SA price much lower than VIC → importing)
      Band : γ_L ≤ ECT ≤ γ_U   (near equilibrium, slow/no adjustment)
      High : ECT > γ_U          (SA price premium → exporting)

    Grid: (γ_L, γ_U) pairs with γ_L < γ_U, each regime ≥ trim_per_regime*T obs.
    Bootstrap: Rademacher wild bootstrap (vectorised).

    Asymmetry is economically meaningful:
      - V-SA import capacity ≠ export capacity
      - SA wind surplus drives negative ECT (low regime) differently
        from SA demand peak driving positive ECT (high regime)
    """
    T, k = X.shape
    rng   = np.random.default_rng(SEED)
    min_n = max(k + 1, int(trim_per_regime * T))

    # Grid: search within [trim, 1-trim] range of ECT
    q_lo, q_hi   = np.quantile(q, [trim_per_regime, 1 - trim_per_regime])
    gamma_L_grid = np.linspace(q_lo, q_hi, n_gamma)
    gamma_U_grid = np.linspace(q_lo, q_hi, n_gamma)
    n_pairs_total = n_gamma * (n_gamma - 1) // 2
    print(f"    ECT grid: [{q_lo:.3f}, {q_hi:.3f}]  "
          f"({n_gamma}×{n_gamma} = up to {n_pairs_total} pairs)")

    Y_all, b_ols, resid = _bootstrap_yall(y, X, n_boot, rng)
    SSR_R_all = _restricted_ssr_all(Y_all, X)

    # Sweep 2D grid — track running minimum across pairs per bootstrap draw
    SSR_U_min = np.full(n_boot + 1, np.inf)
    gamma_L_hat = np.nan
    gamma_U_hat = np.nan
    ssr_obs_best = np.inf
    n_valid = 0

    for gL in gamma_L_grid:
        for gU in gamma_U_grid:
            if gL >= gU:
                continue
            m_lo  = q < gL
            m_mid = (q >= gL) & (q <= gU)
            m_hi  = q > gU
            if m_lo.sum() < min_n or m_mid.sum() < min_n or m_hi.sum() < min_n:
                continue
            n_valid += 1

            ssr_pair = (_ssr_batch(Y_all[m_lo],  X[m_lo])
                      + _ssr_batch(Y_all[m_mid], X[m_mid])
                      + _ssr_batch(Y_all[m_hi],  X[m_hi]))

            # update running minimum per bootstrap column
            improved = ssr_pair < SSR_U_min
            SSR_U_min = np.where(improved, ssr_pair, SSR_U_min)

            # track best pair for observed data (col 0)
            if ssr_pair[0] < ssr_obs_best:
                ssr_obs_best = ssr_pair[0]
                gamma_L_hat  = gL
                gamma_U_hat  = gU

    print(f"    Valid pairs evaluated: {n_valid:,}")

    if n_valid == 0:
        print("    WARNING: no valid pairs found — check trim settings.")
        return None

    F_all = _f_stat_from_ssr(SSR_R_all, SSR_U_min, T)
    F_obs = float(F_all[0])
    p_val = _pvalue(F_obs, F_all[1:], n_boot)

    ssr_R = float(SSR_R_all[0])
    # regime sizes at best pair
    m_lo_hat  = q < gamma_L_hat
    m_mid_hat = (q >= gamma_L_hat) & (q <= gamma_U_hat)
    m_hi_hat  = q > gamma_U_hat

    aic, bic = _model_aic_bic(ssr_obs_best, T, k_total=3 * k)

    boot_cvs = {pct: float(np.percentile(F_all[1:], pct))
                for pct in [80, 90, 95, 99]}

    return dict(
        test_name     = "Two-Threshold Band (3 regimes: low/band/high)",
        sup_F         = F_obs,
        gamma_L_hat   = float(gamma_L_hat),
        gamma_U_hat   = float(gamma_U_hat),
        p_value       = p_val,
        ssr_linear    = ssr_R,
        ssr_threshold = ssr_obs_best,
        var_red_pct   = (ssr_R - ssr_obs_best) / ssr_R * 100,
        n_low         = int(m_lo_hat.sum()),
        n_band        = int(m_mid_hat.sum()),
        n_high        = int(m_hi_hat.sum()),
        T             = T,
        k             = k,
        n_boot        = n_boot,
        n_valid_pairs = n_valid,
        aic           = aic,
        bic           = bic,
        boot_cvs      = boot_cvs,
    )


# ---------------------------------------------------------------------------
# 5. Comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(r_single, r_sym, r_two):
    """
    Compare all three Hansen tests side-by-side.
    r_single: dict from phase1_diagnostics (single threshold, carried in manually)
    r_sym   : result from symmetric_band_test
    r_two   : result from two_threshold_test
    """
    print("\n" + DIVIDER)
    print("  HANSEN TEST COMPARISON — ALL THREE SPECIFICATIONS")
    print(DIVIDER)
    rows = []
    for r, name, n_regimes in [
        (r_single, "Single Threshold (ECT ≤ γ vs > γ)",     2),
        (r_sym,    "Symmetric Band  (|ECT| ≤ γ vs outer)", 2),
        (r_two,    "Two-Threshold   (low / band / high)",   3),
    ]:
        if r is None:
            continue
        rows.append({
            "Model":        name,
            "Regimes":      n_regimes,
            "Sup-F":        r["sup_F"],
            "p-value":      r["p_value"],
            "CV 5%":        r["boot_cvs"].get(95, np.nan),
            "Var red %":    r["var_red_pct"],
            "AIC":          r.get("aic", np.nan),
            "BIC":          r.get("bic", np.nan),
        })

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_p   = tbl.loc[tbl["p-value"].idxmin(), "Model"]
    best_aic = tbl.loc[tbl["AIC"].idxmin(),     "Model"] if "AIC" in tbl else "—"
    best_bic = tbl.loc[tbl["BIC"].idxmin(),     "Model"] if "BIC" in tbl else "—"
    print(f"\n  Lowest p-value  → {best_p}")
    print(f"  Lowest AIC      → {best_aic}")
    print(f"  Lowest BIC      → {best_bic}")

    tbl.to_csv(RESULTS_DIR / "hansen_comparison.csv", index=False)
    print(f"\n  Saved: results/hansen_comparison.csv")


def print_sym_results(r):
    print("\n" + DIVIDER)
    print("  SYMMETRIC BAND TEST RESULTS")
    print(DIVIDER)
    cvs = r["boot_cvs"]
    pct_inner = r["n_inner"] / r["T"] * 100
    pct_outer = r["n_outer"] / r["T"] * 100
    print(f"  Sup-F statistic       : {r['sup_F']:>12.4f}")
    print(f"  Estimated threshold γ̂ : {r['gamma_hat']:>12.4f}  (|ECT| units)")
    print(f"  Bootstrap p-value     : {r['p_value']:>12.4f}")
    print(f"  Bootstrap critical values:")
    print(f"    20% cv = {cvs[80]:>8.4f}")
    print(f"    10% cv = {cvs[90]:>8.4f}")
    print(f"     5% cv = {cvs[95]:>8.4f}")
    print(f"     1% cv = {cvs[99]:>8.4f}")
    print(f"  SSR linear model      : {r['ssr_linear']:>14.2f}")
    print(f"  SSR band model        : {r['ssr_threshold']:>14.2f}")
    print(f"  Variance reduction    : {r['var_red_pct']:>10.2f}%")
    print(f"  Inner band (|ECT|≤γ̂)  : {r['n_inner']:>8,}  obs  ({pct_inner:.1f}%)")
    print(f"  Outer regime (|ECT|>γ̂): {r['n_outer']:>8,}  obs  ({pct_outer:.1f}%)")
    print(f"  AIC                   : {r['aic']:>14.2f}")
    print(f"  BIC                   : {r['bic']:>14.2f}")
    print()

    p = r["p_value"]
    if   p < 0.01: v = "STRONGLY REJECT H0 (p<0.01)."
    elif p < 0.05: v = "REJECT H0 (p<0.05)."
    elif p < 0.10: v = "WEAKLY REJECT H0 (p<0.10)."
    else:          v = "FAIL TO REJECT H0 (p≥0.10)."
    print(f"  Verdict: {v}")
    print(f"\n  Economic interpretation:")
    print(f"    Inner band (|ECT| ≤ {r['gamma_hat']:.2f} $/MWh):")
    print(f"      SA and VIC prices within the no-arbitrage band.")
    print(f"      Interconnector flows insufficient to trigger price convergence.")
    print(f"    Outer regime (|ECT| > {r['gamma_hat']:.2f} $/MWh):")
    print(f"      Large SA-VIC spread: interconnector saturates or export/import spikes.")
    print(f"      Fast error correction toward long-run parity.")


def print_two_threshold_results(r):
    print("\n" + DIVIDER)
    print("  TWO-THRESHOLD (3-REGIME) TEST RESULTS")
    print(DIVIDER)
    cvs = r["boot_cvs"]
    T   = r["T"]
    print(f"  Sup-F statistic       : {r['sup_F']:>12.4f}")
    print(f"  Estimated thresholds  : γ̂_L = {r['gamma_L_hat']:.4f}  |  "
          f"γ̂_U = {r['gamma_U_hat']:.4f}  (ECT units)")
    print(f"  Bootstrap p-value     : {r['p_value']:>12.4f}")
    print(f"  Bootstrap critical values:")
    print(f"    20% cv = {cvs[80]:>8.4f}")
    print(f"    10% cv = {cvs[90]:>8.4f}")
    print(f"     5% cv = {cvs[95]:>8.4f}")
    print(f"     1% cv = {cvs[99]:>8.4f}")
    print(f"  SSR linear model      : {r['ssr_linear']:>14.2f}")
    print(f"  SSR 3-regime model    : {r['ssr_threshold']:>14.2f}")
    print(f"  Variance reduction    : {r['var_red_pct']:>10.2f}%")
    print(f"  Regime 1 (ECT<γ̂_L)   : {r['n_low']:>8,}  obs  ({r['n_low']/T*100:.1f}%)")
    print(f"  Regime 2 (band)       : {r['n_band']:>8,}  obs  ({r['n_band']/T*100:.1f}%)")
    print(f"  Regime 3 (ECT>γ̂_U)   : {r['n_high']:>8,}  obs  ({r['n_high']/T*100:.1f}%)")
    print(f"  Valid pairs evaluated : {r['n_valid_pairs']:>8,}")
    print(f"  AIC                   : {r['aic']:>14.2f}")
    print(f"  BIC                   : {r['bic']:>14.2f}")
    print()

    p = r["p_value"]
    if   p < 0.01: v = "STRONGLY REJECT H0 (p<0.01)."
    elif p < 0.05: v = "REJECT H0 (p<0.05)."
    elif p < 0.10: v = "WEAKLY REJECT H0 (p<0.10)."
    else:          v = "FAIL TO REJECT H0 (p≥0.10)."
    print(f"  Verdict: {v}")
    print(f"\n  Economic interpretation:")
    gL, gU = r['gamma_L_hat'], r['gamma_U_hat']
    print(f"    Regime 1 (ECT < {gL:.2f}):")
    print(f"      SA price well below VIC → SA is net importer.")
    print(f"      Asymmetric adjustment: strong upward correction in SA.")
    print(f"    Regime 2 (band: {gL:.2f} ≤ ECT ≤ {gU:.2f}):")
    print(f"      Near-equilibrium: slow or no adjustment.")
    print(f"      Consistent with no-arbitrage band from transmission costs.")
    print(f"    Regime 3 (ECT > {gU:.2f}):")
    print(f"      SA price premium over VIC → SA is net exporter.")
    print(f"      Asymmetric adjustment: correction from both sides.")


def save_results(r_sym, r_two):
    lines = ["BAND THRESHOLD TEST RESULTS", "=" * 40]
    for r in [r_sym, r_two]:
        if r is None:
            continue
        lines += [
            f"\n{r['test_name']}",
            f"  Sup-F       : {r['sup_F']:.4f}",
            f"  p-value     : {r['p_value']:.4f}",
            f"  CV 5%       : {r['boot_cvs'][95]:.4f}",
            f"  Var red %   : {r['var_red_pct']:.2f}%",
            f"  AIC         : {r.get('aic', np.nan):.2f}",
            f"  BIC         : {r.get('bic', np.nan):.2f}",
        ]
        if "gamma_hat" in r:
            lines.append(f"  γ̂           : {r['gamma_hat']:.4f}")
        if "gamma_L_hat" in r:
            lines.append(f"  γ̂_L, γ̂_U   : {r['gamma_L_hat']:.4f}, {r['gamma_U_hat']:.4f}")
    (RESULTS_DIR / "band_threshold_tests.txt").write_text("\n".join(lines))
    print(f"\n  Saved: results/band_threshold_tests.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(DIVIDER)
    print("  PHASE 1 EXTENSION: BAND THRESHOLD TESTS")
    print("  SA vs VIC Electricity Prices — 2023–2025")
    print(DIVIDER)

    # ---- data ----
    print("\n[1] Loading data + estimating ECT ...")
    prices, beta, ect = load_and_prepare()
    y, X, q = build_ecm(prices, ect, ECM_LAGS)
    print(f"  Cointegrating vector: sa_price − {-beta[1]:.4f} × vic_price")
    print(f"  ECM sample: T = {len(y):,},  k = {X.shape[1]}")
    print(f"  ECT: mean={q.mean():.2f}  std={q.std():.2f}  "
          f"p15={np.percentile(q,15):.2f}  p85={np.percentile(q,85):.2f}")
    print(f"  |ECT|: mean={np.abs(q).mean():.2f}  "
          f"p15={np.percentile(np.abs(q),15):.2f}  p85={np.percentile(np.abs(q),85):.2f}")

    # ---- single-threshold reference (from phase1_diagnostics.py) ----
    r_single = dict(
        sup_F        = 1545.25,
        p_value      = 0.056,
        var_red_pct  = 2.94,
        ssr_linear   = 3034539910.91,
        ssr_threshold= 2945365995.72,
        boot_cvs     = {80: np.nan, 90: 1355.22, 95: 1571.10, 99: 2086.65},
        aic          = np.nan,    # not computed in phase1 script
        bic          = np.nan,
    )

    # ---- symmetric band test ----
    print(f"\n[2] Symmetric Band Test  ({N_BOOT_SYM} bootstrap reps, {N_GAMMA_SYM} grid pts) ...")
    t0 = time.time()
    r_sym = symmetric_band_test(y, X, q, trim=TRIM,
                                n_gamma=N_GAMMA_SYM, n_boot=N_BOOT_SYM)
    print(f"    Done in {time.time()-t0:.1f}s")
    print_sym_results(r_sym)

    # ---- two-threshold test ----
    print(f"\n[3] Two-Threshold Test  ({N_BOOT_2T} bootstrap reps, {N_GAMMA_2T}×{N_GAMMA_2T} grid) ...")
    t0 = time.time()
    r_two = two_threshold_test(y, X, q, trim_per_regime=TRIM_3R,
                               n_gamma=N_GAMMA_2T, n_boot=N_BOOT_2T)
    print(f"    Done in {time.time()-t0:.1f}s")
    if r_two:
        print_two_threshold_results(r_two)

    # ---- comparison table ----
    print_comparison_table(r_single, r_sym, r_two)

    # ---- save ----
    print("\n[4] Saving ...")
    save_results(r_sym, r_two)

    print("\n" + DIVIDER)
    print("  GUIDANCE FOR THE PAPER")
    print(DIVIDER)
    print("""
  Use the specification with the lowest bootstrap p-value AND lowest BIC.

  If Symmetric Band wins:
    → TVECM with |ECT| threshold (Balke-Fomby 1997 neutral-band model).
    → Write: "Price adjustment is absent within the no-arbitrage band
      (|ECT| ≤ γ̂) and fast outside it, consistent with transaction-cost
      theory of price transmission (Balke & Fomby, 1997)."

  If Two-Threshold wins:
    → TVECM with asymmetric thresholds γ_L, γ_U.
    → Write: "The V-SA interconnector imposes asymmetric adjustment limits:
      the lower threshold (γ_L) reflects the import capacity constraint
      while the upper threshold (γ_U) reflects the export constraint,
      motivating a three-regime TVECM (Hansen & Seo, 2002)."

  In either case, Phase 2 will estimate the full TVECM with ECM lags,
  regime-specific adjustment coefficients, and battery interaction terms.
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
