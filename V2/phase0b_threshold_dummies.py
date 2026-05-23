#!/usr/bin/env python3
"""
Phase 0b — Two-Threshold Hansen-Seo Test: With vs Without Dummy Controls
=========================================================================
Question: Does the two-threshold p-value (baseline = 0.040) change when
calendar/seasonal dummy variables are included as exogenous controls in
the ECM regression?

Design
------
The Hansen-Seo Sup-F test compares:
  H0: linear VECM   (one regime)
  H1: 3-regime TVECM (two thresholds on ECT)

Both H0 and H1 use the same regressors X. If dummies are added to X,
they enter BOTH the null and the alternative, so the F-statistic isolates
only the ADDITIONAL variance explained by the threshold structure, beyond
what the dummies already capture.

Two runs:
  (A) Baseline  : X = [1, ECT_{t-1}, Δsa lags × 4, Δvic lags × 4]
                  → replicates phase1_band_threshold.py (p = 0.040)
  (B) With dummies: X_augmented = X_baseline + [holiday, weekend,
                  day-of-week, quarterly, monthly, time-of-day dummies]

Comparison tells us whether the threshold finding is robust to deterministic
seasonal patterns, or whether those patterns were inflating the Sup-F.

Output
------
  results/phase0b_threshold_dummies.txt
  results/phase0b_threshold_comparison.csv
"""

import os, warnings, datetime, time
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMEXPR_DISABLED", "1")

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ECM_LAGS     = 4
TRIM_3R      = 0.10
N_GAMMA_2T   = 40       # 40×40 grid → ~780 valid pairs (same as phase1)
N_BOOT_2T    = 399      # same as phase1
SEED         = 42
DIVIDER      = "=" * 72
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Holiday helpers (same as phase0_data_analytics.py)
# ─────────────────────────────────────────────────────────────────────────────
def _easter(year):
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day   = ((h + l - 7*m + 114) % 31) + 1
    return datetime.date(year, month, day)

def _nth_weekday(year, month, weekday, n):
    d = datetime.date(year, month, 1)
    delta = (weekday - d.weekday()) % 7
    first = d + datetime.timedelta(days=delta)
    return first + datetime.timedelta(weeks=n - 1)

def get_sa_public_holidays(years):
    holidays = set()
    for yr in years:
        easter = _easter(yr)
        for mo, dy in [(1,1),(1,26),(4,25),(12,25),(12,26)]:
            holidays.add(datetime.date(yr, mo, dy))
        for delta in [-2, -1, +1]:
            holidays.add(easter + datetime.timedelta(days=delta))
        holidays.add(_nth_weekday(yr, 5, 0, 2))   # Adelaide Cup (2nd Mon May)
        holidays.add(_nth_weekday(yr, 6, 0, 2))   # King's Birthday (2nd Mon Jun)
        holidays.add(_nth_weekday(yr, 9, 2, 3))   # Royal Adelaide Show (3rd Wed Sep)
        holidays.add(_nth_weekday(yr, 10, 0, 1))  # Labour Day (1st Mon Oct)
    return holidays

# ─────────────────────────────────────────────────────────────────────────────
# Data loading and ECT estimation
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime")
    prices = df[["sa_price","vic_price"]].dropna()
    prices = prices[(prices["sa_price"].abs() < 15001) &
                    (prices["vic_price"].abs() < 15001)].copy()

    res  = coint_johansen(prices[["sa_price","vic_price"]].values,
                          det_order=0, k_ar_diff=ECM_LAGS)
    beta = res.evec[:, 0]
    beta = beta / beta[0]
    ect  = pd.Series(prices[["sa_price","vic_price"]].values @ beta,
                     index=prices.index, name="ECT")
    return df, prices, beta, ect

# ─────────────────────────────────────────────────────────────────────────────
# Build full dummy matrix (aligned to full df index)
# ─────────────────────────────────────────────────────────────────────────────
def build_dummy_matrix(idx):
    """
    Returns (T, K) numpy array of dummy controls:
      - d_holiday, d_holiday_eve, d_weekend, school_holiday
      - dow_Mon … dow_Sat  (Sun = base)
      - q_Q1, q_Q2, q_Q3  (Q4 = base)
      - m_01 … m_11        (Dec = base)
      - tod_morning, tod_solar_peak, tod_evening, tod_night  (overnight = base)
    """
    years    = sorted(set(idx.year.tolist()))
    pub_hols = get_sa_public_holidays(years)
    hol_eve  = {d - datetime.timedelta(days=1) for d in pub_hols}
    dates    = idx.normalize()
    hours    = idx.hour

    cols = {}

    # Holiday flags
    cols["d_holiday"]     = dates.map(lambda d: int(d.date() in pub_hols)).values
    cols["d_holiday_eve"] = dates.map(lambda d: int(d.date() in hol_eve)).values
    cols["d_weekend"]     = np.array((idx.dayofweek >= 5).astype(int))

    # School holidays (approximate SA terms)
    sch = np.zeros(len(idx), dtype=int)
    for yr in years:
        m, d = idx.month, idx.day
        sch |= np.array((idx.year==yr) & (m==1) & (d<=26), dtype=int)
        sch |= np.array((idx.year==yr) & (m==4) & (d>=12) & (d<=26), dtype=int)
        sch |= np.array((idx.year==yr) & (m==7) & (d<=16), dtype=int)
        sch |= np.array((idx.year==yr) & (m==9) & (d>=27), dtype=int)
        sch |= np.array((idx.year==yr) & (m==10) & (d<=15), dtype=int)
        sch |= np.array((idx.year==yr) & (m==12) & (d>=15), dtype=int)
    cols["school_holiday"] = sch

    # Day-of-week FE (Sunday = base)
    dow_names = ["dow_Mon","dow_Tue","dow_Wed","dow_Thu","dow_Fri","dow_Sat"]
    for i, nm in enumerate(dow_names):
        cols[nm] = np.array((idx.dayofweek == i).astype(int))

    # Quarter dummies (Q4 = base)
    for q in [1,2,3]:
        cols[f"q_Q{q}"] = np.array((idx.quarter == q).astype(int))

    # Month dummies (Dec = base)
    for m in range(1,12):
        cols[f"m_{m:02d}"] = np.array((idx.month == m).astype(int))

    # Time-of-day buckets (overnight = base)
    cols["tod_morning"]    = np.array(((hours >= 6)  & (hours <= 9 )).astype(int))
    cols["tod_solar_peak"] = np.array(((hours >= 10) & (hours <= 15)).astype(int))
    cols["tod_evening"]    = np.array(((hours >= 16) & (hours <= 21)).astype(int))
    cols["tod_night"]      = np.array(((hours >= 22) & (hours <= 23)).astype(int))

    D = np.column_stack(list(cols.values()))
    return D, list(cols.keys())

# ─────────────────────────────────────────────────────────────────────────────
# ECM matrix builder (baseline and augmented)
# ─────────────────────────────────────────────────────────────────────────────
def build_ecm(prices, ect, dummy_full=None, n_lags=ECM_LAGS):
    """
    Build (y, X, q) for the Hansen-Seo test.
    dummy_full : (N_full, K) array aligned to prices.index, or None for baseline.

    y : Δsa_price (T,)
    X : [1, ECT_{t-1}, Δsa lags, Δvic lags] (+ dummies if supplied)  (T, k)
    q : ECT_{t-1} (T,)   ← the threshold variable
    """
    sa  = prices["sa_price"]
    vic = prices["vic_price"]
    dsa = sa.diff()
    dvic= vic.diff()

    start = prices.index[n_lags + 1]

    y = dsa.loc[start:].values.astype(float)
    q = ect.shift(1).loc[start:].values.astype(float)

    cols = [np.ones(len(y)), q]
    for lag in range(1, n_lags + 1):
        cols.append(dsa.shift(lag).loc[start:].values.astype(float))
        cols.append(dvic.shift(lag).loc[start:].values.astype(float))

    if dummy_full is not None:
        # Align dummy_full to the 'start:' slice
        n_skip = prices.index.get_loc(start)
        D_slice = dummy_full[n_skip:, :]
        cols.append(D_slice)

    X = np.column_stack(cols)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(q)
    return y[mask], X[mask], q[mask]

# ─────────────────────────────────────────────────────────────────────────────
# Core vectorised Hansen-Seo engine (identical to phase1_band_threshold.py)
# ─────────────────────────────────────────────────────────────────────────────
def _ssr_batch(Y_mat, X):
    Q, _ = np.linalg.qr(X, mode="reduced")
    fitted = Q @ (Q.T @ Y_mat)
    resid  = Y_mat - fitted
    return np.einsum("ij,ij->j", resid, resid)

def _bootstrap_yall(y, X, n_boot, rng):
    T = len(y)
    b_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid     = y - X @ b_ols
    xi_mat    = rng.choice([-1.0, 1.0], size=(T, n_boot))
    Y_all     = np.empty((T, n_boot + 1))
    Y_all[:,0]  = y
    Y_all[:,1:] = (X @ b_ols)[:,None] + resid[:,None] * xi_mat
    return Y_all, b_ols, resid

def _restricted_ssr_all(Y_all, X):
    Q, _ = np.linalg.qr(X, mode="reduced")
    proj  = Q @ (Q.T @ Y_all)
    return (np.einsum("ij,ij->j", Y_all, Y_all)
          - np.einsum("ij,ij->j", proj,  proj))

def _pvalue(F_obs, F_boot, n_boot):
    return float((np.sum(F_boot >= F_obs) + 1) / (n_boot + 1))

def _model_aic_bic(ssr, T, k_total):
    sigma2 = ssr / T
    ll     = -0.5 * T * (np.log(2 * np.pi * sigma2) + 1)
    aic    = -2 * ll + 2 * k_total
    bic    = -2 * ll + np.log(T) * k_total
    return aic, bic

# ─────────────────────────────────────────────────────────────────────────────
# Two-threshold test
# ─────────────────────────────────────────────────────────────────────────────
def two_threshold_test(y, X, q, label="baseline",
                       trim=TRIM_3R, n_gamma=N_GAMMA_2T, n_boot=N_BOOT_2T):
    T, k = X.shape
    rng   = np.random.default_rng(SEED)
    min_n = max(k + 1, int(trim * T))

    q_lo, q_hi   = np.quantile(q, [trim, 1 - trim])
    gamma_L_grid = np.linspace(q_lo, q_hi, n_gamma)
    gamma_U_grid = np.linspace(q_lo, q_hi, n_gamma)
    n_pairs_total = n_gamma * (n_gamma - 1) // 2

    print(f"\n  [{label}]  T={T:,}  k={k}  grid up to {n_pairs_total} pairs  "
          f"boot={n_boot}")

    Y_all, b_ols, resid = _bootstrap_yall(y, X, n_boot, rng)
    SSR_R_all = _restricted_ssr_all(Y_all, X)

    SSR_U_min    = np.full(n_boot + 1, np.inf)
    gamma_L_hat  = gamma_U_hat = np.nan
    ssr_obs_best = np.inf
    n_valid      = 0

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

            improved   = ssr_pair < SSR_U_min
            SSR_U_min  = np.where(improved, ssr_pair, SSR_U_min)
            if ssr_pair[0] < ssr_obs_best:
                ssr_obs_best = ssr_pair[0];  gamma_L_hat = gL;  gamma_U_hat = gU

    if n_valid == 0:
        print("    WARNING: no valid pairs found.")
        return None

    F_all = (SSR_R_all - SSR_U_min) / SSR_R_all * T
    F_obs = float(F_all[0])
    p_val = _pvalue(F_obs, F_all[1:], n_boot)

    m_lo_hat  = q < gamma_L_hat
    m_mid_hat = (q >= gamma_L_hat) & (q <= gamma_U_hat)
    m_hi_hat  = q > gamma_U_hat
    ssr_R     = float(SSR_R_all[0])
    aic, bic  = _model_aic_bic(ssr_obs_best, T, k_total=3 * k)

    boot_cvs = {pct: float(np.percentile(F_all[1:], pct))
                for pct in [80, 90, 95, 99]}

    stars = ("***" if p_val < 0.01 else "**" if p_val < 0.05
             else "*" if p_val < 0.10 else "")
    print(f"    Sup-F = {F_obs:.2f}   p = {p_val:.4f}{stars}   "
          f"CV5% = {boot_cvs[95]:.2f}")
    print(f"    γ_L = {gamma_L_hat:.2f}  γ_U = {gamma_U_hat:.2f}  "
          f"n1={m_lo_hat.sum():,} n2={m_mid_hat.sum():,} n3={m_hi_hat.sum():,}  "
          f"valid pairs={n_valid:,}")
    print(f"    AIC = {aic:.0f}   BIC = {bic:.0f}   "
          f"var_red = {(ssr_R-ssr_obs_best)/ssr_R*100:.2f}%")

    return dict(label=label, sup_F=F_obs, p_value=p_val,
                gamma_L=gamma_L_hat, gamma_U=gamma_U_hat,
                n_low=int(m_lo_hat.sum()), n_band=int(m_mid_hat.sum()),
                n_high=int(m_hi_hat.sum()), T=T, k=k,
                n_valid_pairs=n_valid, aic=aic, bic=bic,
                boot_cvs=boot_cvs, var_red_pct=(ssr_R-ssr_obs_best)/ssr_R*100,
                ssr_linear=ssr_R, ssr_threshold=ssr_obs_best)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(DIVIDER)
    print("  PHASE 0b — TWO-THRESHOLD TEST: WITH vs WITHOUT DUMMY CONTROLS")
    print(DIVIDER)

    print("\n[1] Loading data and estimating ECT …")
    df, prices, beta, ect = load_and_prepare()
    print(f"    Cointegrating vector: sa − {-beta[1]:.4f}×vic")
    print(f"    ECT: mean={ect.mean():.2f}  std={ect.std():.2f}")

    print("\n[2] Building dummy matrix …")
    D_full, d_names = build_dummy_matrix(prices.index)
    print(f"    {D_full.shape[1]} dummy columns: {', '.join(d_names[:6])} …")

    # ── Run A: baseline (no dummies) ─────────────────────────────────────────
    print("\n[3] Run A — BASELINE (no dummies in X)")
    print("    Replicates phase1_band_threshold.py")
    y_b, X_b, q_b = build_ecm(prices, ect, dummy_full=None)
    t0 = time.time()
    r_base = two_threshold_test(y_b, X_b, q_b, label="baseline (no dummies)",
                                n_boot=N_BOOT_2T, n_gamma=N_GAMMA_2T)
    print(f"    Elapsed: {time.time()-t0:.1f}s")

    # ── Run B: with all dummies ───────────────────────────────────────────────
    print("\n[4] Run B — WITH DUMMIES (holiday, weekend, seasonal, time-of-day)")
    y_d, X_d, q_d = build_ecm(prices, ect, dummy_full=D_full)
    t0 = time.time()
    r_dum = two_threshold_test(y_d, X_d, q_d, label="with dummies",
                               n_boot=N_BOOT_2T, n_gamma=N_GAMMA_2T)
    print(f"    Elapsed: {time.time()-t0:.1f}s")

    # ── Run C: holiday + weekend only (parsimonious) ─────────────────────────
    # Build a reduced dummy set: just holiday, weekend, time-of-day buckets
    print("\n[5] Run C — PARSIMONIOUS dummies (holiday + weekend + tod only)")
    hol_idx   = [d_names.index(n) for n in ["d_holiday","d_holiday_eve","d_weekend","school_holiday"]]
    tod_idx   = [d_names.index(n) for n in ["tod_morning","tod_solar_peak","tod_evening","tod_night"]]
    D_pars    = D_full[:, hol_idx + tod_idx]
    y_p, X_p, q_p = build_ecm(prices, ect, dummy_full=D_pars)
    t0 = time.time()
    r_pars = two_threshold_test(y_p, X_p, q_p, label="parsimonious dummies",
                                n_boot=N_BOOT_2T, n_gamma=N_GAMMA_2T)
    print(f"    Elapsed: {time.time()-t0:.1f}s")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  COMPARISON: TWO-THRESHOLD TEST P-VALUE ACROSS SPECIFICATIONS")
    print(DIVIDER)
    results = [r for r in [r_base, r_pars, r_dum] if r is not None]

    hdr = (f"\n  {'Specification':<30} {'k':>4} {'Sup-F':>9} {'p-value':>9} "
           f"{'CV5%':>9} {'γ_L':>8} {'γ_U':>8} {'AIC':>12} {'var_red%':>9}")
    print(hdr)
    print("  " + "-"*100)
    for r in results:
        stars = ("***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05
                 else "*" if r['p_value'] < 0.10 else "   ")
        print(f"  {r['label']:<30} {r['k']:>4} {r['sup_F']:>9.2f} "
              f"{r['p_value']:>8.4f}{stars} {r['boot_cvs'][95]:>9.2f} "
              f"{r['gamma_L']:>8.2f} {r['gamma_U']:>8.2f} "
              f"{r['aic']:>12.0f} {r['var_red_pct']:>9.2f}%")

    print()
    print("  INTERPRETATION:")
    p_base = r_base['p_value'] if r_base else np.nan
    p_full = r_dum['p_value']  if r_dum  else np.nan
    p_pars = r_pars['p_value'] if r_pars else np.nan

    direction = "LARGER" if p_full > p_base else "SMALLER"
    print(f"""
  Baseline p-value          : {p_base:.4f}
  Parsimonious dummies p    : {p_pars:.4f}
  Full dummy set p-value    : {p_full:.4f}

  The two-threshold p-value becomes {direction} after adding dummies.

  Explanation of direction:
  ─────────────────────────
  If p becomes LARGER (weaker evidence for threshold):
    → Some of the variance previously attributed to the threshold
      structure was actually driven by deterministic calendar/seasonal
      patterns (e.g. solar-peak hours systematically co-occur with
      certain ECT regimes during specific seasons). Controlling for
      these patterns reduces the apparent variance reduction from the
      threshold — the Sup-F falls, and the p-value rises.
    → Implication: the 0.040 baseline p-value was PARTIALLY inflated
      by seasonal confounds. The threshold finding is somewhat weaker
      than the naive test suggests, though the economic and ECT-based
      evidence for three regimes remains valid.

  If p becomes SMALLER (stronger evidence for threshold):
    → The dummy controls reduce residual variance in the linear model,
      making the ADDITIONAL variance reduction from the threshold
      structure RELATIVELY larger — the Sup-F rises relative to a
      tighter baseline, and the p-value falls.
    → Implication: the threshold dynamics are INDEPENDENT of seasonal
      patterns; the 0.040 baseline p-value is if anything conservative.

  In either case: the ECT-stationarity (ADF = -38.96) and the
  Johansen cointegration evidence are not affected by dummies.
  The TVECM threshold structure remains economically motivated.
""")

    # ── Save ─────────────────────────────────────────────────────────────────
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if k != "boot_cvs"})
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(RESULTS_DIR / "phase0b_threshold_comparison.csv", index=False)

    # Text summary
    lines = [DIVIDER, "PHASE 0b — THRESHOLD TEST: WITH vs WITHOUT DUMMIES", DIVIDER, ""]
    for r in results:
        stars = ("***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05
                 else "*" if r['p_value'] < 0.10 else "")
        lines += [
            f"Specification : {r['label']}",
            f"  k (regressors)  : {r['k']}",
            f"  Sup-F           : {r['sup_F']:.4f}",
            f"  Bootstrap p     : {r['p_value']:.4f}{stars}",
            f"  CV 5%           : {r['boot_cvs'][95]:.4f}",
            f"  γ_L / γ_U       : {r['gamma_L']:.4f} / {r['gamma_U']:.4f}",
            f"  AIC             : {r['aic']:.2f}",
            f"  Var reduction   : {r['var_red_pct']:.3f}%",
            f"  Regimes (n1/n2/n3): {r['n_low']:,} / {r['n_band']:,} / {r['n_high']:,}",
            "",
        ]
    (RESULTS_DIR / "phase0b_threshold_dummies.txt").write_text("\n".join(lines))

    print(DIVIDER)
    print("Outputs saved:")
    print("  results/phase0b_threshold_comparison.csv")
    print("  results/phase0b_threshold_dummies.txt")
    print(DIVIDER)

if __name__ == "__main__":
    main()
