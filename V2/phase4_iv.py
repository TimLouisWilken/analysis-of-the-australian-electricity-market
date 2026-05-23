#!/usr/bin/env python3
"""
Phase 4: Instrumental Variable (IV/2SLS) Identification of Battery Effect
==========================================================================

Problem: battery_discharging_mw is endogenous — batteries respond to price
spikes (simultaneity). Phase 3 used lagged battery (predetermined) as a
partial fix. Phase 4 provides full IV identification.

Instruments:
  Z1 (primary):  SOC_norm = capacity-normalised 12h rolling net charging (fraction),
                 lagged 1 period.
  Z2 (secondary, robustness): SOC_raw (MWh), lagged 1 — used in over-id Sargan test.
  Z3 (NEW — commissioning events): Z_newcap = total MWh of batteries commissioned
                 in the 14-day window ending at t−1 ("first active" IV).
                 Exogeneity: construction/grid-connection timelines are governed by
                 project contracts, not real-time electricity prices.
                 Relevance: newly commissioned capacity increases fleet discharge
                 ability in subsequent periods.
                 Non-zero only during 14-day windows after each in-sample
                 commissioning event (Torrens Island Aug 2023; Tailem Bend 2 Jan 2025;
                 Blyth Apr 2025; Mannum Oct 2025 — approx. 5% of observations).

Method: manual 2SLS per regime with HAC-robust standard errors
  Stage 1:  battery_dis_t = π₀ + π₁·SOC_{t-1} + X_exog·δ + v_t
  Stage 2:  Δsa_t = μ + α·ECT_{t-1} + β_dis·battery_dis_hat_t
                  + β_chg·battery_chg_t + X_exog·γ + ε_t
  HAC SEs computed on the FULL second-stage residual ε_t (not the mechanical
  residual from using fitted values) — standard textbook correction for 2SLS.

Endogeneity test: Wu-Hausman augmented regression (v_hat added to 2nd stage).

Over-identification: also run with SOC_24h_{t-1} as additional instrument;
  Sargan-Hansen J-test on the over-identified model.

Capacity schedule: 12 SA BESS facilities from BESS SA.xlsx (Templers and Bungama
  removed — not in updated source). Capacity: 284.6 MWh at Jan 2023 → 1295.6 MWh
  at Dec 2025 (4.6× growth). Key commissioning events in sample:
  Torrens Island (250 MWh, Aug 2023), Tailem Bend 2 (84 MWh, Jan 2025),
  Blyth (477 MWh, Apr 2025), Mannum (200 MWh, Oct 2025).

Outputs:
  results/phase4_firststage.csv    — first-stage partial-F for SOC_norm + Z_newcap
  results/phase4_iv_coefs.csv     — 2SLS coefficients per regime
  results/phase4_wu_hausman.csv   — endogeneity test results
  results/phase4_sargan.csv       — Sargan J-tests (both over-id models, 6 rows)
  results/phase4_summary.txt
  results/phase4_coef_comparison.csv
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
# Config — matches Phases 2 & 3
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GAMMA_L = -45.8039
GAMMA_U =  20.0215
ECM_LAGS = 4
HAC_LAGS = 12

CONTROLS = [
    "wind_mw", "solar_total_mw", "demand_mw",
    "battery_discharging_mw", "battery_charging_mw", "interconnector_mw",
]
DIVIDER = "=" * 74

# ---------------------------------------------------------------------------
# SA battery fleet capacity schedule (10 facilities — source: BESS SA.xlsx)
# "Datum der ersten Einspeisung" = date of first feed-in to grid.
# Bolivar and Happy Valley removed — not present in updated source file.
# Hornsdale split into base (2017) + extension (2020) as per spreadsheet.
# ---------------------------------------------------------------------------
BATTERY_CAPACITY = [
    # (name,                               commissioned,   MWh_effective)
    ("Hornsdale Power Reserve",            "2017-10-01",   129.0),
    ("Dalrymple North",                    "2018-12-01",     8.0),
    ("Lake Bonney",                        "2019-12-01",    52.0),
    ("Hornsdale Power Reserve Ext.",       "2020-09-01",    64.5),
    ("Adelaide Desalination Plant",        "2021-05-01",    13.0),
    ("Christies Beach",                    "2022-03-01",     4.3),
    ("Torrens Island",                     "2023-08-01",   250.0),   # in sample ← 1yr earlier
    ("Tailem Bend 2",                      "2025-01-01",    84.0),   # in sample
    ("Blyth",                              "2025-04-01",   477.0),   # in sample ← 477 MWh
    ("Mannum",                             "2025-10-01",   200.0),   # in sample
]

# Window (days) for the commissioning-event IV (Z_newcap).
# Non-zero only in the first NEWCAP_WINDOW_DAYS days after a battery goes live —
# the acceptance-testing / ramp-up period when dispatch is least price-reactive.
NEWCAP_WINDOW_DAYS = 14


# ---------------------------------------------------------------------------
# Reused helpers (identical to phase2/3)
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
    beta = res.evec[:, 0] / res.evec[0, 0]
    ect  = pd.Series(prices.values @ beta, index=prices.index, name="ECT")
    return beta, ect


def regressor_names():
    lag_names = (
        [f"dsa_l{i}"  for i in range(1, ECM_LAGS + 1)] +
        [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)]
    )
    return ["const", "ECT_lag"] + lag_names + CONTROLS


def build_capacity_series(index: pd.DatetimeIndex) -> pd.Series:
    """
    Time-varying total installed battery capacity (MWh) for SA.
    Steps up on each facility's commissioning date.
    """
    cap = pd.Series(0.0, index=index)
    for _, date, mwh in BATTERY_CAPACITY:
        cap[cap.index >= pd.Timestamp(date)] += mwh
    return cap


def build_newcap_series(index: pd.DatetimeIndex) -> pd.Series:
    """
    Commissioning-event IV (Z_newcap): for each timestamp t, the total MWh of
    batteries commissioned in the preceding NEWCAP_WINDOW_DAYS days.

    Economic rationale:
      - During the first ~2 weeks after going live, batteries run acceptance tests
        and operators calibrate their bidding algorithms; dispatch is less
        price-reactive than at steady state → SOC reflects test-protocol charging,
        not market-price optimisation → exogenous variation.
      - Construction/grid-connection timelines are governed by project contracts
        and network planning processes, not real-time electricity prices → exclusion
        restriction plausible.
      - Non-zero only for in-sample commissioning events: Torrens Island (250 MWh,
        2023-08-01), Tailem Bend 2 (84 MWh, 2025-01-01), Blyth (477 MWh,
        2025-04-01), Mannum (200 MWh, 2025-10-01).  Approx. 5% of observations.
    """
    newcap = pd.Series(0.0, index=index)
    window = pd.Timedelta(days=NEWCAP_WINDOW_DAYS)
    for _, date, mwh in BATTERY_CAPACITY:
        ts  = pd.Timestamp(date)
        end = ts + window
        mask = (index >= ts) & (index < end)
        newcap[mask] += mwh
    return newcap


def build_sample(df, ect):
    """
    Contemporaneous battery variables (endogenous — Phase 2 style) plus two
    instruments:
      SOC_norm : capacity-normalised SOC fraction (primary)
                 = rolling 12h net charging (MWh) / total installed capacity (MWh)
                 Removes growing-fleet time trend; directly comparable across 2023-2025.
      SOC_raw  : raw 12h cumulative net charging in MWh (secondary / robustness)
    Both lagged 1 period (predetermined at time of price formation).
    """
    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")
    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}
    for lag in range(1, ECM_LAGS + 1):
        rows[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows[f"dvic_l{lag}"] = dvic.shift(lag)
    for c in CONTROLS:
        rows[c] = df[c] if c in df.columns else np.nan

    # Time-varying capacity denominator
    cap = build_capacity_series(df.index)

    # Net charging (MW); ×0.5 to convert to MWh per 30-min interval
    net         = df["battery_charging_mw"] - df["battery_discharging_mw"]
    raw_mwh_12h = (net * 0.5).rolling(24, min_periods=6).sum()   # MWh over 12h

    # Capacity-normalised SOC fraction  (clipped to ±3 for numerical stability)
    soc_frac = (raw_mwh_12h / cap).clip(-3, 3)

    rows["SOC_norm"] = soc_frac.shift(1)        # primary IV
    rows["SOC_raw"]  = raw_mwh_12h.shift(1)     # secondary IV / robustness

    # Commissioning-event IV: MWh of batteries commissioned in preceding 14 days
    newcap_series    = build_newcap_series(df.index)
    rows["Z_newcap"] = newcap_series.shift(1)   # lagged 1 — predetermined

    rows["cap_mwh"]  = cap                      # pass-through for reporting

    sample = pd.DataFrame(rows).dropna()
    sample["regime"] = np.where(
        sample["ECT_lag"] < GAMMA_L, 1,
        np.where(sample["ECT_lag"] > GAMMA_U, 3, 2)
    )
    return sample


def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def hac_cov(X, resid, maxlags=HAC_LAGS):
    """Newey-West HAC sandwich covariance matrix for OLS."""
    n, k  = X.shape
    Xe    = X * resid[:, None]
    S     = Xe.T @ Xe
    for h in range(1, maxlags + 1):
        w    = 1.0 - h / (maxlags + 1.0)
        Xeh  = Xe[h:]
        Xeh0 = Xe[:-h]
        cross = Xeh0.T @ Xeh
        S    += w * (cross + cross.T)
    S *= n / (n - k)                      # finite-sample correction
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


# ---------------------------------------------------------------------------
# 2SLS implementation with HAC standard errors
# ---------------------------------------------------------------------------
def fit_2sls_hac(y, X_endog, Z_full, endog_col_idx):
    """
    Manual 2SLS with HAC-robust SEs.

    Parameters
    ----------
    y            : (n,) endogenous dependent variable (Δsa)
    X_endog      : (n, k) full regressor matrix including endogenous variable
    Z_full       : (n, k) instrument matrix — same as X_endog except the
                   endogenous column is replaced by the instrument
    endog_col_idx: integer index of the endogenous column in X_endog

    Returns
    -------
    dict with params, se, tvals, pvals, residuals, first_stage_Fstat,
         first_stage_coef, first_stage_se
    """
    n, k = X_endog.shape

    # ── Stage 1: project endogenous variable onto instruments ────────────
    # X_endog_hat = Z(Z'Z)^{-1}Z' X_endog  (project entire X onto Z space)
    ZtZinv = np.linalg.inv(Z_full.T @ Z_full)
    Pz     = Z_full @ ZtZinv @ Z_full.T      # projection matrix
    X_hat  = Pz @ X_endog                    # all columns projected

    # First-stage: regress endogenous column on Z_full alone for diagnostics
    x_end  = X_endog[:, endog_col_idx]
    z_inst = Z_full[:, endog_col_idx]        # the actual instrument column
    fs_res = OLS(x_end, Z_full).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    x_hat_fs = np.asarray(fs_res.fittedvalues)  # first-stage fitted values

    # Build X_hat2sls: replace endogenous column with first-stage fitted values
    X2sls = X_endog.copy().astype(float)
    X2sls[:, endog_col_idx] = x_hat_fs

    # ── Stage 2: OLS of y on X2sls ────────────────────────────────────────
    s2_res = OLS(y, X2sls).fit()
    beta   = s2_res.params

    # ── Correct SEs: use actual residuals, not s2_res.resid ───────────────
    # (2SLS SE must use ε = y - X_endog β, not y - X2sls β)
    resid  = y - X_endog @ beta
    cov    = hac_cov(X_endog, resid, maxlags=HAC_LAGS)
    se     = np.sqrt(np.diag(cov))
    tvals  = beta / se

    from scipy import stats
    df_res = n - k
    pvals  = 2 * stats.t.sf(np.abs(tvals), df=df_res)

    # First-stage partial F on the instrument column
    # Partial F = (increase in RSS from removing instrument) / (s² * dof_adj)
    # Approximated as the t² of the instrument in the first-stage regression
    inst_idx = endog_col_idx   # instrument is at same position in Z
    fs_t     = fs_res.tvalues[inst_idx]
    fs_F     = fs_t ** 2          # partial F ≈ t² for single instrument
    fs_coef  = fs_res.params[inst_idx]
    fs_se    = fs_res.bse[inst_idx]

    return {
        "params":    beta,
        "se":        se,
        "tvals":     tvals,
        "pvals":     pvals,
        "resid":     resid,
        "fs_F":      fs_F,
        "fs_coef":   fs_coef,
        "fs_se":     fs_se,
        "first_stage_res": fs_res,
    }


def wu_hausman_test(y, X_endog, first_stage_resid):
    """
    Wu-Hausman augmented regression test for endogeneity.
    Add first-stage residuals to the structural equation as an additional
    regressor; test H0: its coefficient = 0 (no endogeneity).
    Uses HAC SE.
    """
    X_aug = np.column_stack([X_endog, first_stage_resid])
    res   = OLS(y, X_aug).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    wh_coef = res.params[-1]
    wh_se   = res.bse[-1]
    wh_t    = res.tvalues[-1]
    wh_p    = res.pvalues[-1]
    return wh_coef, wh_se, wh_t, wh_p


def sargan_j_test(resid_2sls, Z_full):
    """
    Sargan-Hansen over-identification test.
    Regress 2SLS residuals on full instrument set Z; nR² ~ χ²(q-k_endog).
    """
    n    = len(resid_2sls)
    res  = OLS(resid_2sls, Z_full).fit()
    J    = n * res.rsquared
    from scipy import stats
    df   = 1    # over-identification: 2 instruments - 1 endogenous
    pval = stats.chi2.sf(J, df)
    return J, pval, df


# ---------------------------------------------------------------------------
# Build per-regime design matrices for 2SLS
# ---------------------------------------------------------------------------
def build_XZ(sample_r, var_names, mode="just"):
    """
    Build the regressor (X) and instrument (Z) matrices for 2SLS.

    In all modes, SOC_norm replaces the endogenous battery_discharging_mw in Z.
    Over-identified modes append a second instrument for the Sargan J-test.

    Parameters
    ----------
    mode : 'just'   → just-identified: Z has SOC_norm only  (primary IV)
           'raw'    → over-id: Z has SOC_norm + SOC_raw     (old robustness check)
           'newcap' → over-id: Z has SOC_norm + Z_newcap    (new commissioning IV)
    """
    col_x = var_names[1:]
    X = sample_r[col_x].copy()
    X.insert(0, "const", 1.0)
    X_np    = X.values.astype(float)
    dis_col = list(X.columns).index("battery_discharging_mw")

    Z_np = X_np.copy()
    Z_np[:, dis_col] = sample_r["SOC_norm"].values   # replace dis with SOC_norm

    if mode == "raw":
        Z_full = np.column_stack([Z_np, sample_r["SOC_raw"].values])
        return X_np, Z_full, dis_col, list(X.columns)
    elif mode == "newcap":
        Z_full = np.column_stack([Z_np, sample_r["Z_newcap"].values])
        return X_np, Z_full, dis_col, list(X.columns)
    else:   # mode == "just"
        return X_np, Z_np, dis_col, list(X.columns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(DIVIDER)
    print("  PHASE 4: IV/2SLS IDENTIFICATION OF BATTERY DAMPENING EFFECT")
    print(f"  Primary instrument: SOC_norm = capacity-normalised 12h net charging (lagged 1)")
    print(f"  Secondary: SOC_raw (MWh) — over-identified Sargan J-test")
    print(f"  Thresholds: γ_L = {GAMMA_L:.4f},  γ_U = {GAMMA_U:.4f}")
    print(DIVIDER)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data and computing ECT ...")
    df        = load_data()
    beta, ect = estimate_ect(df)
    print(f"  Cointegrating vector: sa_price − {-beta[1]:.4f} × vic_price")

    sample = build_sample(df, ect)
    T      = len(sample)
    print(f"  ECM sample: T = {T:,}")
    for r in [1, 2, 3]:
        n = (sample["regime"] == r).sum()
        print(f"    Regime {r}: N = {n:,}  ({n/T*100:.1f}%)")

    var_names = regressor_names()

    # ── 2. Capacity schedule + instrument diagnostics ─────────────────────
    print("\n[2] Battery capacity schedule (SA fleet) ...")
    cumul = 0.0
    print(f"  {'Facility':<28s}  {'Date':>12s}  {'MWh':>8s}  {'Total MWh':>10s}")
    print("  " + "-" * 64)
    for name, date, mwh in BATTERY_CAPACITY:
        cumul += mwh
        in_sample = "(in sample)" if date <= "2025-12-31" else "(outside)"
        print(f"  {name:<28s}  {date:>12s}  {mwh:>8.1f}  {cumul:>10.1f}  {in_sample}")
    print(f"\n  Capacity growth: {sample['cap_mwh'].min():.0f} → {sample['cap_mwh'].max():.0f} MWh "
          f"({sample['cap_mwh'].max()/sample['cap_mwh'].min():.1f}× over sample)")

    print("\n  Instrument diagnostics (full sample):")
    print(f"  {'Instrument':<35s}  {'mean':>8s}  {'std':>8s}  {'corr(dis,Z)':>12s}  "
          f"{'corr(Δsa,Z)':>12s}  {'nonzero obs':>12s}")
    print("  " + "-" * 97)
    for inst, label in [
        ("SOC_norm", "SOC_norm (fraction, primary)       "),
        ("SOC_raw",  "SOC_raw  (MWh, secondary)          "),
        ("Z_newcap", "Z_newcap (MWh, commissioning event)"),
    ]:
        m    = sample[inst].mean()
        s    = sample[inst].std()
        r_d  = sample["battery_discharging_mw"].corr(sample[inst])
        r_s  = sample["dsa"].corr(sample[inst])
        nz   = (sample[inst] != 0.0).sum()
        print(f"  {label:<35s}  {m:>8.3f}  {s:>8.3f}  {r_d:>+12.4f}  {r_s:>+12.4f}  {nz:>12d}")

    r_rel = sample["battery_discharging_mw"].corr(sample["SOC_norm"])
    r_exc = sample["dsa"].corr(sample["SOC_norm"])

    # ── 3. 2SLS per regime ────────────────────────────────────────────────
    print("\n[3] 2SLS estimation by regime ...")
    print("    Primary IV: SOC_norm | Over-id checks: SOC_norm+SOC_raw, SOC_norm+Z_newcap")

    from scipy import stats as _stats

    iv_results  = {}
    fs_rows     = []
    wh_rows     = []
    sargan_rows = []      # both over-id models stored here (with 'model' column)

    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r]
        y   = sub["dsa"].values.astype(float)

        # ── Just-identified: SOC_norm (primary) ─────────────────────────
        X_np, Z_np, dis_col, col_names = build_XZ(sub, var_names, mode="just")

        fs_res_full = OLS(X_np[:, dis_col], Z_np).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
        )
        v_hat  = fs_res_full.resid
        iv_res = fit_2sls_hac(y, X_np, Z_np, dis_col)
        iv_results[r] = iv_res
        iv_results[r]["col_names"] = col_names

        # First-stage diagnostics — SOC_norm partial F
        fs_t = iv_res["fs_F"] ** 0.5
        fs_p = 2 * _stats.t.sf(abs(fs_t), df=len(y) - len(var_names))

        # Z_newcap partial F (in the over-id first stage that includes SOC_norm)
        _, Z_nc, _, _ = build_XZ(sub, var_names, mode="newcap")
        fs_nc_res = OLS(X_np[:, dis_col], Z_nc).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
        )
        # Z_newcap is the last column of Z_nc
        nc_t = fs_nc_res.tvalues[-1]
        nc_F = nc_t ** 2
        nc_nonzero = (sub["Z_newcap"] != 0.0).sum()   # obs with active commissioning

        fs_rows.append({
            "regime":          r,
            "partial_F_norm":  iv_res["fs_F"],
            "coef_SOC_norm":   iv_res["fs_coef"],
            "se_SOC_norm":     iv_res["fs_se"],
            "stars_norm":      stars(fs_p),
            "partial_F_newcap": nc_F,
            "newcap_nonzero_obs": int(nc_nonzero),
            "n_obs":           len(y),
        })

        # Wu-Hausman (uses just-identified first-stage residuals)
        wh_coef, wh_se, wh_t, wh_p = wu_hausman_test(y, X_np, v_hat)
        wh_rows.append({
            "regime":     r,
            "wh_coef":    wh_coef,
            "wh_se":      wh_se,
            "wh_t":       wh_t,
            "wh_p":       wh_p,
            "stars":      stars(wh_p),
            "endogenous": "YES" if wh_p < 0.10 else "no",
        })

        # ── Over-id 1: SOC_norm + SOC_raw → Sargan J-test ────────────────
        _, Z_oid_raw, _, _ = build_XZ(sub, var_names, mode="raw")
        iv_oid_raw         = fit_2sls_hac(y, X_np, Z_oid_raw, dis_col)
        J, pJ, dfJ         = sargan_j_test(iv_oid_raw["resid"], Z_oid_raw)
        sargan_rows.append({
            "regime": r, "model": "SOC_norm+SOC_raw",
            "J_stat": J, "p_value": pJ, "df": dfJ,
            "stars":  stars(pJ),
            "valid":  "instruments consistent" if pJ > 0.10 else "INCONSISTENT ⚠",
        })

        # ── Over-id 2: SOC_norm + Z_newcap → NEW Sargan J-test ───────────
        iv_oid_nc      = fit_2sls_hac(y, X_np, Z_nc, dis_col)
        J_nc, pJ_nc, dfJ_nc = sargan_j_test(iv_oid_nc["resid"], Z_nc)
        sargan_rows.append({
            "regime": r, "model": "SOC_norm+Z_newcap",
            "J_stat": J_nc, "p_value": pJ_nc, "df": dfJ_nc,
            "stars":  stars(pJ_nc),
            "valid":  "instruments consistent" if pJ_nc > 0.10
                      else ("no variation" if nc_nonzero == 0 else "INCONSISTENT ⚠"),
        })

    # ── 4. Print first-stage results ──────────────────────────────────────
    print()
    print(f"  {'Regime':<10s}  {'partial-F (SOC_norm)':>22s}  {'partial-F (Z_newcap)':>22s}  "
          f"{'Z_newcap obs':>13s}  {'N':>7s}")
    print("  " + "-" * 88)
    for row in fs_rows:
        flag_norm   = "OK ✓" if row["partial_F_norm"]   > 10 else "WEAK ⚠"
        flag_newcap = "OK ✓" if row["partial_F_newcap"] > 10 else "WEAK ⚠"
        print(f"  R{row['regime']:<9d}  "
              f"{row['partial_F_norm']:>18.2f} {flag_norm:<5s}  "
              f"{row['partial_F_newcap']:>18.2f} {flag_newcap:<5s}  "
              f"{row['newcap_nonzero_obs']:>13d}  {row['n_obs']:>7d}")

    # ── 5. Print Wu-Hausman ───────────────────────────────────────────────
    print()
    print("  Wu-Hausman endogeneity test (H0: battery_dis is exogenous):")
    print(f"  {'Regime':<10s}  {'WH coef':>10s}  {'t-stat':>8s}  {'p-value':>9s}  {'Endogenous?':>12s}")
    print("  " + "-" * 58)
    for row in wh_rows:
        print(f"  R{row['regime']:<9d}  {row['wh_coef']:>+10.4f}  "
              f"{row['wh_t']:>8.3f}  {row['wh_p']:>9.4f}  "
              f"{row['endogenous']:>12s}  {row['stars']}")

    # ── 5b. Sargan J-test ─────────────────────────────────────────────────
    print()
    for mdl in ["SOC_norm+SOC_raw", "SOC_norm+Z_newcap"]:
        print(f"  Sargan J-test (H0: instruments valid) — model: {mdl}")
        print(f"  {'Regime':<10s}  {'J-stat':>10s}  {'p-value':>9s}  {'df':>4s}  Result")
        print("  " + "-" * 60)
        for row in [r for r in sargan_rows if r["model"] == mdl]:
            print(f"  R{row['regime']:<9d}  {row['J_stat']:>10.4f}  {row['p_value']:>9.4f}  "
                  f"{row['df']:>4d}  {row['valid']}  {row['stars']}")
        print()

    # ── 6. Print 2SLS coefficients ────────────────────────────────────────
    print()
    print(DIVIDER)
    print("  2SLS COEFFICIENTS — SA EQUATION (Δsa)")
    print(DIVIDER)
    print(f"\n  {'Variable':<30s}", end="")
    for r in [1, 2, 3]:
        n = (sample["regime"] == r).sum()
        print(f"  {'Regime '+str(r)+' (N='+str(int(n))+')':>22s}", end="")
    print()
    print("  " + "─" * 98)

    highlight = {"ECT_lag", "battery_discharging_mw", "battery_charging_mw"}
    for i, vn in enumerate(var_names):
        mark = " ◄" if vn in highlight else ""
        print(f"  {vn+mark:<32s}", end="")
        for r in [1, 2, 3]:
            iv = iv_results[r]
            c  = iv["params"][i]
            se = iv["se"][i]
            p  = iv["pvals"][i]
            print(f"  {c:>+9.4f}{stars(p):<3s}({se:.4f})", end="")
        print()

    # ── 7. Comparison table ───────────────────────────────────────────────
    print()
    print(DIVIDER)
    print("  BATTERY DISCHARGE COEFFICIENT COMPARISON  (SA equation, β_dis)")
    print(DIVIDER)
    print(f"\n  Source                {'Regime 1':>14s}  {'Regime 2':>14s}  {'Regime 3':>14s}  Method")
    print("  " + "-" * 80)

    p2 = pd.read_csv(RESULTS_DIR / "phase2_dsa_coefs.csv")
    p3 = pd.read_csv(RESULTS_DIR / "phase3_lagged_dsa_coefs.csv")

    dis_idx = var_names.index("battery_discharging_mw")

    def _safe_stars(df, r, vname):
        row = df[(df["regime"] == r) & (df["variable"] == vname)]
        s   = str(row["stars"].iloc[0])
        return "" if s in ("nan", "None", "") else s

    def p2_val(r):
        row = p2[(p2["regime"] == r) & (p2["variable"] == "battery_discharging_mw")]
        return float(row["coef"].iloc[0]), _safe_stars(p2, r, "battery_discharging_mw")
    def p3_val(r):
        row = p3[(p3["regime"] == r) & (p3["variable"] == "battery_discharging_mw")]
        return float(row["coef"].iloc[0]), _safe_stars(p3, r, "battery_discharging_mw")
    def p4_val(r):
        iv = iv_results[r]
        c  = iv["params"][dis_idx]
        p  = iv["pvals"][dis_idx]
        return c, stars(p)

    for label, fn, note in [
        ("Phase 2 (OLS, contemp)",  p2_val, "simultaneity bias"),
        ("Phase 3 (OLS, lagged)",   p3_val, "predetermined regressor"),
        ("Phase 4 (2SLS, SOC IV)",  p4_val, "IV-identified causal effect"),
    ]:
        vals = [fn(r) for r in [1, 2, 3]]
        line = f"  {label:<22s}"
        for c, s in vals:
            line += f"  {c:>+9.4f}{s:<3s}   "
        line += f"  {note}"
        print(line)

    print()
    for r in [1, 2, 3]:
        iv  = iv_results[r]
        c   = iv["params"][dis_idx]
        p   = iv["pvals"][dis_idx]
        s   = stars(p)
        fst = iv["fs_F"]
        print(f"  Regime {r}: β_dis(IV) = {c:+.4f}{s}   first-stage F = {fst:.1f}")

    # ── 8. Save results ───────────────────────────────────────────────────
    print(f"\n[8] Saving results ...")

    # First-stage CSV
    pd.DataFrame(fs_rows).to_csv(RESULTS_DIR / "phase4_firststage.csv", index=False)
    print(f"  Saved: results/phase4_firststage.csv")

    # Wu-Hausman CSV
    pd.DataFrame(wh_rows).to_csv(RESULTS_DIR / "phase4_wu_hausman.csv", index=False)
    print(f"  Saved: results/phase4_wu_hausman.csv")

    # Sargan J-test CSV
    pd.DataFrame(sargan_rows).to_csv(RESULTS_DIR / "phase4_sargan.csv", index=False)
    print(f"  Saved: results/phase4_sargan.csv")

    # IV coefficients CSV
    iv_rows = []
    for r in [1, 2, 3]:
        iv = iv_results[r]
        for i, vn in enumerate(var_names):
            iv_rows.append({
                "equation": "dsa",
                "regime":   r,
                "variable": vn,
                "coef":     iv["params"][i],
                "hac_se":   iv["se"][i],
                "t_stat":   iv["tvals"][i],
                "p_value":  iv["pvals"][i],
                "stars":    stars(iv["pvals"][i]),
            })
    pd.DataFrame(iv_rows).to_csv(RESULTS_DIR / "phase4_iv_coefs.csv", index=False)
    print(f"  Saved: results/phase4_iv_coefs.csv")

    # Comparison CSV
    comp_rows = []
    for r in [1, 2, 3]:
        for vname in ["battery_discharging_mw", "battery_charging_mw"]:
            idx = var_names.index(vname)
            p2r = p2[(p2["regime"] == r) & (p2["variable"] == vname)]
            p3r = p3[(p3["regime"] == r) & (p3["variable"] == vname)]
            iv  = iv_results[r]
            comp_rows.append({
                "regime":       r,
                "variable":     vname,
                "phase2_coef":  float(p2r["coef"].iloc[0]),
                "phase2_stars": _safe_stars(p2, r, vname),
                "phase3_coef":  float(p3r["coef"].iloc[0]),
                "phase3_stars": _safe_stars(p3, r, vname),
                "phase4_coef":  iv["params"][idx],
                "phase4_stars": stars(iv["pvals"][idx]),
                "phase4_fs_F":  iv["fs_F"],
            })
    pd.DataFrame(comp_rows).to_csv(RESULTS_DIR / "phase4_coef_comparison.csv", index=False)
    print(f"  Saved: results/phase4_coef_comparison.csv")

    # Summary text
    dis_idx   = var_names.index("battery_discharging_mw")
    ect_idx   = var_names.index("ECT_lag")
    lines     = []
    n_fac = len(BATTERY_CAPACITY)
    lines.append("PHASE 4 — IV/2SLS BATTERY EFFECT SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Primary instrument: SOC_norm = capacity-normalised 12h net charging (fraction), lagged 1")
    lines.append(f"Secondary (robustness): SOC_raw (MWh, lagged 1) — over-identified Sargan J-test")
    lines.append(f"New IV: Z_newcap = MWh commissioned in preceding {NEWCAP_WINDOW_DAYS} days, lagged 1")
    lines.append(f"Capacity: {sample['cap_mwh'].min():.0f} → {sample['cap_mwh'].max():.0f} MWh "
                 f"({sample['cap_mwh'].max()/sample['cap_mwh'].min():.1f}× growth, {n_fac} facilities)")
    lines.append(f"Relevance  corr(dis, SOC_norm): {r_rel:+.4f}")
    lines.append(f"Exclusion  corr(Δsa, SOC_norm): {r_exc:+.4f}")
    lines.append("")
    lines.append("First-stage partial F (endogenous var: battery_discharging_mw):")
    for row in fs_rows:
        flag_n  = "(strong)" if row["partial_F_norm"]   > 10 else "(WEAK)"
        flag_nc = "(strong)" if row["partial_F_newcap"] > 10 else "(WEAK)"
        lines.append(f"  Regime {row['regime']}: F(SOC_norm) = {row['partial_F_norm']:.2f} {flag_n}"
                     f"  |  F(Z_newcap) = {row['partial_F_newcap']:.2f} {flag_nc}"
                     f"  [nonzero obs = {row['newcap_nonzero_obs']}]")
    lines.append("")
    lines.append("Sargan J-test — model: SOC_norm + SOC_raw (H0: both instruments valid):")
    for row in [r for r in sargan_rows if r["model"] == "SOC_norm+SOC_raw"]:
        lines.append(f"  Regime {row['regime']}: J = {row['J_stat']:.4f}  p = {row['p_value']:.4f}  {row['stars']}  → {row['valid']}")
    lines.append("")
    lines.append(f"Sargan J-test — model: SOC_norm + Z_newcap (H0: both instruments valid):")
    lines.append(f"  (Z_newcap = MWh commissioned in preceding {NEWCAP_WINDOW_DAYS} days, lagged 1)")
    for row in [r for r in sargan_rows if r["model"] == "SOC_norm+Z_newcap"]:
        nc_obs = next(f["newcap_nonzero_obs"] for f in fs_rows if f["regime"] == row["regime"])
        lines.append(f"  Regime {row['regime']}: J = {row['J_stat']:.4f}  p = {row['p_value']:.4f}  {row['stars']}"
                     f"  Z_newcap nonzero obs = {nc_obs}  → {row['valid']}")
    lines.append("")
    lines.append("Wu-Hausman endogeneity test:")
    for row in wh_rows:
        lines.append(f"  Regime {row['regime']}: t = {row['wh_t']:.3f}  p = {row['wh_p']:.4f}  {row['stars']}  → {row['endogenous']}")
    lines.append("")
    lines.append("2SLS battery discharge coefficient (β_dis), SA equation:")
    for r in [1, 2, 3]:
        iv = iv_results[r]
        c  = iv["params"][dis_idx]
        p  = iv["pvals"][dis_idx]
        se = iv["se"][dis_idx]
        lines.append(f"  Regime {r}: β_dis = {c:+.4f}  se = {se:.4f}  {stars(p)}")
    lines.append("")
    lines.append("2SLS adjustment speed (α = ECT_lag), SA equation:")
    for r in [1, 2, 3]:
        iv = iv_results[r]
        c  = iv["params"][ect_idx]
        p  = iv["pvals"][ect_idx]
        lines.append(f"  Regime {r}: α = {c:+.4f}  {stars(p)}")
    lines.append("")
    lines.append("Coefficient comparison: Phase 2 (OLS) → Phase 3 (lagged) → Phase 4 (IV)")
    for r in [1, 2, 3]:
        c2, s2 = p2_val(r)
        c3, s3 = p3_val(r)
        c4, s4 = p4_val(r)
        lines.append(f"  Regime {r}: {c2:+.4f}{s2}  →  {c3:+.4f}{s3}  →  {c4:+.4f}{s4}")

    (RESULTS_DIR / "phase4_summary.txt").write_text("\n".join(lines))
    print(f"  Saved: results/phase4_summary.txt")

    # ── 9. Final interpretation ───────────────────────────────────────────
    print()
    print(DIVIDER)
    print("  PHASE 4 COMPLETE — INTERPRETATION")
    print(DIVIDER)
    iv_r3   = iv_results[3]
    b_iv_r3 = iv_r3["params"][dis_idx]
    p_iv_r3 = iv_r3["pvals"][dis_idx]
    fst_r3  = iv_r3["fs_F"]
    wh_r3   = wh_rows[2]

    print(f"""
  Regime 3 (SA price premium — key regime for dampening narrative):
    β_dis (Phase 2 OLS, contemp): {p2_val(3)[0]:+.4f}***  — upward biased by simultaneity
    β_dis (Phase 3 OLS, lagged):  {p3_val(3)[0]:+.4f}     — predetermined, sign corrected
    β_dis (Phase 4 2SLS, IV):     {b_iv_r3:+.4f}{stars(p_iv_r3):<3s}  — IV-identified causal effect
    First-stage F:                {fst_r3:.2f}          — {"strong instrument" if fst_r3 > 10 else "weak instrument ⚠"}
    Wu-Hausman (endogeneity):      p = {wh_r3['wh_p']:.4f}  {wh_r3['stars']}  — {"confirms endogeneity; IV preferred" if wh_r3['wh_p'] < 0.10 else "endogeneity not confirmed at 10%"}

  Instrument validity:
    SOC_norm: capacity normalisation removes the growing-fleet trend ({sample['cap_mwh'].min():.0f} → {sample['cap_mwh'].max():.0f} MWh, {sample['cap_mwh'].max()/sample['cap_mwh'].min():.1f}×).
    Relevance: high SOC fraction → more discharge capacity (corr = {r_rel:+.4f})
    Exclusion: SOC fraction does not set market-clearing price (corr Δsa = {r_exc:+.4f})
    Sargan J-test (SOC_norm+SOC_raw): tests whether SOC_raw adds consistent variation.
    Sargan J-test (SOC_norm+Z_newcap): NEW — tests commissioning-event IV validity.
""")


if __name__ == "__main__":
    main()
