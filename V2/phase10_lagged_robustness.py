#!/usr/bin/env python3
"""
Phase 10: Lagged Battery Robustness — Improving R3 Significance
================================================================
Three approaches to sharpen the Phase 3 causal battery discharge estimate
in Regime 3 (β_dis = −0.194, SE = 0.341, p = 0.568):

  Approach 1: Lagged battery × capacity interaction (Phase 3 + Phase 9)
  Approach 2: Lagged battery GARCH-M (Phase 3 + Phase 7)
  Approach 3: Rolling-threshold lagged OLS (Phase 3 + Phase 6)

Output files (7):
  results/phase10_approach1_interact_coefs.csv
  results/phase10_approach1_margins.csv
  results/phase10_approach2_garchm_coefs.csv
  results/phase10_approach2_garchm_params.csv
  results/phase10_approach2_diagnostics.csv
  results/phase10_approach3_rolling_coefs.csv
  results/phase10_summary.txt
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
from statsmodels.tools import add_constant
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model as _arch_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GAMMA_L  = -45.8039
GAMMA_U  =  20.0215
BETA_CI  =  1.2103      # Johansen cointegrating vector (fixed)
ECM_LAGS =  4
HAC_LAGS = 12
MAX_ITER = 10
CONV_TOL = 1e-5

NQ        = 17           # grid search quantile points
MIN_R_PCT = 0.05         # minimum extreme-regime fraction

CONTROLS = [
    "wind_mw",
    "solar_total_mw",
    "demand_mw",
    "battery_discharging_mw",
    "battery_charging_mw",
    "interconnector_mw",
]
BATTERY_COLS   = {"battery_discharging_mw", "battery_charging_mw"}
CONTROLS_NO_BAT = ["wind_mw", "solar_total_mw", "demand_mw", "interconnector_mw"]

# SA BESS fleet
BATTERY_CAPACITY = [
    ("Hornsdale Power Reserve",        "2017-10-01",  129.0),
    ("Dalrymple North",                "2018-12-01",    8.0),
    ("Lake Bonney",                    "2019-12-01",   52.0),
    ("Hornsdale Power Reserve Ext.",   "2020-09-01",   64.5),
    ("Adelaide Desalination Plant",    "2021-05-01",   13.0),
    ("Christies Beach",                "2022-03-01",    4.3),
    ("Torrens Island",                 "2023-08-01",  250.0),
    ("Tailem Bend 2",                  "2025-01-01",   84.0),
    ("Blyth",                          "2025-04-01",  477.0),
    ("Mannum",                         "2025-10-01",  200.0),
]

EVAL_DATES = [
    ("2023-01-01", "Jan 2023"),
    ("2024-01-01", "Jan 2024"),
    ("2025-06-01", "Jun 2025"),
]

DIVIDER = "=" * 74


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime").sort_index()
    df = df[df["sa_price"].abs() < 15001]
    df = df[df["vic_price"].abs() < 15001]
    return df


def estimate_ect(df):
    """Johansen cointegration → normalised vector and ECT series."""
    prices = df[["sa_price", "vic_price"]].dropna()
    res  = coint_johansen(prices.values, det_order=0, k_ar_diff=ECM_LAGS)
    beta = res.evec[:, 0] / res.evec[0, 0]
    ect  = prices.values @ beta
    ect  = pd.Series(ect, index=prices.index, name="ECT")
    return beta, ect


def build_sample_lagged(df, ect):
    """Phase 3 ECM sample: battery vars shifted 1 period (predetermined)."""
    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")

    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}
    for lag in range(1, ECM_LAGS + 1):
        rows[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows[f"dvic_l{lag}"] = dvic.shift(lag)

    for c in CONTROLS:
        if c in df.columns:
            rows[c] = df[c].shift(1) if c in BATTERY_COLS else df[c]
        else:
            rows[c] = np.nan

    sample = pd.DataFrame(rows).dropna()
    sample["regime"] = np.where(
        sample["ECT_lag"] < GAMMA_L, 1,
        np.where(sample["ECT_lag"] > GAMMA_U, 3, 2)
    )
    return sample


def regressor_names():
    lag_names = (
        [f"dsa_l{i}"  for i in range(1, ECM_LAGS + 1)] +
        [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)]
    )
    return ["const", "ECT_lag"] + lag_names + CONTROLS


def build_X(sample_r):
    cols = regressor_names()
    X = sample_r[cols[1:]].copy()
    X.insert(0, "const", 1.0)
    return X.values.astype(float)


def fit_ols_hac(y, X):
    return OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )


def stars(p):
    if not np.isfinite(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def installed_capacity_at(date):
    return sum(
        mwh for _, comm, mwh in BATTERY_CAPACITY
        if pd.Timestamp(comm) <= pd.Timestamp(date)
    )


def add_cap_norm(df):
    cap_series = pd.Series(
        {idx: installed_capacity_at(idx) for idx in df.index},
        name="installed_mwh"
    )
    df = df.copy()
    df["installed_mwh"] = cap_series
    mean_cap = df["installed_mwh"].mean()
    df["cap_norm"] = df["installed_mwh"] / mean_cap
    return df, mean_cap


# GARCH helpers (from Phase 7)
def arch_lm_test(resid, nlags=12):
    e2 = resid ** 2
    n  = len(e2)
    if n <= nlags + 5:
        return np.nan, np.nan
    try:
        X_lm = np.column_stack(
            [np.ones(n - nlags)] +
            [e2[nlags - i : n - i] for i in range(1, nlags + 1)]
        )
        y_lm = e2[nlags:]
        res  = OLS(y_lm, X_lm).fit()
        lm   = float((n - nlags) * res.rsquared)
        pval = float(1.0 - sp_stats.chi2.cdf(lm, nlags))
        return lm, pval
    except Exception:
        return np.nan, np.nan


def lb_stat_p(series, lag=12):
    try:
        lb = acorr_ljungbox(series, lags=[lag], return_df=True)
        return float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan, np.nan


# Grid search (from Phase 6)
def grid_search_thresholds(sample):
    ect_lag = sample["ECT_lag"].values
    n       = len(ect_lag)
    min_n   = max(int(MIN_R_PCT * n), 10)
    min_n2  = int(0.20 * n)

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
    best_sizes = (0, 0, 0)

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


# ============================================================================
# Approach 1: Lagged Battery × Capacity Interaction
# ============================================================================
def run_approach1(sample):
    print("\n" + DIVIDER)
    print("  APPROACH 1: LAGGED BATTERY × CAPACITY INTERACTION")
    print(DIVIDER)

    # Add capacity normalisation
    sample_cap, mean_cap = add_cap_norm(sample)
    print(f"  mean installed capacity: {mean_cap:.1f} MWh")

    # Create interaction terms (battery cols are already lagged in sample)
    sample_cap["bat_dis_cap"] = sample_cap["battery_discharging_mw"] * sample_cap["cap_norm"]
    sample_cap["bat_chg_cap"] = sample_cap["battery_charging_mw"]    * sample_cap["cap_norm"]

    # Feature columns: ECM lags + lagged battery + interactions + other controls
    feat_cols = (
        ["ECT_lag"]
        + [f"dsa_l{i}" for i in range(1, ECM_LAGS + 1)]
        + [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)]
        + ["battery_discharging_mw", "battery_charging_mw"]
        + ["bat_dis_cap", "bat_chg_cap"]
        + CONTROLS_NO_BAT
    )

    coef_rows   = []
    margin_rows = []
    summary_lines = []

    for r in [1, 2, 3]:
        sub   = sample_cap[sample_cap["regime"] == r].copy()
        n_obs = len(sub)
        y     = sub["dsa"].values

        X = add_constant(sub[feat_cols])
        fn = list(X.columns)
        res = OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
        )

        beta_dis  = res.params["battery_discharging_mw"]
        gamma_dis = res.params["bat_dis_cap"]
        se_gd     = res.bse["bat_dis_cap"]
        p_gd      = res.pvalues["bat_dis_cap"]

        beta_chg  = res.params["battery_charging_mw"]
        gamma_chg = res.params["bat_chg_cap"]
        se_gc     = res.bse["bat_chg_cap"]
        p_gc      = res.pvalues["bat_chg_cap"]

        print(f"\n  Regime {r} (N={n_obs:,}):")
        print(f"    β_dis = {beta_dis:+.4f}   γ_dis = {gamma_dis:+.4f}  SE={se_gd:.4f}  p={p_gd:.4f}{stars(p_gd)}")
        print(f"    β_chg = {beta_chg:+.4f}   γ_chg = {gamma_chg:+.4f}  SE={se_gc:.4f}  p={p_gc:.4f}{stars(p_gc)}")
        print(f"    R² = {res.rsquared:.4f}")

        # Save all coefficients
        for nm in fn:
            coef_rows.append({
                "regime":   r,
                "n_obs":    n_obs,
                "variable": nm,
                "coef":     res.params[nm],
                "hac_se":   res.bse[nm],
                "t_stat":   res.tvalues[nm],
                "p_value":  res.pvalues[nm],
                "stars":    stars(res.pvalues[nm]),
            })

        # Delta-method margins at evaluation dates
        cov_mat = res.cov_params()
        var_bd  = cov_mat.loc["battery_discharging_mw", "battery_discharging_mw"]
        var_gd  = cov_mat.loc["bat_dis_cap", "bat_dis_cap"]
        cov_bdg = cov_mat.loc["battery_discharging_mw", "bat_dis_cap"]

        print(f"    Marginal effect at evaluation dates:")
        for date_str, label in EVAL_DATES:
            cap_t      = installed_capacity_at(pd.Timestamp(date_str))
            cap_norm_t = cap_t / mean_cap
            beta_total = beta_dis + gamma_dis * cap_norm_t
            var_total  = var_bd + cap_norm_t**2 * var_gd + 2 * cap_norm_t * cov_bdg
            se_total   = np.sqrt(max(var_total, 0))
            t_total    = beta_total / se_total if se_total > 0 else np.nan
            p_total    = 2 * sp_stats.t.sf(abs(t_total), df=n_obs - len(fn))

            print(f"      {label}: cap_norm={cap_norm_t:.3f}  β_total={beta_total:+.4f}  "
                  f"SE={se_total:.4f}  p={p_total:.4f}{stars(p_total)}")

            margin_rows.append({
                "regime":     r,
                "date":       date_str,
                "label":      label,
                "cap_mwh":    cap_t,
                "cap_norm":   cap_norm_t,
                "beta_dis":   beta_dis,
                "gamma_dis":  gamma_dis,
                "beta_total": beta_total,
                "se_total":   se_total,
                "t_total":    t_total,
                "p_total":    p_total,
                "stars":      stars(p_total),
            })

        # Summary line for R3
        if r == 3:
            # Find Jun 2025 margin
            m_2025 = [m for m in margin_rows if m["regime"] == 3 and "Jun" in m["label"]]
            if m_2025:
                m = m_2025[-1]
                summary_lines.append(
                    f"A1 margins (2025)  {m['beta_total']:+.4f}      "
                    f"{m['se_total']:.4f}   {m['p_total']:.4f}     "
                    f"{'Yes' if m['p_total'] < 0.10 else 'No'}"
                )

    # Save CSVs
    df_coef = pd.DataFrame(coef_rows)
    df_coef.to_csv(RESULTS_DIR / "phase10_approach1_interact_coefs.csv", index=False)
    print(f"\n  Saved: results/phase10_approach1_interact_coefs.csv")

    df_margins = pd.DataFrame(margin_rows)
    df_margins.to_csv(RESULTS_DIR / "phase10_approach1_margins.csv", index=False)
    print(f"  Saved: results/phase10_approach1_margins.csv")

    return summary_lines


# ============================================================================
# Approach 2: Lagged Battery GARCH-M
# ============================================================================
def run_approach2(sample, baseline_results):
    print("\n" + DIVIDER)
    print("  APPROACH 2: LAGGED BATTERY GARCH-M")
    print(DIVIDER)

    vn_base = regressor_names()

    garch_rows = []
    coef_rows  = []
    diag_rows  = []
    summary_lines = []

    for r in [1, 2, 3]:
        sub   = sample[sample["regime"] == r].copy()
        n     = len(sub)
        y     = sub["dsa"].values.astype(float)

        X_base = sub[vn_base[1:]].copy()
        X_base.insert(0, "const", 1.0)
        X_np   = X_base.values.astype(float)

        # Baseline OLS (same as Phase 3)
        res_ols = fit_ols_hac(y, X_np)
        resid   = res_ols.resid.copy()
        baseline_se_dis = res_ols.bse[vn_base.index("battery_discharging_mw")]

        # Pre-GARCH diagnostics
        arch_pre_stat, arch_pre_p = arch_lm_test(resid)
        lb_raw_pre_stat, lb_raw_pre_p = lb_stat_p(resid)

        # Iterated GARCH-M
        sigma_t   = np.full(n, float(np.std(resid)))
        delta_old = 0.0
        garch_res = None
        res_gm    = None
        converged = False
        garch_ok  = True
        n_iter    = 0

        for it in range(1, MAX_ITER + 1):
            n_iter = it
            try:
                am = _arch_model(resid, mean="Zero", vol="GARCH",
                                 p=1, q=1, dist="StudentsT", rescale=True)
                gr = am.fit(disp="off", show_warning=False,
                            options={"maxiter": 500, "ftol": 1e-10})
                sig_new = gr.conditional_volatility
                if np.any(np.isnan(sig_new)) or np.any(sig_new <= 0):
                    garch_ok = False
                    break
                sigma_t   = sig_new
                garch_res = gr
            except Exception:
                garch_ok = False
                break

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
            delta_new = float(res_gm.params[-1])

            if abs(delta_new - delta_old) < CONV_TOL:
                converged = True
                break
            delta_old = delta_new

        # Post diagnostics
        if garch_ok and garch_res is not None:
            arch_post_stat, arch_post_p = arch_lm_test(resid)
            lb_raw_post_stat, lb_raw_post_p = lb_stat_p(resid)
            std_r = resid / np.where(sigma_t > 0, sigma_t, 1e-9)
            lb_std_stat, lb_std_p = lb_stat_p(std_r)
            lb_sq_stat,  lb_sq_p  = lb_stat_p(std_r ** 2)
        else:
            arch_post_stat = arch_post_p = np.nan
            lb_raw_post_stat = lb_raw_post_p = np.nan
            lb_std_stat = lb_std_p = np.nan
            lb_sq_stat  = lb_sq_p  = np.nan

        # Extract GARCH params
        if garch_res is not None:
            gp  = garch_res.params
            gse = garch_res.std_err
            gpv = garch_res.pvalues
            def gget(key):
                return (float(gp.get(key, np.nan)),
                        float(gse.get(key, np.nan)),
                        float(gpv.get(key, np.nan)))
            omega_v,   omega_se,   omega_p   = gget("omega")
            alpha_g_v, alpha_g_se, alpha_g_p = gget("alpha[1]")
            beta_g_v,  beta_g_se,  beta_g_p  = gget("beta[1]")
            nu_v,      nu_se,      nu_p      = gget("nu")
            persistence = alpha_g_v + beta_g_v
        else:
            omega_v = alpha_g_v = beta_g_v = nu_v = np.nan
            omega_se = alpha_g_se = beta_g_se = nu_se = np.nan
            omega_p = alpha_g_p = beta_g_p = nu_p = np.nan
            persistence = np.nan

        # Mean equation results
        res_use = res_gm if res_gm is not None else res_ols
        vn_use  = vn_base + ["sigma_t"] if res_gm is not None else vn_base
        has_gm  = res_gm is not None

        dis_idx = vn_use.index("battery_discharging_mw")
        chg_idx = vn_use.index("battery_charging_mw")
        ect_idx = vn_use.index("ECT_lag")

        beta_dis    = float(res_use.params[dis_idx])
        beta_dis_se = float(res_use.bse[dis_idx])
        beta_dis_p  = float(res_use.pvalues[dis_idx])
        beta_chg    = float(res_use.params[chg_idx])
        beta_chg_se = float(res_use.bse[chg_idx])
        beta_chg_p  = float(res_use.pvalues[chg_idx])
        alpha_val   = float(res_use.params[ect_idx])
        alpha_se    = float(res_use.bse[ect_idx])
        alpha_p     = float(res_use.pvalues[ect_idx])
        delta_val   = float(res_use.params[-1]) if has_gm else np.nan
        delta_se    = float(res_use.bse[-1])    if has_gm else np.nan
        delta_p     = float(res_use.pvalues[-1]) if has_gm else np.nan

        # SE ratio
        se_ratio = baseline_se_dis / beta_dis_se if beta_dis_se > 0 else np.nan

        status = "converged" if converged else ("FAILED" if not garch_ok else "max iter")
        print(f"\n  Regime {r} (N={n:,}): {status} ({n_iter} iter)")
        print(f"    β_dis = {beta_dis:+.4f}  SE = {beta_dis_se:.4f}  p = {beta_dis_p:.4f}{stars(beta_dis_p)}")
        print(f"    SE ratio (plain/GARCH-M): {se_ratio:.3f}" if np.isfinite(se_ratio) else "")
        print(f"    δ (risk premium) = {delta_val:+.4f}  SE = {delta_se:.4f}  p = {delta_p:.4f}{stars(delta_p)}" if has_gm else "")
        if garch_res is not None:
            print(f"    GARCH: α_G={alpha_g_v:.4f}  β_G={beta_g_v:.4f}  persist={persistence:.4f}  ν={nu_v:.2f}")

        # Save rows
        coef_rows.append({
            "regime": r, "n": n,
            "alpha": alpha_val, "alpha_se": alpha_se, "alpha_p": alpha_p,
            "beta_dis": beta_dis, "beta_dis_se": beta_dis_se, "beta_dis_p": beta_dis_p,
            "beta_chg": beta_chg, "beta_chg_se": beta_chg_se, "beta_chg_p": beta_chg_p,
            "delta": delta_val, "delta_se": delta_se, "delta_p": delta_p,
            "se_ratio_plain_gm": se_ratio,
            "rsq": float(res_use.rsquared),
            "converged": converged, "n_iter": n_iter, "garch_ok": garch_ok,
        })

        garch_rows.append({
            "regime": r, "n": n,
            "omega": omega_v, "omega_se": omega_se, "omega_p": omega_p,
            "alpha_g": alpha_g_v, "alpha_g_se": alpha_g_se, "alpha_g_p": alpha_g_p,
            "beta_g": beta_g_v, "beta_g_se": beta_g_se, "beta_g_p": beta_g_p,
            "persistence": persistence,
            "nu": nu_v, "nu_se": nu_se,
            "garch_ok": garch_ok, "converged": converged, "n_iter": n_iter,
        })

        diag_rows.append({
            "regime": r,
            "arch_pre": arch_pre_stat, "arch_pre_p": arch_pre_p,
            "arch_post": arch_post_stat, "arch_post_p": arch_post_p,
            "lb_raw_pre": lb_raw_pre_stat, "lb_raw_pre_p": lb_raw_pre_p,
            "lb_raw_post": lb_raw_post_stat, "lb_raw_post_p": lb_raw_post_p,
            "lb_std": lb_std_stat, "lb_std_p": lb_std_p,
            "lb_sq": lb_sq_stat, "lb_sq_p": lb_sq_p,
        })

        if r == 3:
            summary_lines.append(
                f"A2 GARCH-M         {beta_dis:+.4f}      "
                f"{beta_dis_se:.4f}   {beta_dis_p:.4f}     "
                f"{'Yes' if beta_dis_p < 0.10 else 'No'}"
            )

    # Save
    pd.DataFrame(coef_rows).to_csv(RESULTS_DIR / "phase10_approach2_garchm_coefs.csv", index=False)
    pd.DataFrame(garch_rows).to_csv(RESULTS_DIR / "phase10_approach2_garchm_params.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(RESULTS_DIR / "phase10_approach2_diagnostics.csv", index=False)
    print(f"\n  Saved: results/phase10_approach2_garchm_coefs.csv")
    print(f"  Saved: results/phase10_approach2_garchm_params.csv")
    print(f"  Saved: results/phase10_approach2_diagnostics.csv")

    return summary_lines


# ============================================================================
# Approach 3: Rolling-Threshold Lagged OLS
# ============================================================================
def run_approach3(df):
    print("\n" + DIVIDER)
    print("  APPROACH 3: ROLLING-THRESHOLD LAGGED OLS")
    print(DIVIDER)

    vn = regressor_names()
    coef_rows = []
    summary_lines = []

    # ── Per-window estimation ──────────────────────────────────────────────
    windows = [
        {"label": "2023", "start": "2023-01-01", "end": "2023-12-31 23:59"},
        {"label": "2024", "start": "2024-01-01", "end": "2024-12-31 23:59"},
        {"label": "2025", "start": "2025-01-01", "end": "2025-12-31 23:59"},
    ]

    window_assignments = {}  # index → regime, for pooled model

    for winfo in windows:
        start = pd.Timestamp(winfo["start"])
        end   = pd.Timestamp(winfo["end"])
        df_w  = df[(df.index >= start) & (df.index <= end)].copy()

        if len(df_w) < 200:
            print(f"  {winfo['label']}: too few obs ({len(df_w)}), skipping")
            continue

        # Build ECM sample within window using fixed β for ECT
        sa   = df_w["sa_price"]
        vic  = df_w["vic_price"]
        dsa  = sa.diff().rename("dsa")
        dvic = vic.diff().rename("dvic")
        ect  = (sa - BETA_CI * vic).rename("ECT")

        rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}
        for lag in range(1, ECM_LAGS + 1):
            rows[f"dsa_l{lag}"]  = dsa.shift(lag)
            rows[f"dvic_l{lag}"] = dvic.shift(lag)
        for c in CONTROLS:
            if c in df_w.columns:
                rows[c] = df_w[c].shift(1) if c in BATTERY_COLS else df_w[c]
            else:
                rows[c] = np.nan

        w_sample = pd.DataFrame(rows).dropna()

        if len(w_sample) < 200:
            print(f"  {winfo['label']}: too few obs after lag creation ({len(w_sample)}), skipping")
            continue

        # Grid search for window-specific thresholds
        gL, gU, sizes, ssr = grid_search_thresholds(w_sample)
        n1, n2, n3 = sizes
        n_w = len(w_sample)
        print(f"\n  {winfo['label']} (N={n_w:,}): γ_L={gL:.2f}  γ_U={gU:.2f}  "
              f"R1={n1}({n1/n_w*100:.1f}%)  R2={n2}({n2/n_w*100:.1f}%)  R3={n3}({n3/n_w*100:.1f}%)")

        # Assign window-specific regimes
        w_sample["regime"] = np.where(
            w_sample["ECT_lag"] < gL, 1,
            np.where(w_sample["ECT_lag"] > gU, 3, 2)
        )

        # Store for pooled model
        for idx in w_sample.index:
            window_assignments[idx] = w_sample.loc[idx, "regime"]

        # Per-regime OLS within window
        for r in [1, 2, 3]:
            sub = w_sample[w_sample["regime"] == r]
            if len(sub) < 20:
                print(f"    Regime {r}: too few obs ({len(sub)}), skipping")
                continue

            y = sub["dsa"].values.astype(float)
            X = build_X(sub)
            res = fit_ols_hac(y, X)

            dis_idx = vn.index("battery_discharging_mw")
            chg_idx = vn.index("battery_charging_mw")
            ect_idx = vn.index("ECT_lag")

            bd  = float(res.params[dis_idx])
            bds = float(res.bse[dis_idx])
            bdp = float(res.pvalues[dis_idx])

            print(f"    Regime {r} (N={len(sub):,}): β_dis={bd:+.4f}  SE={bds:.4f}  "
                  f"p={bdp:.4f}{stars(bdp)}  R²={res.rsquared:.4f}")

            coef_rows.append({
                "window":    winfo["label"],
                "regime":    r,
                "n_obs":     len(sub),
                "gamma_L":   gL,
                "gamma_U":   gU,
                "alpha":     float(res.params[ect_idx]),
                "alpha_se":  float(res.bse[ect_idx]),
                "alpha_p":   float(res.pvalues[ect_idx]),
                "beta_dis":  bd,
                "beta_dis_se": bds,
                "beta_dis_p":  bdp,
                "beta_dis_stars": stars(bdp),
                "beta_chg":  float(res.params[chg_idx]),
                "beta_chg_se": float(res.bse[chg_idx]),
                "beta_chg_p":  float(res.pvalues[chg_idx]),
                "rsq":       float(res.rsquared),
            })

            if r == 3:
                summary_lines.append(
                    f"A3 {winfo['label']} R3       {bd:+.4f}      "
                    f"{bds:.4f}   {bdp:.4f}     "
                    f"{'Yes' if bdp < 0.10 else 'No'}"
                )

    # ── Pooled model with time-varying regime assignments ──────────────────
    print(f"\n  Pooled model (time-varying regimes) ...")

    # Build full lagged sample using fixed β ECT
    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")
    ect  = (sa - BETA_CI * vic).rename("ECT")

    rows_full = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}
    for lag in range(1, ECM_LAGS + 1):
        rows_full[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows_full[f"dvic_l{lag}"] = dvic.shift(lag)
    for c in CONTROLS:
        if c in df.columns:
            rows_full[c] = df[c].shift(1) if c in BATTERY_COLS else df[c]
        else:
            rows_full[c] = np.nan

    full_sample = pd.DataFrame(rows_full).dropna()

    # Assign time-varying regimes from window results
    tv_regime = pd.Series(np.nan, index=full_sample.index)
    for idx, reg in window_assignments.items():
        if idx in tv_regime.index:
            tv_regime[idx] = reg

    full_sample["regime"] = tv_regime
    full_sample = full_sample.dropna(subset=["regime"])
    full_sample["regime"] = full_sample["regime"].astype(int)

    print(f"  Pooled sample: N={len(full_sample):,}")

    for r in [1, 2, 3]:
        sub = full_sample[full_sample["regime"] == r]
        if len(sub) < 20:
            print(f"    Regime {r}: too few obs ({len(sub)})")
            continue

        y = sub["dsa"].values.astype(float)
        X = build_X(sub)
        res = fit_ols_hac(y, X)

        dis_idx = vn.index("battery_discharging_mw")
        chg_idx = vn.index("battery_charging_mw")
        ect_idx = vn.index("ECT_lag")

        bd  = float(res.params[dis_idx])
        bds = float(res.bse[dis_idx])
        bdp = float(res.pvalues[dis_idx])

        print(f"    Regime {r} (N={len(sub):,}): β_dis={bd:+.4f}  SE={bds:.4f}  "
              f"p={bdp:.4f}{stars(bdp)}  R²={res.rsquared:.4f}")

        coef_rows.append({
            "window":    "pooled_TV",
            "regime":    r,
            "n_obs":     len(sub),
            "gamma_L":   np.nan,
            "gamma_U":   np.nan,
            "alpha":     float(res.params[ect_idx]),
            "alpha_se":  float(res.bse[ect_idx]),
            "alpha_p":   float(res.pvalues[ect_idx]),
            "beta_dis":  bd,
            "beta_dis_se": bds,
            "beta_dis_p":  bdp,
            "beta_dis_stars": stars(bdp),
            "beta_chg":  float(res.params[chg_idx]),
            "beta_chg_se": float(res.bse[chg_idx]),
            "beta_chg_p":  float(res.pvalues[chg_idx]),
            "rsq":       float(res.rsquared),
        })

        if r == 3:
            summary_lines.append(
                f"A3 pooled TV       {bd:+.4f}      "
                f"{bds:.4f}   {bdp:.4f}     "
                f"{'Yes' if bdp < 0.10 else 'No'}"
            )

    # Save
    pd.DataFrame(coef_rows).to_csv(RESULTS_DIR / "phase10_approach3_rolling_coefs.csv", index=False)
    print(f"\n  Saved: results/phase10_approach3_rolling_coefs.csv")

    return summary_lines


# ============================================================================
# Summary
# ============================================================================
def build_summary(baseline_line, a1_lines, a2_lines, a3_lines):
    lines = [
        "PHASE 10 — LAGGED BATTERY ROBUSTNESS SUMMARY",
        "=" * 65,
        "",
        "Comparison of R3 battery discharge coefficient across approaches:",
        "",
        f"{'Approach':<19s} {'β_dis(R3)':>10s}  {'SE':>8s}   {'p-value':>7s}     {'Significant?':>12s}",
        "-" * 65,
        baseline_line,
    ]
    lines.extend(a1_lines)
    lines.extend(a2_lines)
    lines.extend(a3_lines)
    lines.append("")
    lines.append("Stars: *** p<0.01  ** p<0.05  * p<0.10")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Phase 3 baseline uses Johansen ECT, lagged battery, HAC(12)")
    lines.append("  - A1 uses lagged battery × normalised capacity interaction + delta-method")
    lines.append("  - A2 uses iterated two-step GARCH(1,1)-t in-mean + HAC")
    lines.append("  - A3 uses year-specific SSR-minimising thresholds, lagged battery")
    lines.append("  - A3 pooled TV uses full sample with time-varying regime assignments")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================
def main():
    print(DIVIDER)
    print("  PHASE 10: LAGGED BATTERY ROBUSTNESS — IMPROVING R3 SIGNIFICANCE")
    print(DIVIDER)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df = load_data()
    print(f"  Obs: {len(df):,}  ({df.index.min().date()} to {df.index.max().date()})")

    # ── 2. Estimate ECT and build lagged sample ──────────────────────────
    print("\n[2] Estimating ECT and building lagged sample ...")
    beta, ect = estimate_ect(df)
    sample    = build_sample_lagged(df, ect)
    print(f"  Cointegrating vector: sa − {-beta[1]:.4f} × vic")
    print(f"  ECM sample: T = {len(sample):,}")
    for r in [1, 2, 3]:
        n = (sample["regime"] == r).sum()
        print(f"    Regime {r}: N = {n:,}  ({n/len(sample)*100:.1f}%)")

    # ── 3. Baseline Phase 3 estimates ─────────────────────────────────────
    print("\n[3] Baseline Phase 3 lagged OLS ...")
    vn = regressor_names()
    dis_idx = vn.index("battery_discharging_mw")

    baseline_results = {}
    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r]
        y   = sub["dsa"].values.astype(float)
        X   = build_X(sub)
        baseline_results[r] = fit_ols_hac(y, X)

    bd3  = float(baseline_results[3].params[dis_idx])
    bds3 = float(baseline_results[3].bse[dis_idx])
    bdp3 = float(baseline_results[3].pvalues[dis_idx])
    print(f"  R3 β_dis = {bd3:+.4f}  SE = {bds3:.4f}  p = {bdp3:.4f}")

    baseline_line = (
        f"Phase 3 baseline   {bd3:+.4f}      "
        f"{bds3:.4f}   {bdp3:.4f}     No"
    )

    # ── 4. Approach 1 ─────────────────────────────────────────────────────
    print("\n[4] Running Approach 1 ...")
    a1_lines = run_approach1(sample)

    # ── 5. Approach 2 ─────────────────────────────────────────────────────
    print("\n[5] Running Approach 2 ...")
    a2_lines = run_approach2(sample, baseline_results)

    # ── 6. Approach 3 ─────────────────────────────────────────────────────
    print("\n[6] Running Approach 3 ...")
    a3_lines = run_approach3(df)

    # ── 7. Summary ────────────────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  SUMMARY")
    print(DIVIDER)

    summary_text = build_summary(baseline_line, a1_lines, a2_lines, a3_lines)
    (RESULTS_DIR / "phase10_summary.txt").write_text(summary_text)
    print(f"\n  Saved: results/phase10_summary.txt\n")
    print(summary_text)

    print("\n" + DIVIDER)
    print("  PHASE 10 COMPLETE — 7 output files in results/")
    print(DIVIDER)


if __name__ == "__main__":
    main()
