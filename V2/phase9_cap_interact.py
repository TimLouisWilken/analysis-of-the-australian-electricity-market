#!/usr/bin/env python3
"""
Phase 9: Battery × Installed Capacity Interaction Terms
=========================================================
Tests whether the battery price-dampening effect grew as the SA BESS fleet
expanded 5.7× (274.8 MWh → 1,558 MWh) from 2023 to 2025.

Model (per regime):
  Δsa_t = α·ect_{t-1} + [ECM lags] + β_dis·bat_dis_t + γ_dis·(bat_dis_t × cap_norm_t)
         + β_chg·bat_chg_t + γ_chg·(bat_chg_t × cap_norm_t) + [other controls] + ε_t

  where  cap_norm_t = installed_capacity(t) / mean_capacity   (dimensionless index)

Key hypothesis:
  H₀: γ_dis = 0  (dampening effect constant over time)
  H₁: γ_dis ≠ 0  (dampening grew with fleet size)

  Marginal battery effect at time t:
    ∂Δsa_t / ∂bat_dis_t = β_dis + γ_dis × cap_norm_t

Output:
  results/phase9_interact_coefs.csv    — full coefficient table per regime
  results/phase9_predict_margins.csv   — β_total at 2023-Jan / 2024-Jan / 2025-Jan
  results/phase9_summary.txt
"""

import warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BETA_CI   = 1.2103
GAMMA_L   = -45.8039
GAMMA_U   =  20.0215
ECM_LAGS  = 4
HAC_LAGS  = 12

# SA BESS fleet: (name, commissioning_date, capacity_MWh)
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

CONTROLS_NO_BAT = ["wind_mw", "solar_total_mw", "demand_mw", "interconnector_mw"]

DIVIDER = "=" * 72

# Evaluation dates for predicted margins
EVAL_DATES = [
    ("2023-01-01", "Jan 2023 (pre-Torrens)"),
    ("2024-01-01", "Jan 2024 (post-Torrens)"),
    ("2025-12-01", "Dec 2025 (end-of-sample)"),
]


# ---------------------------------------------------------------------------
# Capacity helper
# ---------------------------------------------------------------------------
def installed_capacity_at(date: pd.Timestamp) -> float:
    """Total MWh of SA batteries commissioned at or before `date`."""
    return sum(
        mwh for _, comm, mwh in BATTERY_CAPACITY
        if pd.Timestamp(comm) <= date
    )


def add_cap_norm(df):
    """
    Add 'installed_mwh' and 'cap_norm' columns to df (indexed by datetime).
    cap_norm = installed_mwh / mean_installed_mwh over the sample.
    """
    cap_series = pd.Series(
        {idx: installed_capacity_at(idx) for idx in df.index},
        name="installed_mwh"
    )
    df = df.copy()
    df["installed_mwh"] = cap_series
    mean_cap = df["installed_mwh"].mean()
    df["cap_norm"] = df["installed_mwh"] / mean_cap
    return df, mean_cap


# ---------------------------------------------------------------------------
# Data loading and ECM sample construction
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_datetime"])
    df = df.set_index("settlement_datetime").sort_index()
    df = df[df["sa_price"].abs() < 15001]
    df = df[df["vic_price"].abs() < 15001]
    return df


def build_full_sample(df):
    """
    Full-sample ECM design matrix with interaction terms.
    Adds bat_dis_cap = bat_dis * cap_norm  and  bat_chg_cap = bat_chg * cap_norm.
    """
    # Add capacity normalisation first (on raw df, full index)
    df, mean_cap = add_cap_norm(df)

    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")
    ect  = (sa - BETA_CI * vic).rename("ECT")

    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}
    for lag in range(1, ECM_LAGS + 1):
        rows[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows[f"dvic_l{lag}"] = dvic.shift(lag)

    # Battery terms — contemporaneous (Phase 2 specification)
    rows["battery_discharging_mw"] = df["battery_discharging_mw"]
    rows["battery_charging_mw"]    = df["battery_charging_mw"]

    # Interaction terms
    rows["bat_dis_cap"] = df["battery_discharging_mw"] * df["cap_norm"]
    rows["bat_chg_cap"] = df["battery_charging_mw"]    * df["cap_norm"]

    # Other controls (no battery — already included above)
    for c in CONTROLS_NO_BAT:
        rows[c] = df[c] if c in df.columns else np.nan

    # Capacity itself (for reference)
    rows["installed_mwh"] = df["installed_mwh"]
    rows["cap_norm"]      = df["cap_norm"]

    sample = pd.DataFrame(rows).dropna()

    # Assign TVECM regimes
    ect_val = sample["ECT_lag"]
    sample["regime"] = 2
    sample.loc[ect_val < GAMMA_L, "regime"] = 1
    sample.loc[ect_val > GAMMA_U, "regime"] = 3

    return sample, mean_cap


# ---------------------------------------------------------------------------
# OLS with HAC — return full covariance for delta-method
# ---------------------------------------------------------------------------
def fit_ols_hac(y, X, hac_lags=HAC_LAGS):
    """Return (params_Series, cov DataFrame, result)."""
    res = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return res.params, res.cov_params(), res


def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
def get_feature_cols(with_interact=True):
    base = (
        ["ECT_lag"]
        + [f"dsa_l{i}" for i in range(1, ECM_LAGS + 1)]
        + [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)]
        + ["battery_discharging_mw", "battery_charging_mw"]
    )
    if with_interact:
        base += ["bat_dis_cap", "bat_chg_cap"]
    base += CONTROLS_NO_BAT
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    sample, mean_cap = build_full_sample(df)

    feat_interact = get_feature_cols(with_interact=True)
    feat_baseline = get_feature_cols(with_interact=False)

    lines = [DIVIDER,
             "PHASE 9 — BATTERY × CAPACITY INTERACTION TERMS",
             DIVIDER,
             f"cap_norm = installed_MWh / mean_MWh  (mean = {mean_cap:.1f} MWh)",
             f"ECM lags L={ECM_LAGS},  HAC bandwidth={HAC_LAGS}",
             ""]

    coef_rows   = []
    margin_rows = []

    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r].copy()
        n_obs = len(sub)

        y = sub["dsa"].values

        # --- Interacted model ---
        X_int = add_constant(sub[feat_interact])
        fn_int = list(X_int.columns)

        params_int, cov_int, res_int = fit_ols_hac(y, X_int)

        # --- Baseline model (no interaction) ---
        X_base = add_constant(sub[feat_baseline])
        fn_base = list(X_base.columns)
        params_base, cov_base, res_base = fit_ols_hac(y, X_base)

        # Extract key coefficients
        def _coef(params, name):
            return params[name] if name in params.index else 0.0

        beta_dis_int  = _coef(params_int, "battery_discharging_mw")
        gamma_dis_int = _coef(params_int, "bat_dis_cap")
        beta_chg_int  = _coef(params_int, "battery_charging_mw")
        gamma_chg_int = _coef(params_int, "bat_chg_cap")
        beta_dis_base = _coef(params_base, "battery_discharging_mw")

        # SE and p-value for interaction term (γ_dis)
        se_gamma_dis = np.sqrt(cov_int.loc["bat_dis_cap", "bat_dis_cap"])
        t_gamma_dis  = gamma_dis_int / se_gamma_dis if se_gamma_dis > 0 else np.nan
        p_gamma_dis  = 2 * stats.t.sf(abs(t_gamma_dis), df=n_obs - len(fn_int))

        se_gamma_chg = np.sqrt(cov_int.loc["bat_chg_cap", "bat_chg_cap"])
        t_gamma_chg  = gamma_chg_int / se_gamma_chg if se_gamma_chg > 0 else np.nan
        p_gamma_chg  = 2 * stats.t.sf(abs(t_gamma_chg), df=n_obs - len(fn_int))

        alpha_int = _coef(params_int, "ECT_lag")
        se_alpha  = np.sqrt(cov_int.loc["ECT_lag", "ECT_lag"])

        lines.append(f"=== Regime {r} (n={n_obs:,}) ===")
        lines.append(f"  β_dis (base, no interact): {beta_dis_base:+.4f}  [Phase 2 reference]")
        lines.append(f"  β_dis (interact model):    {beta_dis_int:+.4f}")
        lines.append(
            f"  γ_dis (bat×cap):           {gamma_dis_int:+.4f}  "
            f"SE={se_gamma_dis:.4f}  t={t_gamma_dis:.3f}  p={p_gamma_dis:.4f}{stars(p_gamma_dis)}"
        )
        lines.append(f"  β_chg (interact model):    {beta_chg_int:+.4f}")
        lines.append(
            f"  γ_chg (chg×cap):           {gamma_chg_int:+.4f}  "
            f"SE={se_gamma_chg:.4f}  t={t_gamma_chg:.3f}  p={p_gamma_chg:.4f}{stars(p_gamma_chg)}"
        )
        lines.append(f"  α (ECT adjustment):        {alpha_int:+.4f}  SE={se_alpha:.4f}")
        lines.append(f"  R²={res_int.rsquared:.4f}  AIC={res_int.aic:.1f}")
        lines.append("")

        # Predicted margins at evaluation dates
        # β_total(t) = β_dis + γ_dis * cap_norm(t)
        # Var(β_total) = Var(β_dis) + cap_norm² * Var(γ_dis) + 2*cap_norm*Cov(β_dis, γ_dis)
        var_bd  = cov_int.loc["battery_discharging_mw", "battery_discharging_mw"]
        var_gd  = cov_int.loc["bat_dis_cap", "bat_dis_cap"]
        cov_bdg = cov_int.loc["battery_discharging_mw", "bat_dis_cap"]

        lines.append(f"  Marginal effect of battery discharge at evaluation dates:")
        for date_str, label in EVAL_DATES:
            cap_t     = installed_capacity_at(pd.Timestamp(date_str))
            cap_norm_t = cap_t / mean_cap
            beta_total = beta_dis_int + gamma_dis_int * cap_norm_t
            var_total  = var_bd + cap_norm_t**2 * var_gd + 2 * cap_norm_t * cov_bdg
            se_total   = np.sqrt(var_total)
            t_total    = beta_total / se_total if se_total > 0 else np.nan
            p_total    = 2 * stats.t.sf(abs(t_total), df=n_obs - len(fn_int))

            lines.append(
                f"    {label}: cap={cap_t:.0f} MWh  cap_norm={cap_norm_t:.3f}  "
                f"β_total={beta_total:+.4f}  SE={se_total:.4f}  p={p_total:.4f}{stars(p_total)}"
            )

            margin_rows.append({
                "regime":       r,
                "date":         date_str,
                "label":        label,
                "cap_mwh":      cap_t,
                "cap_norm":     cap_norm_t,
                "beta_dis":     beta_dis_int,
                "gamma_dis":    gamma_dis_int,
                "beta_total":   beta_total,
                "se_total":     se_total,
                "t_total":      t_total,
                "p_total":      p_total,
                "stars":        stars(p_total),
            })

        lines.append("")

        # Append coefficient rows (all vars in interacted model)
        for nm in fn_int:
            p_val = res_int.pvalues[nm]
            coef_rows.append({
                "regime":    r,
                "n_obs":     n_obs,
                "variable":  nm,
                "coef":      params_int[nm],
                "hac_se":    res_int.bse[nm],   # HAC SE stored in bse for HAC fit
                "t_stat":    res_int.tvalues[nm],
                "p_value":   p_val,
                "stars":     stars(p_val),
            })

    # -----------------------------------------------------------------------
    # Summary: how much did dampening grow?
    # -----------------------------------------------------------------------
    lines.append(DIVIDER)
    lines.append("CAPACITY GROWTH EFFECT SUMMARY")
    lines.append(DIVIDER)
    for r in [1, 2, 3]:
        rows_r = [m for m in margin_rows if m["regime"] == r]
        if len(rows_r) >= 2:
            early = rows_r[0]["beta_total"]
            late  = rows_r[-1]["beta_total"]
            change = late - early
            lines.append(f"Regime {r}: β_total(2023-Jan)={early:+.4f} → β_total(2025-Jun)={late:+.4f}  "
                         f"(Δ={change:+.4f}  "
                         f"{'dampening grew' if (change < 0 and late < 0) else 'see signs'})")
    lines.append("")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    df_coef = pd.DataFrame(coef_rows)
    df_coef.to_csv(RESULTS_DIR / "phase9_interact_coefs.csv", index=False)

    df_margins = pd.DataFrame(margin_rows)
    df_margins.to_csv(RESULTS_DIR / "phase9_predict_margins.csv", index=False)

    txt_path = RESULTS_DIR / "phase9_summary.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    for l in lines:
        print(l)

    print(f"\nCoefficients saved: {RESULTS_DIR / 'phase9_interact_coefs.csv'}")
    print(f"Margins saved:      {RESULTS_DIR / 'phase9_predict_margins.csv'}")
    print(f"Summary saved:      {txt_path}")


if __name__ == "__main__":
    main()
