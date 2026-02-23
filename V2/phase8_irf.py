#!/usr/bin/env python3
"""
Phase 8: Regime-Conditional Impulse-Response Functions (IRFs)
==============================================================
Simulates the bivariate (SA, VIC) price response to a 1 MW battery
discharge shock over 12 periods (6 hours) within each TVECM regime.

Method:
  - Regime-fixed linear recursion from Phase 2 OLS coefficients
  - ECT tracking: ect[h] = ect[h-1] + Δsa[h] − β * Δvic[h]
  - Wild bootstrap (Rademacher, N=500) for 80 % CI bands on SA level IRF
  - All control variables held at zero (ceteris paribus)

Fixed parameters from Phase 1 / Phase 2:
  β_CI   = 1.2103   (Johansen cointegrating vector)
  γ_L    = −45.80,  γ_U = +20.02  (threshold values)
  L      = 4        (ECM lags)

Output:
  results/phase8_irf_data.csv          — point + CI by regime × horizon
  results/phase8_irf.png               — 3-panel figure
  results/phase8_summary.txt
"""

import warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BETA_CI    = 1.2103
GAMMA_L    = -45.8039
GAMMA_U    =  20.0215
ECM_LAGS   = 4
HAC_LAGS   = 12
HORIZON    = 12        # periods (30-min each → 6 hours)
SHOCK_MW   = 1.0       # 1 MW battery discharge shock
N_BOOT     = 500       # wild bootstrap draws
ALPHA_CI   = 0.80      # confidence level
CI_LO      = (1 - ALPHA_CI) / 2          # 0.10
CI_HI      = 1 - (1 - ALPHA_CI) / 2     # 0.90

CONTROLS = [
    "wind_mw", "solar_total_mw", "demand_mw",
    "battery_discharging_mw", "battery_charging_mw", "interconnector_mw",
]

DIVIDER = "=" * 72


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
    Full-sample ECM design matrix (same construction as Phase 2).
    Returns a DataFrame with dsa, dvic, and all regressors.
    """
    sa   = df["sa_price"]
    vic  = df["vic_price"]
    dsa  = sa.diff().rename("dsa")
    dvic = vic.diff().rename("dvic")
    ect  = (sa - BETA_CI * vic).rename("ECT")

    rows = {"dsa": dsa, "dvic": dvic, "ECT_lag": ect.shift(1)}
    for lag in range(1, ECM_LAGS + 1):
        rows[f"dsa_l{lag}"]  = dsa.shift(lag)
        rows[f"dvic_l{lag}"] = dvic.shift(lag)
    for c in CONTROLS:
        rows[c] = df[c] if c in df.columns else np.nan

    sample = pd.DataFrame(rows).dropna()

    # Assign TVECM regimes based on ECT_lag
    ect_val = sample["ECT_lag"]
    sample["regime"] = 2  # band (default)
    sample.loc[ect_val < GAMMA_L, "regime"] = 1  # lower outer
    sample.loc[ect_val > GAMMA_U, "regime"] = 3  # upper outer

    return sample


# ---------------------------------------------------------------------------
# OLS estimation with HAC standard errors
# ---------------------------------------------------------------------------
FEATURE_COLS_SA  = (
    ["const", "ECT_lag"]
    + [f"dsa_l{i}" for i in range(1, ECM_LAGS + 1)]
    + [f"dvic_l{i}" for i in range(1, ECM_LAGS + 1)]
    + CONTROLS
)

FEATURE_COLS_VIC = FEATURE_COLS_SA  # same features, different LHS


def fit_ols_hac(y, X, hac_lags=HAC_LAGS):
    """Fit OLS with Newey-West HAC covariance. Returns (coef, cov, resid)."""
    res = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return res.params.values, res.cov_params().values, res.resid.values


def build_X(sub, eq="sa"):
    """Build design matrix with constant for equation `eq` ('sa' or 'vic')."""
    X = add_constant(sub[[c for c in FEATURE_COLS_SA if c != "const"]])
    return X


def extract_irf_coefs(coef, feature_names):
    """
    Extract the coefficients needed for the bivariate IRF recursion.
    Returns dict: alpha, phi (dsa lags), psi (dvic lags), beta_dis, beta_chg.
    """
    names = list(feature_names)

    def _c(name):
        if name in names:
            return coef[names.index(name)]
        return 0.0

    return {
        "alpha":    _c("ECT_lag"),
        "phi":      [_c(f"dsa_l{i}") for i in range(1, ECM_LAGS + 1)],
        "psi":      [_c(f"dvic_l{i}") for i in range(1, ECM_LAGS + 1)],
        "beta_dis": _c("battery_discharging_mw"),
        "beta_chg": _c("battery_charging_mw"),
    }


# ---------------------------------------------------------------------------
# IRF recursion
# ---------------------------------------------------------------------------
def compute_irf(coefs_sa, coefs_vic, horizon=HORIZON, shock=SHOCK_MW):
    """
    Bivariate TVECM impulse-response to a `shock` MW battery discharge at h=0.

    Returns:
      level_sa  : cumulative Δsa (SA price level response)
      level_vic : cumulative Δvic (VIC price level response)
    """
    d_sa  = np.zeros(horizon)
    d_vic = np.zeros(horizon)
    ect   = 0.0  # start from equilibrium

    for h in range(horizon):
        s = shock if h == 0 else 0.0

        phi_sa = psi_sa = phi_vic_dsa = phi_vic_dvic = 0.0
        for i in range(ECM_LAGS):
            if h - i - 1 >= 0:
                phi_sa       += coefs_sa["phi"][i]  * d_sa[h - i - 1]
                psi_sa       += coefs_sa["psi"][i]  * d_vic[h - i - 1]
                phi_vic_dsa  += coefs_vic["phi"][i] * d_sa[h - i - 1]
                phi_vic_dvic += coefs_vic["psi"][i] * d_vic[h - i - 1]

        d_sa[h]  = coefs_sa["alpha"]  * ect + phi_sa  + psi_sa  + coefs_sa["beta_dis"]  * s
        d_vic[h] = coefs_vic["alpha"] * ect + phi_vic_dsa + phi_vic_dvic + coefs_vic["beta_dis"] * s

        ect += d_sa[h] - BETA_CI * d_vic[h]

    return np.cumsum(d_sa), np.cumsum(d_vic)


# ---------------------------------------------------------------------------
# Wild bootstrap
# ---------------------------------------------------------------------------
def bootstrap_irf(sub_sa, sub_vic, X_sa, X_vic, coefs_sa, coefs_vic,
                   n_boot=N_BOOT, horizon=HORIZON, rng=None):
    """
    Wild bootstrap (Rademacher) CI for bivariate IRF.

    Returns array of shape (n_boot, horizon) for SA level IRF only.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    y_sa  = sub_sa["dsa"].values
    y_vic = sub_vic["dvic"].values  if "dvic" in sub_vic.columns else sub_vic["dsa"].values

    # Fitted values from point estimates
    fit_sa  = X_sa.values  @ coefs_sa["_coef_raw"]
    fit_vic = X_vic.values @ coefs_vic["_coef_raw"]

    resid_sa  = y_sa  - fit_sa
    resid_vic = y_vic - fit_vic

    n = len(resid_sa)
    boot_irf_sa = np.zeros((n_boot, horizon))

    for b in range(n_boot):
        # Same Rademacher signs for both equations (preserve cross-eq structure)
        w = rng.choice([-1.0, 1.0], size=n)

        y_sa_b  = fit_sa  + resid_sa  * w
        y_vic_b = fit_vic + resid_vic * w

        coef_sa_b, _, _ = fit_ols_hac(y_sa_b,  X_sa)
        coef_vic_b, _, _ = fit_ols_hac(y_vic_b, X_vic)

        fn = list(X_sa.columns)
        c_sa_b  = extract_irf_coefs(coef_sa_b,  fn)
        c_vic_b = extract_irf_coefs(coef_vic_b, fn)

        lv_sa, _ = compute_irf(c_sa_b, c_vic_b, horizon=horizon)
        boot_irf_sa[b] = lv_sa

    return boot_irf_sa


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)
    df  = load_data()
    sample = build_full_sample(df)

    regime_labels = {1: "Regime 1 (ECT < γ_L, discount)",
                     2: "Regime 2 (band, normal)",
                     3: "Regime 3 (ECT > γ_U, spike)"}
    regime_colors = {1: "#2166ac", 2: "#1a9641", 3: "#d73027"}
    feature_names = [c for c in FEATURE_COLS_SA if c != "const"]

    results_rows = []
    regime_coefs = {}

    lines = [DIVIDER, "PHASE 8 — TVECM IMPULSE-RESPONSE FUNCTIONS", DIVIDER,
             f"Shock: {SHOCK_MW} MW battery discharge at h=0",
             f"Horizon: {HORIZON} periods ({HORIZON * 0.5:.0f} hours)",
             f"Bootstrap: Rademacher wild, N={N_BOOT}, {ALPHA_CI*100:.0f}% CI",
             ""]

    for r in [1, 2, 3]:
        sub = sample[sample["regime"] == r].copy()
        n_obs = len(sub)

        # Build design matrices
        X_sub = build_X(sub, eq="sa")   # same structure for SA and VIC
        fn = list(X_sub.columns)

        y_sa  = sub["dsa"].values
        y_vic = sub["dvic"].values

        # Fit SA equation
        coef_sa, cov_sa, resid_sa = fit_ols_hac(y_sa, X_sub)
        c_sa = extract_irf_coefs(coef_sa, fn)
        c_sa["_coef_raw"] = coef_sa

        # Fit VIC equation
        coef_vic, cov_vic, resid_vic = fit_ols_hac(y_vic, X_sub)
        c_vic = extract_irf_coefs(coef_vic, fn)
        c_vic["_coef_raw"] = coef_vic

        regime_coefs[r] = (c_sa, c_vic, X_sub)

        # Point-estimate IRF
        lv_sa, lv_vic = compute_irf(c_sa, c_vic)

        # Bootstrap CI on SA level IRF
        boot_sa = bootstrap_irf(
            sub.copy(), sub.copy(), X_sub, X_sub,
            c_sa, c_vic, n_boot=N_BOOT, rng=rng
        )
        ci_lo = np.quantile(boot_sa, CI_LO, axis=0)
        ci_hi = np.quantile(boot_sa, CI_HI, axis=0)

        lines.append(f"--- Regime {r} (n={n_obs:,}) ---")
        lines.append(f"  α_SA={c_sa['alpha']:.4f}  α_VIC={c_vic['alpha']:.4f}")
        lines.append(f"  β_dis_SA={c_sa['beta_dis']:.4f}  β_dis_VIC={c_vic['beta_dis']:.4f}")
        lines.append(f"  SA level IRF at h=12: {lv_sa[-1]:+.4f} $/MWh")
        lines.append(f"  VIC level IRF at h=12: {lv_vic[-1]:+.4f} $/MWh")
        lines.append("")

        for h in range(HORIZON):
            results_rows.append({
                "regime": r,
                "horizon": h,
                "d_sa_irf": (lv_sa[h] - (lv_sa[h - 1] if h > 0 else 0)),
                "level_sa": lv_sa[h],
                "level_vic": lv_vic[h],
                "ci_lo_sa": ci_lo[h],
                "ci_hi_sa": ci_hi[h],
            })

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    hours = np.arange(HORIZON) * 0.5  # x-axis in hours

    for ax, r in zip(axes, [1, 2, 3]):
        rows = [rw for rw in results_rows if rw["regime"] == r]
        lv_sa  = np.array([rw["level_sa"]  for rw in rows])
        lv_vic = np.array([rw["level_vic"] for rw in rows])
        ci_lo  = np.array([rw["ci_lo_sa"]  for rw in rows])
        ci_hi  = np.array([rw["ci_hi_sa"]  for rw in rows])

        col = regime_colors[r]

        # CI band (SA)
        ax.fill_between(hours, ci_lo, ci_hi, alpha=0.25, color=col, label=f"{int(ALPHA_CI*100)}% CI (SA)")

        # SA level IRF
        ax.plot(hours, lv_sa,  color=col,   lw=2.0, linestyle="-",  label="SA price IRF")

        # VIC level IRF
        ax.plot(hours, lv_vic, color="gray", lw=1.5, linestyle="--", label="VIC price IRF")

        # Zero line
        ax.axhline(0, color="black", lw=0.8, linestyle=":")

        ax.set_title(f"Regime {r}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hours after shock", fontsize=9)
        ax.set_ylabel("$/MWh" if r == 1 else "", fontsize=9)
        ax.set_xticks(np.arange(0, HORIZON * 0.5 + 0.1, 1.0))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Subtitle with key parameters
    regime_info = (
        "R1: ECT < −45.80 (discount)  |  "
        "R2: band (normal)  |  "
        "R3: ECT > +20.02 (spike)"
    )
    fig.suptitle(
        f"TVECM Impulse-Response Functions\n"
        f"1 MW Battery Discharge Shock  —  {regime_info}",
        fontsize=10, y=1.02
    )
    fig.tight_layout()

    plot_path = RESULTS_DIR / "phase8_irf.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    lines.append(f"Plot saved: {plot_path}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    df_out = pd.DataFrame(results_rows)
    csv_path = RESULTS_DIR / "phase8_irf_data.csv"
    df_out.to_csv(csv_path, index=False)
    lines.append(f"IRF data saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Summary text
    # -----------------------------------------------------------------------
    txt_path = RESULTS_DIR / "phase8_summary.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    for l in lines:
        print(l)

    print(f"\nSummary saved: {txt_path}")


if __name__ == "__main__":
    main()
