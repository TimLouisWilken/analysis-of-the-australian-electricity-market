#!/usr/bin/env python3
"""
Phase 11: Markov-Switching ECM (MS-ECM)
========================================
3-state Markov-Switching ECM for Δsa_t with endogenous regime switching
via Hamilton filter.  Regimes emerge from data rather than fixed ECT
thresholds (cf. TVECM Phase 2).

Model:
  Δsa_t = μ_{s_t} + α_{s_t}·ECT_{t-1}
         + Σ φ_{i,s_t}·Δsa_{t-i} + Σ ψ_{i,s_t}·Δvic_{t-i}
         + β_dis_{s_t}·bat_dis_{t-1} + β_chg_{s_t}·bat_chg_{t-1}
         + [controls] + ε_t
  ε_t ~ N(0, σ²_{s_t})
  P(s_t=j | s_{t-1}=i) = p_{ij}   (3×3 transition matrix)

Estimation: Custom Hamilton filter + Kim smoother + EM algorithm
(statsmodels.MarkovRegression crashes on NumPy 2.4)

Output files:
  results/phase11_ms_coefs.csv        — regime-specific coefficients with SEs
  results/phase11_ms_transition.csv   — 3×3 transition probability matrix
  results/phase11_ms_probs.csv        — smoothed regime probabilities
  results/phase11_ms_concordance.csv  — MS vs TVECM regime agreement
  results/phase11_ms_summary.txt      — narrative summary
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

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH   = Path(__file__).parent / "data" / "processed" / "sa_nem_2023_2025.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GAMMA_L  = -45.8039
GAMMA_U  =  20.0215
ECM_LAGS =  4
HAC_LAGS = 12

N_REGIMES   = 3
EM_MAX_ITER = 200
EM_TOL      = 1e-6
N_RESTARTS  = 5

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
# Data loading (from Phase 3)
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
    ect  = prices.values @ beta
    ect  = pd.Series(ect, index=prices.index, name="ECT")
    return beta, ect


def build_sample_lagged(df, ect):
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


def stars(p):
    if not np.isfinite(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# ---------------------------------------------------------------------------
# Hamilton filter (vectorised over regimes)
# ---------------------------------------------------------------------------
def hamilton_filter(y, X, betas, sigmas, trans_mat):
    """
    Hamilton (1989) filter.  Vectorised: regime-conditional densities
    computed for all K regimes simultaneously via matrix ops.
    Time loop is sequential (inherent to the filter).
    """
    T = len(y)
    K = len(sigmas)

    # Pre-compute all regime residuals: resid[j] = y - X @ beta_j, shape (K, T)
    fitted = betas @ X.T                        # (K, T)
    resid  = y[None, :] - fitted                # (K, T)

    # Pre-compute regime-conditional densities for all t: eta (K, T)
    sig = np.maximum(sigmas, 1e-10)
    eta = np.exp(-0.5 * (resid / sig[:, None])**2) / (sig[:, None] * np.sqrt(2 * np.pi))

    filtered  = np.zeros((T, K))
    predicted = np.zeros((T, K))

    # Steady-state initial distribution
    eigvals, eigvecs = np.linalg.eig(trans_mat.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi0 = np.real(eigvecs[:, idx])
    pi0 = np.abs(pi0) / np.abs(pi0).sum()

    log_lik = 0.0

    for t in range(T):
        if t == 0:
            predicted[t] = pi0
        else:
            predicted[t] = filtered[t-1] @ trans_mat

        # Update: joint = prediction * conditional density
        joint = predicted[t] * eta[:, t]
        f_t = joint.sum()
        if f_t < 1e-300:
            f_t = 1e-300
        filtered[t] = joint / f_t
        log_lik += np.log(f_t)

    return filtered, predicted, log_lik


# ---------------------------------------------------------------------------
# Kim smoother (vectorised inner loops)
# ---------------------------------------------------------------------------
def kim_smoother(filtered, predicted, trans_mat):
    """Kim (1994) backward smoother — vectorised over regimes."""
    T, K = filtered.shape
    smoothed = np.zeros((T, K))
    # Only store transition sum per state (not full joint)
    # smoothed_marginal[i] = Σ_j smoothed_joint[t, i, j] summed across t
    trans_numer = np.zeros((K, K))  # numerator for transition matrix update

    smoothed[T - 1] = filtered[T - 1]

    for t in range(T - 2, -1, -1):
        pred_safe = np.maximum(predicted[t + 1], 1e-300)
        ratio = smoothed[t + 1] / pred_safe    # (K,)

        # smoothed_joint[t, i, j] = filtered[t,i] * P[i,j] * smoothed[t+1,j] / predicted[t+1,j]
        # = filtered[t, i] * P[i, j] * ratio[j]
        # outer product: filtered[t] (K,) x (trans_mat * ratio) → (K, K)
        sj = filtered[t][:, None] * trans_mat * ratio[None, :]  # (K, K)
        trans_numer += sj

        smoothed[t] = sj.sum(axis=1)
        s = smoothed[t].sum()
        if s > 0:
            smoothed[t] /= s

    return smoothed, trans_numer


# ---------------------------------------------------------------------------
# EM algorithm
# ---------------------------------------------------------------------------
def em_m_step(y, X, smoothed, trans_numer):
    """M-step: update params given smoothed probs.  No large matrices."""
    T, K = smoothed.shape
    k = X.shape[1]

    betas  = np.zeros((K, k))
    sigmas = np.zeros(K)

    for j in range(K):
        w = smoothed[:, j]
        w_sum = w.sum()
        if w_sum < 1.0:
            betas[j] = np.linalg.lstsq(X, y, rcond=None)[0]
            sigmas[j] = np.std(y - X @ betas[j])
            continue

        # Weighted OLS via sqrt-weight trick (no TxT matrix!)
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]    # (T, k) — elementwise
        yw = y * sqrt_w             # (T,)
        XWX = Xw.T @ Xw            # (k, k)
        XWy = Xw.T @ yw            # (k,)
        try:
            betas[j] = np.linalg.solve(XWX, XWy)
        except np.linalg.LinAlgError:
            betas[j] = np.linalg.lstsq(XWX, XWy, rcond=None)[0]

        resid = y - X @ betas[j]
        sigmas[j] = np.sqrt(max((w * resid**2).sum() / w_sum, 1e-8))

    # Transition matrix from accumulated joint probs
    trans_mat = np.zeros((K, K))
    for i in range(K):
        row_sum = trans_numer[i].sum()
        if row_sum > 0:
            trans_mat[i] = trans_numer[i] / row_sum
        else:
            trans_mat[i] = 1.0 / K

    # Ensure proper probability matrix
    for i in range(K):
        trans_mat[i] = np.maximum(trans_mat[i], 1e-8)
        trans_mat[i] /= trans_mat[i].sum()

    return betas, sigmas, trans_mat


def initialise_params(y, X, rng, init_method="random"):
    T, k = X.shape
    K = N_REGIMES

    if init_method == "kmeans":
        sorted_idx = np.argsort(y)
        split = np.array_split(sorted_idx, K)
        betas  = np.zeros((K, k))
        sigmas = np.zeros(K)
        for j in range(K):
            idx = split[j]
            if len(idx) > k + 2:
                betas[j] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
                sigmas[j] = max(np.std(y[idx] - X[idx] @ betas[j]), 1.0)
            else:
                betas[j] = np.linalg.lstsq(X, y, rcond=None)[0]
                sigmas[j] = np.std(y)
    else:
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_ols = np.std(y - X @ beta_ols)
        betas  = np.zeros((K, k))
        sigmas = np.zeros(K)
        for j in range(K):
            betas[j] = beta_ols + rng.normal(0, 0.1 * np.abs(beta_ols) + 0.01, size=k)
            scale_factors = [0.3, 1.0, 3.0]
            sigmas[j] = max(sigma_ols * scale_factors[j] * (0.8 + 0.4 * rng.random()), 1.0)

    trans_mat = np.zeros((K, K))
    for i in range(K):
        diag_prob = 0.7 + 0.2 * rng.random()
        off_diag = rng.dirichlet(np.ones(K - 1)) * (1 - diag_prob)
        trans_mat[i, i] = diag_prob
        off_idx = [j for j in range(K) if j != i]
        for idx_j, j in enumerate(off_idx):
            trans_mat[i, j] = off_diag[idx_j]

    return betas, sigmas, trans_mat


def fit_ms_ecm(y, X, n_restarts=N_RESTARTS, verbose=True):
    T, k = X.shape
    K = N_REGIMES

    best_ll = -np.inf
    best_params = None
    best_smoothed = None
    best_history = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed=42 + restart * 17)
        method = "kmeans" if restart == 0 else "random"
        betas, sigmas, trans_mat = initialise_params(y, X, rng, method)

        ll_history = []
        converged = False

        for it in range(EM_MAX_ITER):
            try:
                filtered, predicted, log_lik = hamilton_filter(
                    y, X, betas, sigmas, trans_mat
                )
            except Exception:
                log_lik = -np.inf
                break

            if not np.isfinite(log_lik):
                break

            ll_history.append(log_lik)

            if it > 0 and abs(ll_history[-1] - ll_history[-2]) < EM_TOL:
                converged = True
                break

            smoothed, trans_numer = kim_smoother(filtered, predicted, trans_mat)
            betas, sigmas, trans_mat = em_m_step(y, X, smoothed, trans_numer)

        if log_lik > best_ll and np.isfinite(log_lik):
            filtered, predicted, log_lik = hamilton_filter(
                y, X, betas, sigmas, trans_mat
            )
            smoothed, trans_numer = kim_smoother(filtered, predicted, trans_mat)

            best_ll = log_lik
            best_params = (betas.copy(), sigmas.copy(), trans_mat.copy())
            best_smoothed = smoothed.copy()
            best_history = ll_history.copy()

        if verbose:
            n_it = len(ll_history)
            print(f"    Restart {restart+1}/{n_restarts}: "
                  f"LL={log_lik:.2f}  iter={n_it}  "
                  f"{'converged' if converged else 'not converged'}")

    if best_params is None:
        raise RuntimeError("All EM restarts failed")

    return best_params, best_smoothed, best_ll, best_history


# ---------------------------------------------------------------------------
# Standard errors (EM-based weighted OLS)
# ---------------------------------------------------------------------------
def compute_standard_errors(y, X, betas, sigmas, trans_mat, smoothed):
    """
    SEs from weighted OLS per regime (Hamilton 1994, Ch. 22).
    Uses smoothed probabilities as weights — standard for EM-estimated MS models.
    """
    K, k = betas.shape
    T = len(y)

    beta_ses = np.zeros((K, k))
    sigma_ses = np.zeros(K)

    for j in range(K):
        w = smoothed[:, j]
        w_sum = w.sum()
        if w_sum < 2.0:
            beta_ses[j] = np.full(k, np.nan)
            sigma_ses[j] = np.nan
            continue

        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]
        try:
            XWX_inv = np.linalg.inv(Xw.T @ Xw)
            beta_ses[j] = np.sqrt(np.maximum(np.diag(sigmas[j]**2 * XWX_inv), 0))
        except np.linalg.LinAlgError:
            beta_ses[j] = np.full(k, np.nan)

        eff_n = w_sum
        sigma_ses[j] = sigmas[j] / np.sqrt(max(2 * eff_n, 1))

    trans_ses = np.full((K, K), np.nan)
    for i in range(K):
        n_from_i = smoothed[:-1, i].sum()
        if n_from_i > 1:
            for j_col in range(K):
                p = trans_mat[i, j_col]
                trans_ses[i, j_col] = np.sqrt(max(p * (1 - p) / n_from_i, 0))

    return beta_ses, sigma_ses, trans_ses


# ---------------------------------------------------------------------------
# Regime ordering + concordance
# ---------------------------------------------------------------------------
def order_regimes(betas, sigmas, trans_mat, smoothed, beta_ses, sigma_ses, trans_ses):
    order = np.argsort(sigmas)
    betas_o     = betas[order]
    sigmas_o    = sigmas[order]
    smoothed_o  = smoothed[:, order]
    beta_ses_o  = beta_ses[order]
    sigma_ses_o = sigma_ses[order]
    trans_mat_o = trans_mat[np.ix_(order, order)]
    trans_ses_o = trans_ses[np.ix_(order, order)]
    return betas_o, sigmas_o, trans_mat_o, smoothed_o, beta_ses_o, sigma_ses_o, trans_ses_o


def compare_with_tvecm(smoothed, tvecm_regimes):
    ms_regimes = np.argmax(smoothed, axis=1) + 1
    tvecm = tvecm_regimes.values

    total = len(ms_regimes)
    agree = (ms_regimes == tvecm).sum()
    concordance = agree / total

    rows = []
    rows.append({"metric": "overall_concordance", "value": concordance,
                 "n_agree": agree, "n_total": total})

    for r in [1, 2, 3]:
        tvecm_mask = tvecm == r
        ms_mask = ms_regimes == r
        if tvecm_mask.sum() > 0:
            agree_r = (ms_regimes[tvecm_mask] == r).sum()
            conc_r = agree_r / tvecm_mask.sum()
        else:
            agree_r = 0
            conc_r = np.nan
        rows.append({"metric": f"regime_{r}_concordance", "value": conc_r,
                     "n_agree": agree_r, "n_total": int(tvecm_mask.sum()),
                     "ms_n": int(ms_mask.sum()), "tvecm_n": int(tvecm_mask.sum())})

    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            n_ij = ((tvecm == i) & (ms_regimes == j)).sum()
            rows.append({"metric": f"confusion_tvecm{i}_ms{j}",
                         "value": n_ij / total if total > 0 else 0,
                         "n_agree": int(n_ij), "n_total": total})

    return pd.DataFrame(rows), ms_regimes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(DIVIDER)
    print("  PHASE 11: MARKOV-SWITCHING ECM (MS-ECM)")
    print("  Custom Hamilton filter + EM algorithm")
    print(DIVIDER)

    # ── 1. Data ────────────────────────────────────────────────────────────
    print("\n[1] Loading data and computing ECT ...")
    df = load_data()
    beta_ci, ect = estimate_ect(df)
    print(f"  Cointegrating vector: sa − {-beta_ci[1]:.4f} × vic")
    print(f"  Obs (after price-cap filter): {len(df):,}")

    # ── 2. Build sample ───────────────────────────────────────────────────
    print("\n[2] Building sample (lagged battery) ...")
    sample = build_sample_lagged(df, ect)
    T = len(sample)
    print(f"  ECM sample: T = {T:,}")
    for r in [1, 2, 3]:
        n = (sample["regime"] == r).sum()
        print(f"    TVECM Regime {r}: N = {n:,}  ({n/T*100:.1f}%)")

    vn = regressor_names()
    y = sample["dsa"].values.astype(float)
    X = build_X(sample)
    k = X.shape[1]

    # ── 3. Fit MS-ECM ─────────────────────────────────────────────────────
    print(f"\n[3] Fitting {N_REGIMES}-state MS-ECM via EM "
          f"({N_RESTARTS} restarts, max {EM_MAX_ITER} iter) ...")
    params, smoothed, log_lik, ll_history = fit_ms_ecm(y, X, verbose=True)
    betas, sigmas, trans_mat = params

    print(f"\n  Best log-likelihood: {log_lik:.2f}")
    print(f"  EM iterations: {len(ll_history)}")
    mono = all(ll_history[i] <= ll_history[i+1] + 1e-8
               for i in range(len(ll_history)-1))
    print(f"  LL monotonicity: {mono}")

    # ── 4. Standard errors ────────────────────────────────────────────────
    print("\n[4] Computing standard errors (weighted OLS / EM-based) ...")
    beta_ses, sigma_ses, trans_ses = compute_standard_errors(
        y, X, betas, sigmas, trans_mat, smoothed
    )

    # ── 5. Order regimes by variance ──────────────────────────────────────
    print("\n[5] Ordering regimes by variance ...")
    betas, sigmas, trans_mat, smoothed, beta_ses, sigma_ses, trans_ses = \
        order_regimes(betas, sigmas, trans_mat, smoothed,
                      beta_ses, sigma_ses, trans_ses)

    print(f"  σ by state: "
          f"{', '.join(f'S{j+1}={sigmas[j]:.2f}' for j in range(N_REGIMES))}")

    ms_assign = np.argmax(smoothed, axis=1) + 1
    for j in range(1, N_REGIMES + 1):
        n_j = (ms_assign == j).sum()
        print(f"    MS State {j}: N = {n_j:,}  ({n_j/T*100:.1f}%)")

    # ── 6. Display results ────────────────────────────────────────────────
    print(f"\n[6] Regime-specific coefficients:")
    dis_idx = vn.index("battery_discharging_mw")
    chg_idx = vn.index("battery_charging_mw")
    ect_idx = vn.index("ECT_lag")

    print(f"\n  {'Variable':<30s}  {'State 1 (low σ)':>16s}  "
          f"{'State 2 (mid σ)':>16s}  {'State 3 (high σ)':>16s}")
    print("  " + "-" * 84)
    for idx_v, vname in [(0, "const (μ)"),
                          (ect_idx, "ECT_lag (α)"),
                          (dis_idx, "battery_dis (β_dis)"),
                          (chg_idx, "battery_chg (β_chg)")]:
        row = f"  {vname:<30s}"
        for j in range(N_REGIMES):
            c = betas[j, idx_v]
            se = beta_ses[j, idx_v]
            p = 2 * sp_stats.norm.sf(abs(c / se)) if se > 0 else np.nan
            row += f"  {c:>+8.4f}{stars(p):<3s}({se:.4f})"
        print(row)

    row = f"  {'σ (std dev)':<30s}"
    for j in range(N_REGIMES):
        row += f"  {sigmas[j]:>8.2f}   ({sigma_ses[j]:.2f})"
    print(row)

    print(f"\n  Transition matrix:")
    print(f"  {'':>12s}  {'→ S1':>8s}  {'→ S2':>8s}  {'→ S3':>8s}")
    for i in range(N_REGIMES):
        row = f"  S{i+1} →       "
        for j in range(N_REGIMES):
            row += f"  {trans_mat[i,j]:>8.4f}"
        print(row)

    # ── 7. Compare with TVECM ─────────────────────────────────────────────
    print(f"\n[7] Comparing with TVECM regime assignments ...")
    concordance_df, ms_regimes = compare_with_tvecm(smoothed, sample["regime"])
    overall_conc = concordance_df[
        concordance_df["metric"] == "overall_concordance"
    ]["value"].iloc[0]
    print(f"  Overall concordance: {overall_conc:.1%}")

    for r in [1, 2, 3]:
        row = concordance_df[
            concordance_df["metric"] == f"regime_{r}_concordance"
        ]
        if len(row) > 0:
            print(f"    Regime {r}: {row['value'].iloc[0]:.1%} agreement "
                  f"(TVECM N={row['n_total'].iloc[0]:,})")

    # ── 8. Save results ───────────────────────────────────────────────────
    print(f"\n[8] Saving results ...")

    # 8a. Coefficients CSV
    coef_rows = []
    for j in range(N_REGIMES):
        for i_v, vname in enumerate(vn):
            c = betas[j, i_v]
            se = beta_ses[j, i_v]
            t_stat = c / se if se > 0 else np.nan
            p_val = 2 * sp_stats.norm.sf(abs(t_stat)) if np.isfinite(t_stat) else np.nan
            coef_rows.append({
                "state": j + 1, "variable": vname,
                "coef": c, "se": se, "t_stat": t_stat,
                "p_value": p_val, "stars": stars(p_val),
                "sigma": sigmas[j], "sigma_se": sigma_ses[j],
            })
    pd.DataFrame(coef_rows).to_csv(
        RESULTS_DIR / "phase11_ms_coefs.csv", index=False)
    print(f"  Saved: results/phase11_ms_coefs.csv")

    # 8b. Transition matrix
    trans_rows = []
    for i in range(N_REGIMES):
        for j in range(N_REGIMES):
            trans_rows.append({
                "from_state": i + 1, "to_state": j + 1,
                "probability": trans_mat[i, j],
                "se": trans_ses[i, j] if np.isfinite(trans_ses[i, j]) else np.nan,
            })
    pd.DataFrame(trans_rows).to_csv(
        RESULTS_DIR / "phase11_ms_transition.csv", index=False)
    print(f"  Saved: results/phase11_ms_transition.csv")

    # 8c. Smoothed probabilities
    pd.DataFrame({
        "datetime": sample.index,
        "prob_state1": smoothed[:, 0],
        "prob_state2": smoothed[:, 1],
        "prob_state3": smoothed[:, 2],
        "ms_regime": ms_regimes,
        "tvecm_regime": sample["regime"].values,
    }).to_csv(RESULTS_DIR / "phase11_ms_probs.csv", index=False)
    print(f"  Saved: results/phase11_ms_probs.csv")

    # 8d. Concordance
    concordance_df.to_csv(
        RESULTS_DIR / "phase11_ms_concordance.csv", index=False)
    print(f"  Saved: results/phase11_ms_concordance.csv")

    # 8e. Summary
    n_params_total = N_REGIMES * k + N_REGIMES + N_REGIMES * (N_REGIMES - 1)
    aic = -2 * log_lik + 2 * n_params_total
    bic = -2 * log_lik + np.log(T) * n_params_total

    c_r3 = betas[N_REGIMES-1, dis_idx]
    se_r3 = beta_ses[N_REGIMES-1, dis_idx]
    p_r3 = 2 * sp_stats.norm.sf(abs(c_r3 / se_r3)) if se_r3 > 0 else np.nan

    lines = []
    add = lines.append
    add("PHASE 11 — MARKOV-SWITCHING ECM SUMMARY")
    add("=" * 65)
    add(f"Sample: T = {T:,}  ({sample.index[0]}  to  {sample.index[-1]})")
    add(f"Specification: {N_REGIMES}-state MS-ECM, ECM lags = {ECM_LAGS}")
    add(f"Estimation: EM algorithm, {N_RESTARTS} restarts, max {EM_MAX_ITER} iter")
    add(f"Best log-likelihood: {log_lik:.2f}")
    add(f"AIC: {aic:.2f}  BIC: {bic:.2f}  (n_params = {n_params_total})")
    add("")

    add("REGIME-SPECIFIC COEFFICIENTS")
    add("-" * 65)
    add(f"{'Variable':<28s}  {'S1 (low σ)':>14s}  {'S2 (mid σ)':>14s}  "
        f"{'S3 (high σ)':>14s}")
    add("-" * 65)
    for idx_v, vname in [(0, "const (μ)"),
                          (ect_idx, "ECT_lag (α)"),
                          (dis_idx, "bat_dis (β_dis)"),
                          (chg_idx, "bat_chg (β_chg)")]:
        row = f"{vname:<28s}"
        for j in range(N_REGIMES):
            c = betas[j, idx_v]
            se = beta_ses[j, idx_v]
            p = 2 * sp_stats.norm.sf(abs(c / se)) if se > 0 else np.nan
            row += f"  {c:>+8.4f}{stars(p):<3s}   "
        add(row)
    row = f"{'σ':<28s}"
    for j in range(N_REGIMES):
        row += f"  {sigmas[j]:>8.2f}       "
    add(row)
    add("")

    add("TRANSITION MATRIX")
    add("-" * 45)
    add(f"{'':>12s}  {'→ S1':>8s}  {'→ S2':>8s}  {'→ S3':>8s}")
    for i in range(N_REGIMES):
        row = f"S{i+1} →       "
        for j in range(N_REGIMES):
            row += f"  {trans_mat[i,j]:>8.4f}"
        add(row)
    add("")

    add("CONCORDANCE WITH TVECM")
    add("-" * 45)
    add(f"Overall concordance: {overall_conc:.1%}")
    for r in [1, 2, 3]:
        row = concordance_df[concordance_df["metric"] == f"regime_{r}_concordance"]
        if len(row) > 0:
            add(f"  Regime {r}: {row['value'].iloc[0]:.1%} agreement")
    add("")

    add("COMPARISON WITH TVECM BATTERY COEFFICIENTS")
    add("-" * 65)
    add(f"{'':>12s}  {'MS-ECM β_dis':>14s}  {'MS-ECM SE':>10s}  {'TVECM equiv':>14s}")
    try:
        p3 = pd.read_csv(RESULTS_DIR / "phase3_lagged_dsa_coefs.csv")
        for j in range(N_REGIMES):
            c = betas[j, dis_idx]
            se = beta_ses[j, dis_idx]
            p3_row = p3[(p3["regime"] == j+1) &
                        (p3["variable"] == "battery_discharging_mw")]
            p3_c = float(p3_row["coef"].iloc[0]) if len(p3_row) else np.nan
            add(f"  State {j+1}:     {c:>+10.4f}    {se:>10.4f}    {p3_c:>+10.4f}")
    except Exception:
        for j in range(N_REGIMES):
            add(f"  State {j+1}:     {betas[j, dis_idx]:>+10.4f}    "
                f"{beta_ses[j, dis_idx]:>10.4f}")
    add("")

    add("INTERPRETATION")
    add("-" * 65)
    add("The MS-ECM allows regimes to be determined endogenously by the data")
    add("rather than imposed via fixed ECT thresholds (TVECM).")
    add("  1. Regime boundaries are probabilistic, not hard cutoffs")
    add("  2. Transition dynamics captured by the Markov chain")
    add("  3. Variance switching is explicitly modelled per state")
    add("")
    add(f"Battery discharge in high-σ state (≈ TVECM R3):")
    add(f"  β_dis = {c_r3:+.4f}  SE = {se_r3:.4f}  p = {p_r3:.4f}")
    if c_r3 < 0:
        add("  → Negative: battery discharge dampens prices in spike regime")
    else:
        add("  → Positive: strategic discharge / market power effect")

    summary_text = "\n".join(lines)
    (RESULTS_DIR / "phase11_ms_summary.txt").write_text(summary_text)
    print(f"  Saved: results/phase11_ms_summary.txt")

    # ── Final ──────────────────────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  PHASE 11 COMPLETE — 5 output files")
    print(DIVIDER)
    print(f"""
  Key results:
    σ by state:  S1={sigmas[0]:.2f}  S2={sigmas[1]:.2f}  S3={sigmas[2]:.2f}
    β_dis (S3):  {c_r3:+.4f}  (SE={se_r3:.4f}, p={p_r3:.4f})
    Concordance: {overall_conc:.1%}
    Log-lik:     {log_lik:.2f}   AIC: {aic:.2f}   BIC: {bic:.2f}

  Output files:
    results/phase11_ms_coefs.csv
    results/phase11_ms_transition.csv
    results/phase11_ms_probs.csv
    results/phase11_ms_concordance.csv
    results/phase11_ms_summary.txt
""")


if __name__ == "__main__":
    main()
