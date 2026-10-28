# Battery Storage and Electricity Price Convergence

### A Threshold Vector Error-Correction Analysis of the South Australian–Victorian Market

> Empirical study of battery energy storage and wholesale electricity price dynamics in Australia's National Electricity Market, 2023–2025.

---

## Overview

South Australia (SA) has one of the highest renewable shares of any grid: wind and solar averaged 90% of operational demand over 2023–2025, and 27.5% of half-hourly prices were negative. The right tail is extreme as well: 0.4% of intervals exceed \$1,000/MWh, with a maximum of \$16,970/MWh.

Against this backdrop, SA's battery energy storage system (BESS) fleet expanded **4.7-fold** (271 → 1,282 MWh) in three years. This project asks a single causal question:

> **Does a growing BESS fleet exert a measurable causal dampening effect on wholesale electricity prices, and does this effect vary across price regimes?**

The main difficulty is simultaneity: batteries discharge because prices are already high, so OLS gives a positive coefficient on battery discharge in the spike regime (+1.20\*\*\*). Lagged regressors, the Markov-switching ECM and the jump-diffusion drift term yield negative discharge coefficients; 2SLS and GARCH-M estimates remain positive. The causal evidence is therefore mixed. It is strongest for spike prevention (fewer spike-regime intervals as the fleet grew) and weak for batteries lowering a price once a spike has started.

---

## Headline Results

| Metric | Value |
|---|---|
| Sample | 52,608 half-hourly observations, 1 Jan 2023 → 31 Dec 2025 |
| Cointegrating vector (Johansen) | `ECT = p_SA − 1.2103 × p_VIC` |
| Threshold band (Hansen–Seo, *p* = 0.030) | γ_L = −\$45.80, γ_U = +\$20.02 /MWh |
| Spike-regime SA half-life | **0.8 h** (α = −0.349\*\*\*) |
| OLS bias (β_dis, spike regime) | **+1.20\*\*\*** → −0.19 (n.s., *p* = 0.57) under lagged regressors |
| First-stage partial F (SOC instrument) | 30 – 200 across regimes (all strong) |
| Sargan over-id (SOC + Z\_newcap) | **passes in all three regimes** |
| Consumer welfare gain (counterfactual) | **\$10.83 M / year** |
| Decline in OLS simultaneity bias by 2025 | **94 %** (capacity-interaction) |
| SA–VIC price correlation | 0.38 (2023) → **0.80** (2025) |
| Spike-regime frequency | 9.4 % (2023) → **5.0 %** (2025; 5 % is the minimum regime size of the grid search) |

---

## Key Visuals

### SA fleet capacity growth (step function)
![Capacity growth](V2/results/fig_capacity_growth.png)

### SA wholesale price with TVECM regime episodes
![Price regimes](V2/results/fig_sa_price_regimes.png)

### Intraday battery dispatch vs SA price (duck-curve)
![Intraday pattern](V2/results/fig_intraday_pattern.png)

### Rolling 90-day SA–VIC price correlation
![Rolling correlation](V2/results/fig_rolling_correlation.png)

### Distribution of the error-correction term with estimated thresholds
![ECT distribution](V2/results/fig_ect_distribution.png)

### Bivariate impulse-response: 1 MW discharge shock by regime
![IRF](V2/results/phase8_irf.png)

---

## Methodology — what is used and why

The project applies a **chained sequence of econometric techniques**, where each method either tests an assumption of the next, or provides a robustness check on the previous one.

### 1. Pre-estimation diagnostics

- **Unit root tests** — ADF, Phillips–Perron, KPSS (Kwiatkowski et al. 1992). KPSS is preferred over ADF/PP because spike outliers bias ADF-type tests towards rejection (Escribano et al. 2011; Weron 2006). Both price series are treated as I(1).
- **Johansen cointegration** (Johansen 1988) — identifies one cointegrating vector with normalised coefficient β = 1.2103. ECT is confirmed stationary via ADF (stat −38.96, *p* < 0.001).
- **Hansen–Seo two-threshold test** (Hansen & Seo 2002) — Sup-F test with Rademacher wild bootstrap selects a three-regime specification over a single-threshold and a symmetric-band alternative (*p* = 0.030; lowest AIC/BIC).

### 2. Core model — Three-regime TVECM

For each regime *r* ∈ {1, 2, 3}, OLS with **Newey–West HAC** standard errors (Newey & West 1987, 12 lags):

```
Δp_SA_t = μ_r + α_r · ECT_{t-1}
        + Σᵢ₌₁⁴ φᵢʳ · Δp_SA_{t-i}  + Σᵢ₌₁⁴ ψᵢʳ · Δp_VIC_{t-i}
        + β_dis_r · B_dis_t  + β_chg_r · B_chg_t          ← battery
        + γ'ʳ · x_t                                        ← controls
        + ε_t
```

Controls `x_t`: wind, solar, demand, interconnector flow, calendar dummies (holidays, weekends, time-of-day buckets). Same equation estimated for `Δp_VIC_t`.

Regime membership is determined by lagged ECT relative to (γ_L, γ_U): Regime 1 = SA deep discount, Regime 2 = no-arbitrage band, Regime 3 = SA price spike.

### 3. Identification — three strategies for the causal battery effect

| # | Strategy | What it removes / adds |
|---|---|---|
| 1 | **Predetermined (lagged) regressors** | Cuts the contemporaneous simultaneity channel. β_dis in the spike regime moves from +1.20\*\*\* to −0.19 (n.s., *p* = 0.57). |
| 2 | **2SLS / Instrumental variables** | Primary instrument: capacity-normalised 12-h state-of-charge, `Z_SOC = Σ(B_chg − B_dis)·0.5h / K_{t-1}`. Validating instrument: `Z_newcap`, MWh commissioned in 14-day window ending at *t*−1 (exogenous because construction timelines are pre-determined years in advance by project finance). First-stage F = 30–200, Sargan over-id passes in all three regimes. The 2SLS β_dis stays positive in all three regimes (+1.77\*\*\*, +0.43\*\*\*, +0.78\*\*). |
| 3 | **Static counterfactual** | Constructs a "no-battery" price path: `sa_CF_t = sa_actual − Σ_{s≤t} β_r(s)·B_{s}`. Welfare gain ≈ **\$10.83 M / year** of consumer surplus. |

**Wu–Hausman** (Hausman 1978) endogeneity test confirms endogeneity in R1 and R2 (*p* < 0.001) but not R3 (*p* = 0.731). For R3 the lagged-regressor estimator is therefore the preferred causal point estimate; IV remains the headline strategy for R1/R2.

### 4. Robustness layer

| Phase | Method | What it tests | Result |
|---|---|---|---|
| 5 | **Lag-order sensitivity** (L ∈ {4, 6, 8, 10, 12}) | Whether truncation explains residual autocorrelation in outer regimes | β_dis stable within 0.04 across all L. AR is volatility-clustering, not lag-misspecification. |
| 6 | **Rolling thresholds** (annual / semi-annual) | Time-varying band as fleet grows | R3 frequency falls 9.4% → 5.0%. The 2025 value sits on the 5 % minimum-regime floor of the grid search, so the size of the decline is not well identified; the band widening is partly an artefact of the same constraint. |
| 7 | **TVECM-GARCH(1,1)-*t* in-mean** (Engle 1982; Bollerslev 1986) | Whether OLS is contaminated by IGARCH dynamics or risk premia | β_dis change < 0.03 (R3 stays positive, +1.23\*\*\*); IGARCH in R2/R3; standardised-residual AR resolved. |
| 8 | **Bivariate impulse-response** (Rademacher wild bootstrap CI) | Persistence of a 1 MW discharge shock | R3 shock decays to +0.03 \$/MWh within 6 h (half-life ≈ 2 periods, 1 h). |
| 9 | **Battery × capacity interaction** | Whether the effect strengthened as the fleet grew | γ_dis significant (R2: *p* = 0.001; R3: *p* = 0.001). Spike-regime OLS coefficient falls 94 % (+4.76\*\*\* → +0.29, n.s.) and the R2 coefficient becomes insignificant between Jan 2023 and Dec 2025. |
| 10 | **Lagged-battery R3 robustness** (3 sub-approaches) | Whether the spike-regime causal effect can reach significance | Negative in the pooled specifications, mixed signs in the year-by-year estimates; none significant (R3 residual SD ≈ \$589/MWh). |
| 11 | **Markov-switching ECM** (Hamilton 1989) | Whether volatility-based latent regimes confirm the TVECM finding | β_dis negative in all three states (S1, S2 at *p* < 0.001 covering 98.5 % of obs). The high-volatility state gives α = −0.352\*\*\*, close to the TVECM spike-regime estimate of −0.349\*\*\*, although the two models classify regimes by different criteria (ECT level vs conditional variance; concordance 32 %). Custom Hamilton filter + EM, because statsmodels' MarkovRegression was incompatible with NumPy 2.4 at the time of writing. |
| 12 | **Merton-type jump-diffusion** (Merton 1976; Weron 2006) | Whether spikes are best modelled as a Poisson jump process, and whether batteries suppress arrival rate | β_dis = −0.054\*\*\* in the drift. λ_dis = +0.005\*\*\*: lagged discharge predicts a higher jump probability, consistent with operators positioning ahead of expected scarcity. AIC improves by 213,849 over OLS. |

### 5. Why this combination?

The hypothesis is regime- and timing-dependent: batteries are unlikely to lower a price already set by gas peakers (merit-order logic), but discharge ahead of expected scarcity may prevent a spike from forming. Testing this needs:

1. A **regime-aware** core model → TVECM with Hansen–Seo thresholds.
2. **Endogeneity-corrected** estimates → lagged regressors + IV with a credible instrument set.
3. A **welfare quantification** → static counterfactual against the estimated coefficients.
4. **Direct evidence of the timing channel** → jump-diffusion model, which separates drift (causal) from arrival rate (strategic anticipation).
5. **Cross-validation of the regime classification** → Markov-switching ECM, which lets the data choose regimes endogenously.

---

## Implications for trading and investment

These are hypotheses suggested by the estimates, not tested strategies. Nothing here has been backtested, and the ECT is computed from settled prices, which are not known at the time a trade would be placed.

**SA–VIC spread.** Outside the no-arbitrage band the spread `p_SA − 1.21 × p_VIC` mean-reverts quickly, and the adjusting market differs by regime:

| Regime | Condition | Adjusting market | Half-life |
|---|---|---|---|
| Spike (R3) | ECT > +\$20/MWh | SA falls (α_SA = −0.349\*\*\*) | ≈ 0.8 h |
| Discount (R1) | ECT < −\$46/MWh | VIC falls (α_VIC = +0.525\*\*\*) | ≈ 0.5 h |
| Band (R2) | −\$46 ≤ ECT ≤ +\$20 | no clear signal | — |

**Battery dispatch and spike risk.** In the jump-diffusion model, lagged battery discharge predicts a higher jump probability (λ_dis = +0.005\*\*\*). The mean jump probability is 3.7 % per 30-min interval (about 1.8 per day); 100 MW of lagged discharge raises it by about 0.018. This is consistent with operators positioning ahead of expected scarcity, and suggests dispatch data carries information about near-term spike risk.

**Tail risk.** Volatility in regimes 2 and 3 is close to integrated (α_G + β_G ≈ 1.00), and the Student-*t* degrees of freedom are about 2.9, so Gaussian or mean-reverting GARCH VaR will understate risk. The estimated jump component (μ_J = +\$368/MWh, σ_J = \$1,359/MWh) is in `V2/results/phase12_jd_params.csv` and can be used for a jump-aware VaR.

**Cap pricing.** The spike-regime share fell from 9.4 % (2023) to 7.2 % (2024) and 5.0 % (2025). The 2025 figure sits on the 5 % lower bound of the threshold grid search, so the decline should not be used as a precise decay rate. Caps priced on 2023 realisations would likely be too expensive for 2025 conditions.

**BESS merchant revenue.** Intervals above \$300/MWh fell from 112 in August 2023 to 19 in August 2025, while installed capacity roughly doubled (521 → 1,082 MWh). The SA–VIC price correlation rose from 0.38 to 0.80 over the sample. Both point to compressing spike and inter-regional arbitrage revenue, so revenue models that extrapolate 2023 spreads are likely to overstate returns. In the capacity-interaction model the spike-regime OLS coefficient on contemporaneous discharge fell from +4.76\*\*\* (January 2023) to +0.29 (n.s., December 2025), which is consistent with dispatch becoming less reactive as the fleet grew.

---

## Limitations

Three caveats qualify the findings:

1. **Realised vs forecast renewables.** Wind, solar, and demand enter as realised values, but BESS operators dispatch on day-ahead and 5-minute *forecasts*. Forecast-error noise is therefore folded into the battery-dispatch coefficient and likely biases it towards zero.
2. **BESS modelled as price-arbitrage only.** SA batteries also earn revenue from frequency-regulation and grid-ancillary-services (FCAS) markets. A full welfare evaluation would need to account for these parallel revenue and dispatch channels.
3. **Regime 3 wide uncertainty.** Spike-regime residual volatility is extreme (σ ≈ \$589/MWh in the TVECM, \$1,819/MWh in the MS-ECM high-σ state). Spike-regime point estimates should be read as central tendencies rather than precise magnitudes; for this reason the project reports several estimators rather than relying on a single coefficient.

---

## Repository layout

```
.
├── README.md                                 (this file)
├── V2/
│   ├── data/processed/sa_nem_2023_2025.csv   master dataset (52,608 rows, 18 cols)
│   ├── data/raw/                             not tracked; API/NEMWEB cache re-created by the download scripts
│   ├── download_data.py                      OpenElectricity API v4 puller
│   ├── add_battery_data.py                   adds BESS columns
│   ├── fill_interconnector.py                NEMWEB MMSDM gap-fill
│   ├── phase0_data_analytics.py              distributional + filtered unit root tests
│   ├── phase0b_threshold_dummies.py          Hansen–Seo with calendar dummies
│   ├── phase1_diagnostics.py                 ADF / PP / KPSS + Johansen + single-threshold Hansen
│   ├── phase1_band_threshold.py              symmetric-band + two-threshold Hansen
│   ├── phase2_tvecm.py                       three-regime TVECM (contemporaneous)
│   ├── phase3_counterfactual.py              lagged battery + static counterfactual
│   ├── phase4_iv.py                          2SLS with SOC_norm + Z_newcap
│   ├── phase5_lag_robustness.py              L ∈ {4, 6, 8, 10, 12}
│   ├── phase6_rolling_threshold.py           time-varying thresholds + capacity regression
│   ├── phase7_garch_m.py                     TVECM-GARCH(1,1)-t in-mean
│   ├── phase8_irf.py                         bivariate IRFs with wild bootstrap
│   ├── phase9_cap_interact.py                battery × installed-capacity interaction
│   ├── phase10_lagged_robustness.py          lagged-battery R3 robustness (3 approaches)
│   ├── phase11_markov_switching.py           3-state MS-ECM (custom Hamilton filter + EM)
│   ├── phase12_jump_diffusion.py             Merton-type jump-diffusion (MLE)
│   ├── generate_figures.py                   figure rendering script
│   ├── BESS SA.xlsx                          SA battery fleet schedule (10 facilities)
│   ├── results/                              CSV outputs + PNG figures for every phase
│   ├── paper/sa_nem_battery_tvecm.tex        full LaTeX paper
│   ├── paper/references.bib                  bibliography
│   └── project_outline.md                    one-page abstract
```

---

## Reproducing the analysis

### Environment

```bash
# Python 3.12 (Anaconda recommended)
pip install pandas numpy statsmodels==0.14.2 arch scipy matplotlib requests tqdm openpyxl
```

> **Note:** With NumPy 2.x some scripts print a `numexpr` compatibility warning; it does not affect the results. Importing via `statsmodels.api` raised a SciPy import error in the original environment, so the scripts import statsmodels classes directly (e.g. `from statsmodels.regression.linear_model import OLS`).

### Data acquisition (one-off; cached afterwards)

```bash
export OE_API_KEY="<your OpenElectricity API key>"
python V2/download_data.py        # ~30 min, fully cached
python V2/add_battery_data.py     # ~10 min, fully cached
python V2/fill_interconnector.py  # NEMWEB gap-fill for missing interconnector months
```

The scripts cache the raw API and NEMWEB downloads in `V2/data/raw/` (about 120 MB, not tracked). The resulting master dataset (`V2/data/processed/sa_nem_2023_2025.csv`) is included in the repo, so the analysis phases below can be run without downloading anything.

### Run the phases (suggested order)

```bash
cd V2
python phase0_data_analytics.py      # distributions & filtered unit roots
python phase1_diagnostics.py         # ADF/PP/KPSS, Johansen, single threshold
python phase1_band_threshold.py      # symmetric band + two-threshold Hansen
python phase2_tvecm.py               # baseline TVECM (contemporaneous)
python phase3_counterfactual.py      # lagged battery + welfare counterfactual
python phase4_iv.py                  # 2SLS identification
python phase5_lag_robustness.py      # lag sensitivity
python phase6_rolling_threshold.py   # time-varying thresholds
python phase7_garch_m.py             # GARCH-M extension
python phase8_irf.py                 # impulse-response functions
python phase9_cap_interact.py        # capacity-interaction
python phase10_lagged_robustness.py  # lagged R3 robustness (3 approaches)
python phase11_markov_switching.py   # Markov-switching ECM
python phase12_jump_diffusion.py     # jump-diffusion model
python generate_figures.py           # render all README figures
```

Each script writes CSV / PNG / TXT outputs to `V2/results/`. Most phases run in under two minutes; the Markov-switching ECM (~10 min) and the jump-diffusion model (~5 min) take longest.

### Building the paper

```bash
cd V2/paper
latexmk -pdf sa_nem_battery_tvecm.tex   # biblatex + biber
```

---

## Author

**Tim Louis Wilken** — 2026.

## Citation

If you reference this work, please cite:

> Wilken, T. L. (2026). *Battery Storage and Electricity Price Convergence: A Threshold Vector Error-Correction Analysis of the South Australian–Victorian Market.*

## Data sources & acknowledgements

- **OpenElectricity API v4** — SA1 & VIC1 regional reference prices, generation by fuel-tech, demand. <https://openelectricity.org.au>
- **NEMWEB MMSDM** — V-SA interconnector dispatch (`DISPATCHINTERCONNECTORRES`). <https://nemweb.com.au>
- **BESS SA.xlsx** — SA battery fleet schedule compiled from project announcements and AEMO commissioning notices.

Key references are listed in `V2/paper/references.bib` (Balke & Fomby 1997; Hansen & Seo 2002; Johansen 1988; Newey & West 1987; Hamilton 1989; Merton 1976; Weron 2006; Escribano et al. 2011; AEMO 2024 ISP; Hirth 2013; Mwampashi & Nikitopoulos 2025; Stanciu & Mitu 2025; de Menezes & Houllier 2016; Hauzenberger et al. 2023).
