# Battery Storage and Electricity Price Dynamics in South Australia's NEM: A Threshold Vector Error Correction Approach

## Research Question

Does a growing BESS fleet have a measurable causal dampening effect on wholesale electricity prices in South Australia, and does this effect differ across price regimes?

## Data

- **Source:** OpenElectricity API (SA1/VIC1 regions), NEMWEB MMSDM interconnector data, SA BESS fleet schedule (`BESS SA.xlsx`)
- **Sample:** 52,608 half-hourly observations, 1 Jan 2023 to 31 Dec 2025
- **Variables:** SA and VIC regional reference prices, battery discharge/charge (MW), wind and solar generation, demand, interconnector flow, installed BESS capacity (10 facilities, 271 to 1,282 MWh)

## Methodology

**Core model:** Three-regime threshold vector error correction model (TVECM) for the cointegrated SA–VIC price pair. Regime membership is set by the lagged error-correction term (ECT = P_SA − 1.21 × P_VIC). Thresholds are estimated with the Hansen–Seo Sup-F test (Rademacher wild bootstrap). Battery dispatch enters as regime-specific regressors.

**Identification (3 approaches):**

1. **Lagged battery regressors:** predetermined (t−1) battery variables remove the contemporaneous simultaneity channel. The spike-regime coefficient moves from +1.20\*\*\* to −0.19 (not significant).
2. **Instrumental variables (2SLS):** capacity-normalised state of charge as the primary instrument (partial F = 30–200 across regimes), with a commissioning-event instrument for the Sargan test. The 2SLS discharge coefficients remain positive in all three regimes.
3. **Static counterfactual:** a "no-battery" price path used to estimate the consumer welfare effect (about \$10.8 M per year).

**Robustness and extensions:**

- Lag sensitivity (L = 4–12): coefficients stable within 0.04
- GARCH(1,1)-t in-mean: near-integrated volatility in regimes 2 and 3; battery coefficients change by less than 0.03
- Impulse responses: a 1 MW discharge shock in the spike regime decays to +0.03 \$/MWh within 6 hours
- Battery × capacity interaction: the spike-regime OLS coefficient falls 94 % between January 2023 and December 2025
- Markov-switching ECM (3 states): negative discharge coefficients in all states, significant in the two low-volatility states
- Jump-diffusion model: negative discharge coefficient in the drift (−0.054\*\*\*); lagged discharge predicts a higher jump probability (λ_dis > 0)

## Key Findings

1. **Simultaneity bias:** OLS gives a positive discharge coefficient because batteries discharge when prices are already high. Lagged regressors, the Markov-switching ECM and the jump-diffusion drift term give negative coefficients; 2SLS and GARCH-M do not. The causal evidence is mixed.
2. **Spike prevention rather than spike reduction:** There is little evidence that discharge lowers a price once a spike has started. The spike-regime share fell from 9.4 % (2023) to 5.0 % (2025) as the fleet grew 4.7×; the 2025 value sits on the 5 % lower bound of the threshold grid search.
3. **Asymmetric price adjustment:** SA adjusts in the spike regime (α = −0.35, half-life 0.8 h); VIC adjusts in the discount regime (α = +0.53). The no-arbitrage band spans [−46, +20] \$/MWh.
4. **Market integration:** The SA–VIC price correlation rose from 0.38 (2023) to 0.80 (2025).
