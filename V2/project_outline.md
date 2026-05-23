# Battery Storage and Electricity Price Dynamics in South Australia's NEM: A Threshold Vector Error Correction Approach

## Research Question

Does battery storage activity (charging/discharging) affect electricity price dynamics in South Australia's National Electricity Market differently depending on the prevailing price regime? Specifically, do batteries dampen price spikes in the high-spread regime and absorb surplus generation in the low-price regime?

## Data

- **Source:** OpenElectricity API (SA1/VIC1 regions) + NEMWEB MMSDM interconnector data + BESS facility registry
- **Sample:** 52,608 half-hourly observations, 1 Jan 2023 -- 31 Dec 2025
- **Variables:** SA and VIC regional reference prices, battery discharge/charge (MW), wind and solar generation, demand, interconnector flow, installed BESS capacity (12 facilities, 285 --> 1,296 MWh)

## Methodology

**Core model:** Three-regime Threshold Vector Error Correction Model (TVECM) for the cointegrated SA--VIC price pair, with regime membership determined by the lagged error-correction term (ECT = P_SA - 1.21 x P_VIC). Thresholds estimated via Hansen's (1997, 2002) Sup-F bootstrap test. Battery dispatch enters as regime-specific regressors.

**Identification strategy (3 approaches):**

1. **Lagged battery regressors** -- replacing contemporaneous with predetermined (t-1) battery variables removes simultaneity bias and reveals causal price-dampening (sign flip from +1.20 to -0.19 in the spike regime)
2. **Instrumental variables (2SLS)** -- capacity-normalised state-of-charge as the primary instrument (partial F = 30--200 across regimes), validated by a commissioning-event IV via Sargan J-test
3. **Static counterfactual** -- constructing a "no-battery" price path to quantify the consumer welfare gain from spike prevention

**Robustness and extensions:**

- Lag sensitivity analysis (L = 4--12): coefficients stable within 0.04 across all specifications
- GARCH(1,1)-t in-mean: captures IGARCH-type volatility clustering; battery coefficients unchanged (<0.03); resolves residual autocorrelation in standardised residuals
- Impulse-response functions: 1 MW discharge shock in the spike regime decays to near-zero within 1 hour (half-life ~ 2 periods) due to strong error correction
- Battery x capacity interaction: discharge coefficient declines 75--84% from 2023 to 2025 as fleet matures from reactive to strategic dispatch
- Markov-Switching ECM (3-state): endogenous volatility-based regimes confirm negative discharge effect (significant at p < 0.001 in 98.5% of observations)
- Jump-diffusion model: separates normal dynamics from Poisson spike events; battery dampens drift (beta = -0.054***) while discharge positively predicts (but does not cause) jump occurrence

## Key Findings

1. **Simultaneity bias in naive models:** Simple OLS regressions show a *positive* coefficient on battery discharge -- suggesting batteries raise prices. This is a textbook simultaneity problem: batteries discharge *because* prices are already spiking, not the other way around. Once endogeneity is addressed (lagged regressors, IV/2SLS, GARCH-M, Markov-switching, jump-diffusion), the sign flips or collapses, revealing the true causal direction.

2. **Batteries prevent spikes rather than reduce ongoing ones:** The evidence consistently shows that batteries do not meaningfully reduce a price spike that is already underway -- once expensive peaking generators are dispatched and setting the marginal price, additional battery output has limited effect, consistent with merit-order logic. Instead, the welfare gain arises from *spike prevention*: strategic battery positioning ahead of anticipated scarcity events displaces the need for costly peakers entirely, preventing spikes from materialising. The jump-diffusion model supports this directly -- lagged discharge predicts lower jump probability through the drift channel, while battery operators position ahead of anticipated jumps (positive lambda_dis). The declining spike-regime frequency (9.4% to 5.0%) as the fleet grew 4.6x further confirms that batteries reduce the *occurrence* of extreme price events rather than their severity once triggered.

3. **Asymmetric price adjustment:** SA corrects alone in the spike regime (alpha = -0.35, half-life 0.8h); VIC corrects alone in the discount regime (alpha = +0.53). The no-arbitrage band spans [-46, +20] $/MWh.

4. **Growing impact:** As SA's BESS fleet expanded 4.6x, the frequency of spike-regime episodes fell from 9.4% to 5.0%, and the SA--VIC price correlation rose from 0.38 to 0.80.