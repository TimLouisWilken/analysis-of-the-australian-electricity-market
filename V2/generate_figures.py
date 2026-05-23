"""Generate additional figures for the paper."""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

DATA = "data/processed/sa_nem_2023_2025.csv"
RESULTS = "results"

df = pd.read_csv(DATA, parse_dates=["settlement_datetime"], index_col="settlement_datetime")

# --- ECT and regime assignment ---
BETA = 1.2103
GAMMA_L, GAMMA_U = -45.80, 20.02

df["ect"] = df["sa_price"] - BETA * df["vic_price"]
df["ect_lag"] = df["ect"].shift(1)
df["regime"] = np.where(df["ect_lag"] < GAMMA_L, 1,
               np.where(df["ect_lag"] > GAMMA_U, 3, 2))

# ============================================================
# FIGURE 1: SA price with regime shading — 3 yearly panels, daily max
# ============================================================
years = [2023, 2024, 2025]
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

for ax, yr in zip(axes, years):
    mask = df.index.year == yr
    sub = df.loc[mask]

    # Daily max shows spikes; daily mean shows baseline
    daily_max = sub["sa_price"].resample("D").max()
    daily_mean = sub["sa_price"].resample("D").mean()

    ax.fill_between(daily_max.index, daily_mean.values, daily_max.values,
                    color="grey", alpha=0.25, label="Daily range (mean to max)")
    ax.plot(daily_mean.index, daily_mean.values, color="black", linewidth=0.7,
            label="Daily mean")

    # Shade days with any Regime 3 observations (ECT-based spike regime)
    r3_daily = sub["regime"].eq(3).resample("D").max().astype(bool)
    for idx, is_r3 in r3_daily.items():
        if is_r3:
            ax.axvspan(idx, idx + pd.Timedelta(days=1),
                       alpha=0.20, color="red", linewidth=0)

    # Also shade Regime 1 days (SA deep discount)
    r1_daily = sub["regime"].eq(1).resample("D").max().astype(bool)
    for idx, is_r1 in r1_daily.items():
        if is_r1:
            ax.axvspan(idx, idx + pd.Timedelta(days=1),
                       alpha=0.15, color="blue", linewidth=0)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("$/MWh")
    ax.set_title(f"{yr}", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # Year-specific y-limits to show spikes
    ymax = min(daily_max.max() * 1.1, 2000)
    ymin = max(daily_mean.min() * 1.1, -300)
    ax.set_ylim(ymin, ymax)

axes[0].legend(loc="upper right", fontsize=9)
# Add regime legend manually
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor="red", alpha=0.20, label="Regime 3 (SA spike, ECT > +20)"),
    Patch(facecolor="blue", alpha=0.15, label="Regime 1 (SA discount, ECT < −46)"),
]
axes[2].legend(handles=legend_patches, loc="upper right", fontsize=9)

fig.suptitle("SA wholesale electricity price by year with TVECM regime episodes",
             fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{RESULTS}/fig_sa_price_regimes.png")
plt.close()
print("  Saved fig_sa_price_regimes.png")

# ============================================================
# FIGURE 2: Battery capacity growth step function
# ============================================================
capacity_schedule = [
    ("2017-10-01", 129.0, "Hornsdale"),
    ("2018-12-01", 8.0, "Dalrymple"),
    ("2019-12-01", 52.0, "Lake Bonney"),
    ("2020-09-01", 64.5, "Hornsdale Ext."),
    ("2021-05-01", 13.0, "Adelaide Desal."),
    ("2022-03-01", 4.3, "Christies Beach"),
    ("2023-08-01", 250.0, "Torrens Island"),
    ("2025-01-01", 84.0, "Tailem Bend 2"),
    ("2025-04-01", 477.0, "Blyth"),
    ("2025-10-01", 200.0, "Mannum"),
]

dates_cap = pd.date_range("2023-01-01", "2025-12-31", freq="D")
cap_series = pd.Series(0.0, index=dates_cap)
cumcap = 0.0
for d, mwh, name in capacity_schedule:
    dt = pd.Timestamp(d)
    cumcap += mwh
    cap_series.loc[cap_series.index >= dt] = cumcap

# Only show in-sample period
cap_sample = cap_series["2023-01-01":"2025-12-31"]

fig, ax = plt.subplots(figsize=(10, 4))
ax.step(cap_sample.index, cap_sample.values, where="post", color="steelblue", linewidth=2)

# Annotate in-sample commissioning events
for d, mwh, name in capacity_schedule:
    dt = pd.Timestamp(d)
    if dt >= pd.Timestamp("2023-01-01"):
        val = cap_series.loc[cap_series.index >= dt].iloc[0]
        ax.annotate(f"{name}\n+{mwh:.0f} MWh",
                    xy=(dt, val), xytext=(0, 15),
                    textcoords="offset points", fontsize=8,
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

ax.set_ylabel("Installed BESS capacity (MWh)")
ax.set_xlabel("")
ax.set_title("SA battery fleet capacity growth (2023--2025)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
fig.autofmt_xdate(rotation=30)
fig.savefig(f"{RESULTS}/fig_capacity_growth.png")
plt.close()
print("  Saved fig_capacity_growth.png")

# ============================================================
# FIGURE 3: Intraday battery dispatch and price pattern
# ============================================================
df["hour"] = df.index.hour + df.index.minute / 60

hourly_price = df.groupby(df.index.hour)["sa_price"].mean()
hourly_net = df.groupby(df.index.hour)["battery_net_mw"].mean()

fig, ax1 = plt.subplots(figsize=(10, 4.5))
color_price = "black"
color_bat = "steelblue"

ax1.bar(hourly_net.index, hourly_net.values, color=color_bat, alpha=0.6, label="Battery net MW (+ = discharge)")
ax1.set_ylabel("Battery net MW", color=color_bat)
ax1.set_xlabel("Hour of day")
ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")

ax2 = ax1.twinx()
ax2.plot(hourly_price.index, hourly_price.values, color=color_price, linewidth=2, marker="o", markersize=4, label="Mean SA price")
ax2.set_ylabel("SA price ($/MWh)", color=color_price)

ax1.set_title("Intraday battery dispatch and SA price pattern (sample average)")
ax1.set_xticks(range(0, 24, 2))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

fig.savefig(f"{RESULTS}/fig_intraday_pattern.png")
plt.close()
print("  Saved fig_intraday_pattern.png")

# ============================================================
# FIGURE 4: Rolling SA-VIC correlation (90-day window)
# ============================================================
rolling_corr = df["sa_price"].rolling(48*90, min_periods=48*30).corr(df["vic_price"])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rolling_corr.index, rolling_corr.values, color="steelblue", linewidth=1)
ax.set_ylabel("Pearson correlation")
ax.set_xlabel("")
ax.set_title("Rolling 90-day SA--VIC price correlation")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")
fig.autofmt_xdate(rotation=30)
fig.savefig(f"{RESULTS}/fig_rolling_correlation.png")
plt.close()
print("  Saved fig_rolling_correlation.png")

# ============================================================
# FIGURE 5: ECT distribution with threshold lines
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ect_clean = df["ect"].dropna()
# Clip for visualisation
ect_clip = ect_clean.clip(-300, 300)
ax.hist(ect_clip, bins=200, density=True, color="grey", alpha=0.7, edgecolor="none")
ax.axvline(GAMMA_L, color="red", linewidth=2, linestyle="--", label=f"$\\gamma_L = {GAMMA_L}$")
ax.axvline(GAMMA_U, color="red", linewidth=2, linestyle="--", label=f"$\\gamma_U = {GAMMA_U}$")
ax.set_xlabel("ECT = $p^{SA}$ − 1.21 × $p^{VIC}$ ($/MWh)")
ax.set_ylabel("Density")
ax.set_title("Distribution of the error-correction term with estimated thresholds")
ax.legend(fontsize=10)
fig.savefig(f"{RESULTS}/fig_ect_distribution.png")
plt.close()
print("  Saved fig_ect_distribution.png")

print("\nAll figures generated.")
