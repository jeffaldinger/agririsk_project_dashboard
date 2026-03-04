"""
refresh_live_data.py
Run this script to fetch real weather data from Open-Meteo and cache it locally.
Re-run any time you want to update with more recent data.

Usage:
    python data/refresh_live_data.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from engine.weather_fetcher import fetch_all_regions, build_monthly_aggregates

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 55)
print("  AgriRisk — Live Weather Data Refresh")
print("  Source: Open-Meteo Archive API (free, no key)")
print("=" * 55)
print()

# ── 1. Fetch live weather ─────────────────────────────────────────────────────
print("Fetching daily weather for all regions...")
daily_live = fetch_all_regions(verbose=True)
print(f"\n✅ {len(daily_live):,} daily records fetched across {daily_live['region_id'].nunique()} regions")
print(f"   Date range: {daily_live['date'].min().date()} → {daily_live['date'].max().date()}")

# ── 2. Merge with synthetic NDVI (no free satellite source without account) ───
print("\nMerging with synthetic NDVI data...")

# Load synthetic data to extract NDVI column
synthetic_path = os.path.join(OUT_DIR, "daily_weather_ndvi.csv")
if not os.path.exists(synthetic_path):
    print("  WARNING: synthetic data not found, run generate_data.py first")
    print("  NDVI will be estimated from climate zone baselines")
    # Fallback: estimate NDVI from climate zone
    NDVI_BASE = {
        "semi-arid": 0.38, "continental": 0.62, "subtropical": 0.70,
        "humid subtropical": 0.65, "high desert": 0.30,
    }
    import numpy as np
    daily_live["ndvi"] = daily_live["climate_zone"].map(NDVI_BASE).fillna(0.5)
    daily_live["ndvi"] += np.random.normal(0, 0.02, len(daily_live))
    daily_live["ndvi"] = daily_live["ndvi"].clip(0, 1).round(4)
else:
    synthetic = pd.read_csv(synthetic_path, parse_dates=["date"])
    # Match on region_id + day-of-year to get realistic seasonal NDVI pattern
    synthetic["doy"] = synthetic["date"].dt.dayofyear
    daily_live["doy"] = daily_live["date"].dt.dayofyear

    ndvi_lookup = (
        synthetic.groupby(["region_id", "doy"])["ndvi"]
        .mean()
        .reset_index()
    )
    daily_live = daily_live.merge(ndvi_lookup, on=["region_id", "doy"], how="left")
    daily_live["ndvi"] = daily_live["ndvi"].interpolate().clip(0, 1).round(4)
    daily_live.drop(columns=["doy"], inplace=True)
    print("  NDVI merged from synthetic seasonal baseline (NASA MODIS proxy)")

# ── 3. Build monthly aggregates ───────────────────────────────────────────────
print("\nBuilding monthly aggregates...")
monthly_live = build_monthly_aggregates(daily_live)

# Add NDVI monthly stats
daily_live["month"] = daily_live["date"].dt.to_period("M").astype(str)
ndvi_monthly = daily_live.groupby(["region_id", "month"]).agg(
    ndvi_mean=("ndvi", "mean"),
    ndvi_min =("ndvi", "min"),
).reset_index()
monthly_live = monthly_live.merge(ndvi_monthly, on=["region_id", "month"], how="left")

# ── 4. Save ───────────────────────────────────────────────────────────────────
daily_out   = os.path.join(OUT_DIR, "live_daily_weather.csv")
monthly_out = os.path.join(OUT_DIR, "live_monthly_aggregates.csv")

daily_live.to_csv(daily_out,   index=False)
monthly_live.to_csv(monthly_out, index=False)

print(f"\n✅ Saved {len(daily_live):,} daily records   → {daily_out}")
print(f"✅ Saved {len(monthly_live):,} monthly records → {monthly_out}")
print()
print("Run 'streamlit run app.py' and switch to Live Data mode in the sidebar.")
