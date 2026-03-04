"""
generate_data.py
Generates synthetic but realistic agricultural weather + NDVI datasets
mimicking NOAA climate normals and NASA MODIS vegetation index outputs.
Run once: python data/generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os

np.random.seed(42)

# ── 1. Agricultural regions (US + a few international) ───────────────────────
REGIONS = [
    {"id": "CA-SJV", "name": "California Central Valley", "lat": 36.7, "lon": -119.7, "primary_crop": "Almonds/Tomatoes",   "climate": "semi-arid"},
    {"id": "IA-CRN", "name": "Iowa Corn Belt",             "lat": 42.0, "lon": -93.5,  "primary_crop": "Corn/Soybeans",      "climate": "continental"},
    {"id": "TX-HPL", "name": "Texas High Plains",          "lat": 33.5, "lon": -101.9, "primary_crop": "Cotton/Wheat",       "climate": "semi-arid"},
    {"id": "FL-EGL", "name": "Florida Everglades AG",      "lat": 26.6, "lon": -80.7,  "primary_crop": "Sugarcane/Rice",     "climate": "subtropical"},
    {"id": "WA-YKM", "name": "Washington Yakima Valley",   "lat": 46.6, "lon": -120.5, "primary_crop": "Apples/Hops",        "climate": "semi-arid"},
    {"id": "KS-GBL", "name": "Kansas Great Plains",        "lat": 38.7, "lon": -98.3,  "primary_crop": "Winter Wheat",       "climate": "continental"},
    {"id": "GA-PDP", "name": "Georgia Piedmont",           "lat": 33.1, "lon": -83.6,  "primary_crop": "Peanuts/Peaches",    "climate": "humid subtropical"},
    {"id": "ND-RED", "name": "North Dakota Red River",     "lat": 47.2, "lon": -97.1,  "primary_crop": "Wheat/Sunflower",    "climate": "continental"},
    {"id": "CO-SLV", "name": "Colorado San Luis Valley",   "lat": 37.5, "lon": -106.0, "primary_crop": "Potatoes/Barley",    "climate": "high desert"},
    {"id": "MS-DEL", "name": "Mississippi Delta",          "lat": 33.4, "lon": -90.9,  "primary_crop": "Cotton/Soybeans",    "climate": "humid subtropical"},
]

# ── 2. Climate parameters per region type ────────────────────────────────────
CLIMATE_PARAMS = {
    "semi-arid":          {"temp_mean": 18, "temp_range": 15, "precip_annual": 280, "ndvi_base": 0.38},
    "continental":        {"temp_mean": 10, "temp_range": 25, "precip_annual": 750, "ndvi_base": 0.62},
    "subtropical":        {"temp_mean": 24, "temp_range": 10, "precip_annual": 1400, "ndvi_base": 0.70},
    "humid subtropical":  {"temp_mean": 18, "temp_range": 18, "precip_annual": 1200, "ndvi_base": 0.65},
    "high desert":        {"temp_mean": 8,  "temp_range": 20, "precip_annual": 200,  "ndvi_base": 0.30},
}

# ── 3. Generate daily weather time series (2 years) ──────────────────────────
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 12, 31)
DAYS       = (END_DATE - START_DATE).days + 1
dates      = [START_DATE + timedelta(days=i) for i in range(DAYS)]
doy        = np.array([d.timetuple().tm_yday for d in dates])   # day of year

records = []
for region in REGIONS:
    p = CLIMATE_PARAMS[region["climate"]]

    # Temperature: sinusoidal seasonal + noise
    temp_seasonal = p["temp_mean"] + (p["temp_range"] / 2) * np.sin(2 * np.pi * (doy - 80) / 365)
    temp_daily    = temp_seasonal + np.random.normal(0, 3, DAYS)

    # Precipitation: seasonal gamma-distributed daily amounts, some dry spells
    precip_seasonal = (p["precip_annual"] / 365) * (1 + 0.6 * np.sin(2 * np.pi * (doy - 120) / 365))
    rain_prob       = np.clip(precip_seasonal / 8, 0.05, 0.65)
    precip_daily    = np.where(
        np.random.random(DAYS) < rain_prob,
        np.random.gamma(2, precip_seasonal / rain_prob / 2, DAYS),
        0
    )

    # Inject a drought event in summer 2024 for some regions
    if region["climate"] in ["semi-arid", "high desert", "continental"]:
        drought_mask = (doy >= 150) & (doy <= 220) & (np.array([d.year for d in dates]) == 2024)
        precip_daily[drought_mask] *= 0.15
        temp_daily[drought_mask]   += np.random.uniform(2, 5)

    # Inject a flood event for subtropical regions
    if region["climate"] in ["subtropical", "humid subtropical"]:
        flood_day = np.random.randint(200, 280)
        flood_mask = (doy >= flood_day) & (doy <= flood_day + 5)
        precip_daily[flood_mask] += np.random.uniform(60, 120)

    # NDVI: peaks with vegetation season, depressed by drought
    ndvi_seasonal = p["ndvi_base"] + 0.15 * np.sin(2 * np.pi * (doy - 120) / 365)
    ndvi_noise    = np.random.normal(0, 0.02, DAYS)
    ndvi = ndvi_seasonal + ndvi_noise

    # Depress NDVI during drought window
    if region["climate"] in ["semi-arid", "high desert", "continental"]:
        ndvi[drought_mask] -= 0.10

    ndvi = np.clip(ndvi, 0, 1)

    # 30-day rolling precipitation (for drought index)
    precip_series = pd.Series(precip_daily)
    precip_30d    = precip_series.rolling(30, min_periods=1).sum().values
    precip_7d     = precip_series.rolling(7,  min_periods=1).sum().values

    for i, d in enumerate(dates):
        records.append({
            "date":           d.strftime("%Y-%m-%d"),
            "region_id":      region["id"],
            "region_name":    region["name"],
            "lat":            region["lat"],
            "lon":            region["lon"],
            "primary_crop":   region["primary_crop"],
            "climate_zone":   region["climate"],
            "temp_c":         round(float(temp_daily[i]), 2),
            "precip_mm":      round(max(0, float(precip_daily[i])), 2),
            "precip_30d_mm":  round(float(precip_30d[i]), 2),
            "precip_7d_mm":   round(float(precip_7d[i]), 2),
            "ndvi":           round(float(ndvi[i]), 4),
        })

df = pd.DataFrame(records)

# ── 4. Compute monthly aggregates ────────────────────────────────────────────
df["date"]  = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M")

monthly = df.groupby(["region_id", "region_name", "lat", "lon",
                       "primary_crop", "climate_zone", "month"]).agg(
    temp_mean_c   =("temp_c",       "mean"),
    temp_min_c    =("temp_c",       "min"),
    temp_max_c    =("temp_c",       "max"),
    precip_total  =("precip_mm",    "sum"),
    precip_30d_avg=("precip_30d_mm","mean"),
    ndvi_mean     =("ndvi",         "mean"),
    ndvi_min      =("ndvi",         "min"),
).reset_index()

monthly["month"] = monthly["month"].astype(str)

# ── 5. Save ───────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
df.to_csv(f"{out_dir}/daily_weather_ndvi.csv",   index=False)
monthly.to_csv(f"{out_dir}/monthly_aggregates.csv", index=False)

with open(f"{out_dir}/regions.json", "w") as f:
    json.dump(REGIONS, f, indent=2)

print(f"✅ Generated {len(df):,} daily records across {len(REGIONS)} regions")
print(f"✅ Generated {len(monthly):,} monthly aggregate records")
print(f"✅ Files saved to {out_dir}/")
