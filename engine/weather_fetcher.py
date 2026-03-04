"""
weather_fetcher.py
Fetches real historical weather data from Open-Meteo's free archive API.
No API key or account required.

Usage:
    from engine.weather_fetcher import fetch_all_regions, REGIONS
    df = fetch_all_regions()
"""

import time
import requests
import pandas as pd
from datetime import date, timedelta

# ── Region definitions (lat/lon match the synthetic data) ────────────────────
REGIONS = [
    {"id": "CA-SJV", "name": "California Central Valley", "lat": 36.7,  "lon": -119.7, "primary_crop": "Almonds/Tomatoes",  "climate": "semi-arid",          "timezone": "America/Los_Angeles"},
    {"id": "IA-CRN", "name": "Iowa Corn Belt",            "lat": 42.0,  "lon": -93.5,  "primary_crop": "Corn/Soybeans",     "climate": "continental",         "timezone": "America/Chicago"},
    {"id": "TX-HPL", "name": "Texas High Plains",         "lat": 33.5,  "lon": -101.9, "primary_crop": "Cotton/Wheat",      "climate": "semi-arid",           "timezone": "America/Chicago"},
    {"id": "FL-EGL", "name": "Florida Everglades AG",     "lat": 26.6,  "lon": -80.7,  "primary_crop": "Sugarcane/Rice",    "climate": "subtropical",         "timezone": "America/New_York"},
    {"id": "WA-YKM", "name": "Washington Yakima Valley",  "lat": 46.6,  "lon": -120.5, "primary_crop": "Apples/Hops",       "climate": "semi-arid",           "timezone": "America/Los_Angeles"},
    {"id": "KS-GBL", "name": "Kansas Great Plains",       "lat": 38.7,  "lon": -98.3,  "primary_crop": "Winter Wheat",      "climate": "continental",         "timezone": "America/Chicago"},
    {"id": "GA-PDP", "name": "Georgia Piedmont",          "lat": 33.1,  "lon": -83.6,  "primary_crop": "Peanuts/Peaches",   "climate": "humid subtropical",   "timezone": "America/New_York"},
    {"id": "ND-RED", "name": "North Dakota Red River",    "lat": 47.2,  "lon": -97.1,  "primary_crop": "Wheat/Sunflower",   "climate": "continental",         "timezone": "America/Chicago"},
    {"id": "CO-SLV", "name": "Colorado San Luis Valley",  "lat": 37.5,  "lon": -106.0, "primary_crop": "Potatoes/Barley",   "climate": "high desert",         "timezone": "America/Denver"},
    {"id": "MS-DEL", "name": "Mississippi Delta",         "lat": 33.4,  "lon": -90.9,  "primary_crop": "Cotton/Soybeans",   "climate": "humid subtropical",   "timezone": "America/Chicago"},
]

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
START_DATE   = "2023-01-01"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
]


def fetch_region(region: dict, start: str = START_DATE, end: str = None) -> pd.DataFrame:
    """
    Fetches daily weather for a single region from Open-Meteo archive API.
    Returns a DataFrame with columns matching the synthetic data schema.
    """
    if end is None:
        # Go up to yesterday (archive has ~5 day lag)
        end = str(date.today() - timedelta(days=5))

    params = {
        "latitude":  region["lat"],
        "longitude": region["lon"],
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(DAILY_VARS),
        "timezone":   region["timezone"],
    }

    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    df = pd.DataFrame({
        "date":       pd.to_datetime(daily["time"]),
        "temp_c":     daily["temperature_2m_mean"],
        "temp_max_c": daily["temperature_2m_max"],
        "temp_min_c": daily["temperature_2m_min"],
        "precip_mm":  daily["precipitation_sum"],
    })

    # Fill any missing values with interpolation
    df["temp_c"]     = df["temp_c"].interpolate()
    df["temp_max_c"] = df["temp_max_c"].interpolate()
    df["temp_min_c"] = df["temp_min_c"].interpolate()
    df["precip_mm"]  = df["precip_mm"].fillna(0)

    # Add region metadata
    df["region_id"]   = region["id"]
    df["region_name"] = region["name"]
    df["lat"]         = region["lat"]
    df["lon"]         = region["lon"]
    df["primary_crop"]= region["primary_crop"]
    df["climate_zone"]= region["climate"]

    # Rolling precipitation windows (used by risk engine)
    df["precip_30d_mm"] = df["precip_mm"].rolling(30, min_periods=1).sum()
    df["precip_7d_mm"]  = df["precip_mm"].rolling(7,  min_periods=1).sum()

    return df


def fetch_all_regions(
    start: str = START_DATE,
    end: str = None,
    pause: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetches weather for all 10 regions and returns a combined DataFrame.
    pause: seconds to wait between API calls (be polite to the free API)
    """
    frames = []
    for i, region in enumerate(REGIONS):
        if verbose:
            print(f"  [{i+1}/{len(REGIONS)}] Fetching {region['name']}...")
        try:
            df = fetch_region(region, start=start, end=end)
            frames.append(df)
            if verbose:
                print(f"         {len(df)} days retrieved")
        except Exception as e:
            print(f"         WARNING: Failed to fetch {region['name']}: {e}")
        time.sleep(pause)

    if not frames:
        raise RuntimeError("No data retrieved from Open-Meteo. Check your internet connection.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["region_id", "date"]).reset_index(drop=True)
    return combined


def build_monthly_aggregates(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates live daily weather into monthly summaries,
    matching the schema expected by the risk engine.
    """
    df = daily_df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)

    monthly = df.groupby([
        "region_id", "region_name", "lat", "lon",
        "primary_crop", "climate_zone", "month"
    ]).agg(
        temp_mean_c    =("temp_c",       "mean"),
        temp_min_c     =("temp_min_c",   "min"),
        temp_max_c     =("temp_max_c",   "max"),
        precip_total   =("precip_mm",    "sum"),
        precip_30d_avg =("precip_30d_mm","mean"),
    ).reset_index()

    return monthly
