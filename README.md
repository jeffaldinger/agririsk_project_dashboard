# 🌾 AgriRisk Field Intelligence Dashboard

A local Streamlit dashboard that monitors **drought, frost, and flood risk** across US agricultural regions using synthetic weather and vegetation (NDVI) data.

Built as a portfolio demonstration of agricultural data analysis, risk modeling, and interactive visualization.

---

## Features

| Module | What it does |
|---|---|
| **Data generation** | Synthesizes 2 years of realistic daily weather + NDVI time series for 10 US ag regions, mimicking NOAA ISD and NASA MODIS outputs |
| **Risk engine** | Computes 0–100 risk scores for drought (SPI proxy + NDVI anomaly + heat), frost (growing season temp floor), and flood (precip intensity + 7-day spike) |
| **Dashboard UI** | Interactive Plotly map, region-level time-series trends, risk heatmap (region × month), and active alert panel |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset (run once)
python data/generate_data.py

# 3. Launch dashboard
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
agri_risk_dashboard/
├── app.py                        # Streamlit dashboard (main entry point)
├── requirements.txt
├── data/
│   ├── generate_data.py          # Synthetic NOAA/MODIS-style data generator
│   ├── daily_weather_ndvi.csv    # 7,310 daily records × 10 regions (generated)
│   ├── monthly_aggregates.csv    # 240 monthly summaries (generated)
│   └── regions.json              # Region metadata
└── engine/
    └── risk_engine.py            # Risk scoring logic (drought/frost/flood)
```

---

## Risk Methodology

### Drought Score (0–100)
Weighted combination of:
- **Precipitation deficit** (SPI proxy): monthly total vs. long-term monthly mean — up to 40 pts
- **NDVI depression**: vegetation index below climatological baseline — up to 40 pts
- **Temperature excess**: heat anomaly amplifying water stress — up to 20 pts

### Frost Score (0–100)
- Growing season (April–October): penalises minimum temps below 5°C at 10 pts/°C
- Dormant season: penalises minimum temps below 0°C at 5 pts/°C

### Flood Score (0–100)
- Monthly precipitation intensity relative to long-term mean (flood_intensity) — up to 60 pts
- 7-day extreme precipitation spike bonus (flash flood proxy) — up to 40 pts
- Downweighted 30% in dry climates (semi-arid, high desert)

### Composite Score
`Composite = 0.45 × Drought + 0.25 × Frost + 0.30 × Flood`

| Score | Label |
|---|---|
| 0–25 | 🟢 Low |
| 26–50 | 🟡 Moderate |
| 51–75 | 🟠 High |
| 76–100 | 🔴 Critical |

---

## Data Sources (Synthetic)

The dataset is synthetically generated to mimic:
- **NOAA ISD Climate Normals** — temperature and precipitation statistics
- **NASA MODIS MOD13A3** — monthly NDVI vegetation index
- **USDA NASS** — crop type by region

Real production deployment would swap `generate_data.py` for API calls to:
- `https://www.ncei.noaa.gov/cdo-web/api/v2/` (NOAA CDO API)
- NASA EarthData MODIS Land Products
- OpenWeatherMap / Open-Meteo for near-realtime data

---

## Extending This Project

- **Real data**: Replace the CSV loader with NOAA + Open-Meteo API calls
- **ML forecasting**: Add a 30-day risk forecast using scikit-learn or Prophet
- **Crop calendar**: Integrate USDA crop calendars to weight frost/flood risk by growth stage
- **Export**: Add PDF report generation per region
