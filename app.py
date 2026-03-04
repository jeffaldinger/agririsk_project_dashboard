"""
app.py  —  AgriRisk Field Intelligence Dashboard
Run with:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.risk_engine import compute_risks, get_latest_snapshot, get_active_alerts
from engine.report_generator import generate_pdf_report, generate_csv_export

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriRisk Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }

  .stApp { background-color: #0d1117; }

  /* Header strip */
  .dash-header {
    background: linear-gradient(135deg, #1a2a1a 0%, #0d1f0d 50%, #111827 100%);
    border: 1px solid #2d4a2d;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .dash-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: #4ade80;
    letter-spacing: -0.5px;
    margin: 0;
  }
  .dash-subtitle {
    font-size: 0.85rem;
    color: #8b949e;
    margin: 4px 0 0 0;
  }

  /* KPI cards */
  .kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .kpi-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; line-height: 1; }
  .kpi-label { font-size: 0.75rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

  /* Alert badges */
  .alert-critical { background: #450a0a; border-left: 3px solid #ef4444; border-radius: 6px; padding: 10px 14px; margin: 4px 0; }
  .alert-high     { background: #431407; border-left: 3px solid #fb923c; border-radius: 6px; padding: 10px 14px; margin: 4px 0; }
  .alert-moderate { background: #422006; border-left: 3px solid #facc15; border-radius: 6px; padding: 10px 14px; margin: 4px 0; }

  /* Section headers */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
    margin-top: 4px;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d;
  }

  /* Hide streamlit chrome */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  /* Plotly chart backgrounds */
  .js-plotly-plot { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    daily   = pd.read_csv(f"{base}/data/daily_weather_ndvi.csv",  parse_dates=["date"])
    monthly = pd.read_csv(f"{base}/data/monthly_aggregates.csv")
    risk_df = compute_risks(monthly, daily)
    return daily, monthly, risk_df

with st.spinner("Loading field intelligence data…"):
    daily_df, monthly_df, risk_df = load_data()

snapshot   = get_latest_snapshot(risk_df)
all_regions = sorted(risk_df["region_name"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">🌾 AgriRisk Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)

    selected_regions = st.multiselect(
        "Regions",
        options=all_regions,
        default=all_regions,
    )

    alert_threshold = st.slider(
        "Alert threshold (score)",
        min_value=25, max_value=80, value=50, step=5,
        help="Minimum risk score to appear in the Active Alerts panel"
    )

    risk_type = st.selectbox(
        "Primary risk to display on map",
        options=["composite_score", "drought_score", "frost_score", "flood_score"],
        format_func=lambda x: x.replace("_score", "").title(),
    )

    st.markdown("---")

    # month selector
    all_months = sorted(risk_df["month"].unique())
    selected_month = st.selectbox("Analysis month", options=all_months, index=len(all_months)-1)

    st.markdown("---")
    st.markdown('<p class="section-label">Export Report</p>', unsafe_allow_html=True)

    export_region = st.selectbox(
        "Region to export",
        options=all_regions,
        key="export_region",
    )

    with st.spinner("Preparing PDF..."):
        pdf_bytes = generate_pdf_report(export_region, risk_df)

    safe_name = export_region.replace(" ", "_").replace("/", "-")
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"AgriRisk_{safe_name}_{selected_month}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    csv_bytes = generate_csv_export(export_region, risk_df)
    st.download_button(
        label="Download CSV Data",
        data=csv_bytes,
        file_name=f"AgriRisk_{safe_name}_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("""
    <p style='font-size:0.7rem;color:#484f58;line-height:1.6'>
    <b style='color:#4ade80'>Data sources (synthetic)</b><br>
    Weather: NOAA ISD normals<br>
    Vegetation: NASA MODIS NDVI<br>
    Risk model: SPI + NDVI anomaly<br>
    </p>
    """, unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
filtered_risk = risk_df[
    (risk_df["region_name"].isin(selected_regions)) &
    (risk_df["month"] == selected_month)
]
filtered_all  = risk_df[risk_df["region_name"].isin(selected_regions)]
alerts        = get_active_alerts(filtered_risk, threshold=alert_threshold)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
  <div>
    <p class="dash-title">🌾 AgriRisk Field Intelligence</p>
    <p class="dash-subtitle">Drought · Frost · Flood risk monitoring across agricultural regions &nbsp;|&nbsp; Reporting period: <b style='color:#c9d1d9'>{selected_month}</b></p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
n_critical = int((filtered_risk["composite_score"] >= 76).sum())
n_high     = int(((filtered_risk["composite_score"] >= 51) & (filtered_risk["composite_score"] < 76)).sum())
avg_drought= filtered_risk["drought_score"].mean()
avg_ndvi   = filtered_risk["ndvi_mean"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:#ef4444">{n_critical}</div>
      <div class="kpi-label">Critical alerts</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:#fb923c">{n_high}</div>
      <div class="kpi-label">High risk regions</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:#facc15">{avg_drought:.0f}</div>
      <div class="kpi-label">Avg drought score</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:#38bdf8">{avg_ndvi:.3f}</div>
      <div class="kpi-label">Avg NDVI</div>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:#4ade80">{len(selected_regions)}</div>
      <div class="kpi-label">Regions monitored</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAP + ALERTS ──────────────────────────────────────────────────────────────
col_map, col_alerts = st.columns([2, 1])

with col_map:
    st.markdown('<p class="section-label">Regional Risk Map</p>', unsafe_allow_html=True)

    risk_label = risk_type.replace("_score", "").title()
    fig_map = px.scatter_map(
        filtered_risk,
        lat="lat", lon="lon",
        size=risk_type,
        color=risk_type,
        hover_name="region_name",
        hover_data={
            "primary_crop":    True,
            "drought_score":   True,
            "frost_score":     True,
            "flood_score":     True,
            "composite_score": True,
            "lat": False, "lon": False,
            risk_type: False,
        },
        color_continuous_scale=["#1a4731","#4ade80","#facc15","#fb923c","#ef4444"],
        range_color=[0, 100],
        size_max=35,
        zoom=3.2,
        center={"lat": 39, "lon": -98},
        map_style="carto-darkmatter",
        title=f"{risk_label} Risk — {selected_month}",
        labels={risk_type: f"{risk_label} Score"},
    )
    fig_map.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
        coloraxis_colorbar=dict(
            tickfont=dict(color="#8b949e"),
            title=dict(font=dict(color="#8b949e")),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_alerts:
    st.markdown('<p class="section-label">Active Alerts</p>', unsafe_allow_html=True)
    if alerts.empty:
        st.success("✅ No active alerts above threshold for selected regions.")
    else:
        for _, row in alerts.iterrows():
            score = row["composite_score"]
            if score >= 76:
                css_class = "alert-critical"; badge = "🔴 CRITICAL"
            elif score >= 51:
                css_class = "alert-high";     badge = "🟠 HIGH"
            else:
                css_class = "alert-moderate"; badge = "🟡 MODERATE"

            drivers = []
            if row["drought_score"] >= alert_threshold: drivers.append(f"Drought {row['drought_score']:.0f}")
            if row["frost_score"]   >= alert_threshold: drivers.append(f"Frost {row['frost_score']:.0f}")
            if row["flood_score"]   >= alert_threshold: drivers.append(f"Flood {row['flood_score']:.0f}")

            st.markdown(f"""
            <div class="{css_class}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <b style="font-size:0.85rem">{row['region_name']}</b>
                <span style="font-size:0.75rem">{badge}</span>
              </div>
              <div style="font-size:0.75rem;color:#8b949e;margin-top:4px">
                {row['primary_crop']}<br>
                <span style="color:#c9d1d9">{'  ·  '.join(drivers)}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TIME SERIES ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Risk Trends Over Time</p>', unsafe_allow_html=True)

trend_region = st.selectbox(
    "Select region for trend analysis",
    options=selected_regions,
    key="trend_region"
)

region_ts = filtered_all[filtered_all["region_name"] == trend_region].sort_values("month")

fig_ts = make_subplots(
    rows=2, cols=2,
    subplot_titles=["Composite Risk Score", "NDVI (Vegetation Health)", "Monthly Precipitation (mm)", "Temperature (°C)"],
    vertical_spacing=0.18,
    horizontal_spacing=0.1,
)

PLOT_BG = "#161b22"
GRID    = "#21262d"

def ts_line(y_col, color, row, col, name, fill=None):
    fig_ts.add_trace(go.Scatter(
        x=region_ts["month"],
        y=region_ts[y_col],
        name=name,
        line=dict(color=color, width=2),
        fill=fill,
        fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if fill else None,
    ), row=row, col=col)

ts_line("composite_score", "#fb923c", 1, 1, "Composite")
ts_line("drought_score",   "#ef4444", 1, 1, "Drought")
ts_line("frost_score",     "#38bdf8", 1, 1, "Frost")
ts_line("flood_score",     "#818cf8", 1, 1, "Flood")

ts_line("ndvi_mean", "#4ade80", 1, 2, "NDVI Mean", fill="tozeroy")
ts_line("ndvi_min",  "#16a34a", 1, 2, "NDVI Min")

# Precip bars
fig_ts.add_trace(go.Bar(
    x=region_ts["month"],
    y=region_ts["precip_total"],
    name="Precip",
    marker_color="#38bdf8",
    opacity=0.7,
), row=2, col=1)

# Temp range fill
fig_ts.add_trace(go.Scatter(
    x=pd.concat([region_ts["month"], region_ts["month"].iloc[::-1]]),
    y=pd.concat([region_ts["temp_max_c"], region_ts["temp_min_c"].iloc[::-1]]),
    fill="toself",
    fillcolor="rgba(250,204,21,0.12)",
    line=dict(color="rgba(250,204,21,0)"),
    name="Temp range",
), row=2, col=2)
fig_ts.add_trace(go.Scatter(
    x=region_ts["month"], y=region_ts["temp_mean_c"],
    line=dict(color="#facc15", width=2), name="Temp mean",
), row=2, col=2)

fig_ts.update_layout(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font_color="#c9d1d9",
    height=500,
    showlegend=True,
    legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1, font_size=11),
    margin=dict(l=10, r=10, t=50, b=10),
)
for i in range(1, 5):
    row, col = ((i-1)//2)+1, ((i-1)%2)+1
    fig_ts.update_xaxes(gridcolor=GRID, showgrid=True, row=row, col=col, tickangle=45, tickfont_size=9)
    fig_ts.update_yaxes(gridcolor=GRID, showgrid=True, row=row, col=col)

st.plotly_chart(fig_ts, use_container_width=True)

# ── HEATMAP: region × month composite risk ────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">Risk Heatmap — All Regions × All Months</p>', unsafe_allow_html=True)

hm_pivot = filtered_all.pivot_table(
    index="region_name", columns="month", values="composite_score"
).round(1)

fig_hm = px.imshow(
    hm_pivot,
    color_continuous_scale=["#1a4731", "#4ade80", "#facc15", "#fb923c", "#ef4444"],
    range_color=[0, 100],
    aspect="auto",
    labels=dict(color="Risk Score"),
)
fig_hm.update_layout(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font_color="#c9d1d9",
    margin=dict(l=10, r=10, t=10, b=50),
    height=350,
    xaxis=dict(tickangle=45, tickfont_size=9),
    coloraxis_colorbar=dict(tickfont=dict(color="#8b949e")),
)
st.plotly_chart(fig_hm, use_container_width=True)

# ── DATA TABLE ────────────────────────────────────────────────────────────────
with st.expander("📋 Raw risk data table"):
    display_cols = [
        "region_name", "primary_crop", "month",
        "composite_score", "composite_label",
        "drought_score", "drought_label",
        "frost_score", "frost_label",
        "flood_score", "flood_label",
        "ndvi_mean", "precip_total", "temp_mean_c",
    ]
    st.dataframe(
        filtered_all[display_cols].sort_values(["month", "composite_score"], ascending=[True, False]),
        use_container_width=True,
        height=300,
    )
