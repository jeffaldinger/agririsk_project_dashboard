"""
risk_engine.py
Computes drought, frost, and flood risk scores for each region/month.

Risk scores are 0–100 where:
  0–25  → Low
  26–50 → Moderate
  51–75 → High
  76+   → Critical

Methodology mirrors standard agro-meteorological indices:
  - Drought  : NDVI anomaly + Standardized Precipitation Index (SPI-like) + temp excess
  - Frost    : Days near/below 0°C relative to crop-sensitive growth windows
  - Flood    : 7-day and 30-day extreme precipitation percentiles
"""

import pandas as pd
import numpy as np

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_LABELS = {
    (0,  26): ("Low",      "#4ade80"),
    (26, 51): ("Moderate", "#facc15"),
    (51, 76): ("High",     "#fb923c"),
    (76, 101):("Critical", "#ef4444"),
}

def score_to_label(score: float) -> tuple[str, str]:
    for (lo, hi), (label, color) in RISK_LABELS.items():
        if lo <= score < hi:
            return label, color
    return "Critical", "#ef4444"


def compute_risks(monthly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins monthly aggregates with computed risk scores.
    Returns enriched DataFrame ready for the dashboard.
    """

    df = monthly_df.copy()
    df["month_dt"] = pd.to_datetime(df["month"])

    # ── Per-region historical baselines (from full daily record) ─────────────
    daily_df = daily_df.copy()
    daily_df["month"] = pd.to_datetime(daily_df["date"]).dt.to_period("M").astype(str)
    daily_df["month_num"] = pd.to_datetime(daily_df["date"]).dt.month

    # Long-term monthly means per region (used for anomaly calc)
    baseline = (
        daily_df.groupby(["region_id", "month_num"])
        .agg(
            precip_ltm=("precip_mm",  lambda x: x.sum() / daily_df["date"].nunique() * 30),
            ndvi_ltm   =("ndvi",       "mean"),
            temp_ltm   =("temp_c",     "mean"),
        )
        .reset_index()
    )

    df["month_num"] = df["month_dt"].dt.month
    df = df.merge(baseline, on=["region_id", "month_num"], how="left")

    # ── DROUGHT SCORE ─────────────────────────────────────────────────────────
    # Component 1: Precipitation deficit (SPI proxy) — 0 to 40 pts
    precip_norm = df["precip_total"] / (df["precip_ltm"] + 1e-6)
    precip_deficit_score = np.clip((1 - precip_norm) * 60, 0, 40)

    # Component 2: NDVI depression below long-term mean — 0 to 40 pts
    ndvi_anom = df["ndvi_ltm"] - df["ndvi_mean"]   # positive = vegetation stress
    ndvi_stress_score = np.clip(ndvi_anom * 300, 0, 40)

    # Component 3: Temperature excess (heat amplifies drought) — 0 to 20 pts
    temp_excess = np.clip(df["temp_mean_c"] - df["temp_ltm"], 0, None)
    temp_score  = np.clip(temp_excess * 3, 0, 20)

    df["drought_score"] = np.clip(precip_deficit_score + ndvi_stress_score + temp_score, 0, 100)

    # ── FROST SCORE ───────────────────────────────────────────────────────────
    # Based on minimum temperature during growing months (April–October)
    growing_season = df["month_num"].between(4, 10)

    frost_raw = np.where(
        growing_season,
        np.clip((5 - df["temp_min_c"]) * 10, 0, 100),  # every 1°C below 5 = +10 pts
        np.clip((0 - df["temp_min_c"]) * 5,  0, 60),   # dormant season: less penalized
    )
    df["frost_score"] = np.clip(frost_raw, 0, 100)

    # ── FLOOD SCORE ───────────────────────────────────────────────────────────
    # Based on monthly total + 7-day extreme relative to long-term
    flood_intensity = df["precip_total"] / (df["precip_ltm"] + 1e-6)
    flood_base      = np.clip((flood_intensity - 1.5) * 40, 0, 60)

    # 7-day spike bonus (flash flood proxy)
    spike_bonus = np.clip((df["precip_30d_avg"] - 15) * 1.5, 0, 40)

    # Suppress flood risk in dry climates (semi-arid, high desert) slightly
    dry_discount = np.where(df["climate_zone"].isin(["semi-arid", "high desert"]), 0.7, 1.0)
    df["flood_score"] = np.clip((flood_base + spike_bonus) * dry_discount, 0, 100)

    # ── COMPOSITE RISK ────────────────────────────────────────────────────────
    df["composite_score"] = (
        df["drought_score"] * 0.45 +
        df["frost_score"]   * 0.25 +
        df["flood_score"]   * 0.30
    ).clip(0, 100)

    # ── Labels ────────────────────────────────────────────────────────────────
    for col in ["drought_score", "frost_score", "flood_score", "composite_score"]:
        short = col.replace("_score", "")
        df[f"{short}_label"], df[f"{short}_color"] = zip(*df[col].apply(score_to_label))

    df["drought_score"]    = df["drought_score"].round(1)
    df["frost_score"]      = df["frost_score"].round(1)
    df["flood_score"]      = df["flood_score"].round(1)
    df["composite_score"]  = df["composite_score"].round(1)

    return df


def get_latest_snapshot(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Returns the most recent month's risk data per region."""
    latest_month = risk_df["month"].max()
    return risk_df[risk_df["month"] == latest_month].copy()


def get_active_alerts(risk_df: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
    """Returns rows where any risk score exceeds threshold, sorted by severity."""
    snap = get_latest_snapshot(risk_df)
    alerts = snap[
        (snap["drought_score"] >= threshold) |
        (snap["frost_score"]   >= threshold) |
        (snap["flood_score"]   >= threshold)
    ].copy()
    alerts = alerts.sort_values("composite_score", ascending=False)
    return alerts
