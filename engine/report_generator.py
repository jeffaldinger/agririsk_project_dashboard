"""
report_generator.py
Generates a styled PDF field risk report and CSV export for a selected region.

Dependencies: reportlab, matplotlib, pandas
"""

import io
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable

# ── Colour palette (matches dashboard) ───────────────────────────────────────
C_BG        = colors.HexColor("#0d1117")
C_SURFACE   = colors.HexColor("#161b22")
C_BORDER    = colors.HexColor("#21262d")
C_GREEN     = colors.HexColor("#4ade80")
C_YELLOW    = colors.HexColor("#facc15")
C_ORANGE    = colors.HexColor("#fb923c")
C_RED       = colors.HexColor("#ef4444")
C_BLUE      = colors.HexColor("#38bdf8")
C_TEXT      = colors.HexColor("#c9d1d9")
C_MUTED     = colors.HexColor("#8b949e")
C_WHITE     = colors.white

RISK_COLORS = {
    "Low":      "#4ade80",
    "Moderate": "#facc15",
    "High":     "#fb923c",
    "Critical": "#ef4444",
}

SCORE_COLORS = {
    "drought_score": "#ef4444",
    "frost_score":   "#38bdf8",
    "flood_score":   "#818cf8",
    "composite_score": "#fb923c",
}


def _score_color(score: float) -> colors.HexColor:
    if score >= 76: return C_RED
    if score >= 51: return C_ORANGE
    if score >= 26: return C_YELLOW
    return C_GREEN


def _make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=28, leading=34,
        textColor=C_GREEN, fontName="Helvetica-Bold",
        alignment=TA_LEFT, spaceAfter=6,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=11, leading=16,
        textColor=C_MUTED, fontName="Helvetica",
        alignment=TA_LEFT, spaceAfter=4,
    )
    styles["section_head"] = ParagraphStyle(
        "section_head", fontSize=8, leading=12,
        textColor=C_GREEN, fontName="Helvetica-Bold",
        alignment=TA_LEFT, spaceAfter=6,
        spaceBefore=14, letterSpacing=2,
    )
    styles["body"] = ParagraphStyle(
        "body", fontSize=9, leading=14,
        textColor=C_TEXT, fontName="Helvetica",
        alignment=TA_LEFT, spaceAfter=4,
    )
    styles["small"] = ParagraphStyle(
        "small", fontSize=7.5, leading=11,
        textColor=C_MUTED, fontName="Helvetica",
        alignment=TA_LEFT,
    )
    styles["label"] = ParagraphStyle(
        "label", fontSize=7, leading=10,
        textColor=C_MUTED, fontName="Helvetica-Bold",
        alignment=TA_LEFT, letterSpacing=1,
    )
    return styles


# ── Chart generators ─────────────────────────────────────────────────────────

def _chart_risk_trends(region_ts: pd.DataFrame, region_name: str) -> io.BytesIO:
    """Line chart of all four risk scores over time."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8), facecolor="#161b22")
    ax.set_facecolor("#161b22")

    for col, color, label in [
        ("composite_score", "#fb923c", "Composite"),
        ("drought_score",   "#ef4444", "Drought"),
        ("frost_score",     "#38bdf8", "Frost"),
        ("flood_score",     "#818cf8", "Flood"),
    ]:
        ax.plot(region_ts["month"], region_ts[col], color=color,
                linewidth=1.8, label=label)

    ax.set_ylim(0, 100)
    ax.axhline(50, color="#21262d", linewidth=0.8, linestyle="--")
    ax.axhline(75, color="#21262d", linewidth=0.8, linestyle="--")

    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.xaxis.set_tick_params(rotation=45)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.yaxis.label.set_color("#8b949e")
    ax.set_ylabel("Risk Score (0–100)", color="#8b949e", fontsize=7)

    # Show every 3rd month label to avoid crowding
    ticks = region_ts["month"].tolist()
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels([t if i % 3 == 0 else "" for i, t in enumerate(ticks)],
                       fontsize=6.5, color="#8b949e")

    legend = ax.legend(loc="upper left", fontsize=7, framealpha=0.2,
                       facecolor="#0d1117", edgecolor="#21262d",
                       labelcolor="#c9d1d9")
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#161b22")
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_ndvi(region_ts: pd.DataFrame) -> io.BytesIO:
    """NDVI mean + min area chart."""
    fig, ax = plt.subplots(figsize=(7.2, 2.2), facecolor="#161b22")
    ax.set_facecolor("#161b22")

    x = range(len(region_ts))
    ax.fill_between(x, region_ts["ndvi_min"], region_ts["ndvi_mean"],
                    color="#4ade80", alpha=0.15)
    ax.plot(x, region_ts["ndvi_mean"], color="#4ade80", linewidth=1.8, label="NDVI Mean")
    ax.plot(x, region_ts["ndvi_min"],  color="#16a34a", linewidth=1.0,
            linestyle="--", label="NDVI Min")

    ax.set_ylim(0, 1)
    ticks = region_ts["month"].tolist()
    ax.set_xticks(list(x))
    ax.set_xticklabels([t if i % 3 == 0 else "" for i, t in enumerate(ticks)],
                       fontsize=6.5, color="#8b949e", rotation=45)
    ax.tick_params(colors="#8b949e", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.set_ylabel("NDVI", color="#8b949e", fontsize=7)

    ax.legend(loc="upper left", fontsize=7, framealpha=0.2,
              facecolor="#0d1117", edgecolor="#21262d", labelcolor="#c9d1d9")
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#161b22")
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_precip_temp(region_ts: pd.DataFrame) -> io.BytesIO:
    """Dual-axis bar + line for precip and temperature."""
    fig, ax1 = plt.subplots(figsize=(7.2, 2.2), facecolor="#161b22")
    ax1.set_facecolor("#161b22")
    ax2 = ax1.twinx()

    x = range(len(region_ts))
    ax1.bar(x, region_ts["precip_total"], color="#38bdf8", alpha=0.6, width=0.6, label="Precip (mm)")
    ax2.plot(x, region_ts["temp_mean_c"], color="#facc15", linewidth=1.8, label="Temp (°C)")
    ax2.fill_between(x, region_ts["temp_min_c"], region_ts["temp_max_c"],
                     color="#facc15", alpha=0.08)

    ticks = region_ts["month"].tolist()
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([t if i % 3 == 0 else "" for i, t in enumerate(ticks)],
                        fontsize=6.5, color="#8b949e", rotation=45)
    for ax in [ax1, ax2]:
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#21262d")
    ax1.set_ylabel("Precipitation (mm)", color="#38bdf8", fontsize=7)
    ax2.set_ylabel("Temperature (°C)",   color="#facc15", fontsize=7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7,
               framealpha=0.2, facecolor="#0d1117", edgecolor="#21262d", labelcolor="#c9d1d9")
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#161b22")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── KPI score box helper ──────────────────────────────────────────────────────

def _score_table(latest: pd.Series) -> Table:
    """4-column score summary table."""
    def cell(label, score, rl):
        sc = colors.HexColor(RISK_COLORS.get(rl, "#4ade80"))
        return [
            Paragraph(f'<font color="#8b949e" size="7">{label.upper()}</font>', ParagraphStyle("x", fontSize=7, fontName="Helvetica")),
            Paragraph(f'<font color="{RISK_COLORS.get(rl,"#4ade80")}" size="18"><b>{score:.0f}</b></font>', ParagraphStyle("x", fontSize=18, fontName="Helvetica-Bold")),
            Paragraph(f'<font color="{RISK_COLORS.get(rl,"#4ade80")}" size="8">{rl}</font>', ParagraphStyle("x", fontSize=8, fontName="Helvetica")),
        ]

    data = [[
        cell("Composite",  latest["composite_score"], latest["composite_label"]),
        cell("Drought",    latest["drought_score"],   latest["drought_label"]),
        cell("Frost",      latest["frost_score"],     latest["frost_label"]),
        cell("Flood",      latest["flood_score"],     latest["flood_label"]),
    ]]

    # Flatten each cell into a sub-table
    col_data = []
    for item in data[0]:
        col_data.append(Table([[r] for r in item], colWidths=[42*mm]))

    outer = Table([col_data], colWidths=[42*mm]*4)
    outer.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), C_SURFACE),
        ("BOX",         (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",   (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",  (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING",(0,0), (-1,-1), 10),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    return outer


def _monthly_table(region_ts: pd.DataFrame, styles: dict) -> Table:
    """Monthly risk data table."""
    cols = ["month", "composite_score", "composite_label",
            "drought_score", "frost_score", "flood_score",
            "ndvi_mean", "precip_total", "temp_mean_c"]
    headers = ["Month", "Composite", "Level", "Drought", "Frost",
               "Flood", "NDVI", "Precip (mm)", "Temp (°C)"]

    header_row = [Paragraph(f'<font color="#8b949e" size="7"><b>{h}</b></font>',
                             ParagraphStyle("th", fontSize=7, fontName="Helvetica-Bold"))
                  for h in headers]

    rows = [header_row]
    for _, row in region_ts.sort_values("month").iterrows():
        sc = _score_color(row["composite_score"])
        def fmt(val, dec=1):
            return Paragraph(f'<font color="#c9d1d9" size="7">{val:.{dec}f}</font>',
                             ParagraphStyle("td", fontSize=7, fontName="Helvetica"))
        def fmts(val):
            return Paragraph(f'<font color="#c9d1d9" size="7">{val}</font>',
                             ParagraphStyle("td", fontSize=7, fontName="Helvetica"))
        def fmtl(val):
            lc = colors.HexColor(RISK_COLORS.get(val, "#4ade80"))
            return Paragraph(f'<font color="{RISK_COLORS.get(val,"#4ade80")}" size="7">{val}</font>',
                             ParagraphStyle("td", fontSize=7, fontName="Helvetica"))

        rows.append([
            fmts(row["month"]),
            fmt(row["composite_score"]),
            fmtl(row["composite_label"]),
            fmt(row["drought_score"]),
            fmt(row["frost_score"]),
            fmt(row["flood_score"]),
            fmt(row["ndvi_mean"], 3),
            fmt(row["precip_total"], 0),
            fmt(row["temp_mean_c"]),
        ])

    col_widths = [24*mm, 18*mm, 18*mm, 16*mm, 14*mm, 14*mm, 14*mm, 20*mm, 18*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  C_SURFACE),
        ("BACKGROUND",    (0,1), (-1,-1), colors.HexColor("#0d1117")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#0d1117"), colors.HexColor("#111820")]),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t


# ── Page templates ────────────────────────────────────────────────────────────

def _on_page(canvas, doc):
    """Dark background + footer on every page."""
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(C_BG)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    # Footer
    canvas.setFillColor(C_MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(20*mm, 12*mm, "AgriRisk Field Intelligence  ·  Confidential")
    canvas.drawRightString(w - 20*mm, 12*mm,
                           f"Page {doc.page}  ·  Generated {datetime.date.today().strftime('%d %b %Y')}")
    canvas.setStrokeColor(C_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(20*mm, 18*mm, w - 20*mm, 18*mm)
    canvas.restoreState()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf_report(region_name: str, risk_df: pd.DataFrame) -> bytes:
    """
    Generates a complete PDF risk report for a single region.
    Returns raw PDF bytes suitable for st.download_button.
    """
    styles = _make_styles()
    region_ts = risk_df[risk_df["region_name"] == region_name].sort_values("month")
    latest    = region_ts.iloc[-1]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=24*mm,
    )

    story = []

    # ── COVER SECTION ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("AGRIRISK", ParagraphStyle(
        "brand", fontSize=9, textColor=C_GREEN, fontName="Helvetica-Bold",
        letterSpacing=3, spaceAfter=8,
    )))
    story.append(Paragraph("Field Risk Intelligence Report", styles["cover_title"]))
    story.append(Paragraph(region_name, ParagraphStyle(
        "region", fontSize=16, textColor=C_TEXT, fontName="Helvetica",
        spaceAfter=4,
    )))
    story.append(Paragraph(
        f"Primary crop: {latest['primary_crop']}  ·  Climate zone: {latest['climate_zone']}  ·  "
        f"Reporting period: {region_ts['month'].min()} – {region_ts['month'].max()}",
        styles["cover_sub"],
    ))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 6*mm))

    # ── LATEST RISK SCORES ────────────────────────────────────────────────────
    story.append(Paragraph("CURRENT RISK SNAPSHOT", styles["section_head"]))
    story.append(Paragraph(
        f"Risk scores shown for the most recent reporting month: <b>{latest['month']}</b>. "
        "Scores range from 0 (no risk) to 100 (extreme risk).",
        styles["body"],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(_score_table(latest))
    story.append(Spacer(1, 6*mm))

    # ── RISK TREND CHART ──────────────────────────────────────────────────────
    story.append(Paragraph("RISK SCORE TRENDS", styles["section_head"]))
    story.append(Paragraph(
        "Historical drought, frost, flood, and composite risk scores across the full observation period. "
        "Dashed lines indicate the High (75) and Moderate (50) alert thresholds.",
        styles["body"],
    ))
    story.append(Spacer(1, 2*mm))
    trend_buf = _chart_risk_trends(region_ts, region_name)
    story.append(Image(trend_buf, width=170*mm, height=66*mm))
    story.append(Spacer(1, 6*mm))

    # ── NDVI CHART ────────────────────────────────────────────────────────────
    story.append(Paragraph("VEGETATION HEALTH (NDVI)", styles["section_head"]))
    story.append(Paragraph(
        "Normalized Difference Vegetation Index (NDVI) measures crop canopy health. "
        "Values below the long-term baseline indicate vegetation stress and amplify drought risk scores.",
        styles["body"],
    ))
    story.append(Spacer(1, 2*mm))
    ndvi_buf = _chart_ndvi(region_ts)
    story.append(Image(ndvi_buf, width=170*mm, height=55*mm))
    story.append(Spacer(1, 6*mm))

    # ── PRECIP + TEMP CHART ───────────────────────────────────────────────────
    story.append(Paragraph("PRECIPITATION & TEMPERATURE", styles["section_head"]))
    story.append(Paragraph(
        "Monthly total precipitation (bars) and mean temperature with min/max range (line). "
        "Extreme precipitation months are the primary driver of flood risk scores.",
        styles["body"],
    ))
    story.append(Spacer(1, 2*mm))
    pt_buf = _chart_precip_temp(region_ts)
    story.append(Image(pt_buf, width=170*mm, height=55*mm))

    # ── PAGE BREAK → DATA TABLE ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("MONTHLY RISK DATA TABLE", styles["section_head"]))
    story.append(Paragraph(
        "Full monthly breakdown of all risk scores and underlying meteorological indicators.",
        styles["body"],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(_monthly_table(region_ts, styles))
    story.append(Spacer(1, 8*mm))

    # ── METHODOLOGY ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("RISK METHODOLOGY", styles["section_head"]))

    method_text = """
    <b>Drought Score</b> (0–100): Weighted combination of precipitation deficit relative to long-term
    monthly mean (SPI proxy, up to 40 pts), NDVI depression below climatological baseline (up to 40 pts),
    and temperature excess amplifying water stress (up to 20 pts).<br/><br/>
    <b>Frost Score</b> (0–100): Penalises minimum temperatures below 5°C during the growing season
    (April–October) at 10 pts per degree, and below 0°C during dormant months at 5 pts per degree.<br/><br/>
    <b>Flood Score</b> (0–100): Based on monthly precipitation intensity relative to long-term mean
    (up to 60 pts) plus a 7-day extreme precipitation spike bonus as a flash flood proxy (up to 40 pts).
    Downweighted 30% in arid climates.<br/><br/>
    <b>Composite Score</b>: Drought x 0.45 + Frost x 0.25 + Flood x 0.30<br/><br/>
    <b>Data sources</b>: Synthetic dataset generated to mimic NOAA ISD Climate Normals and
    NASA MODIS MOD13A3 monthly NDVI. Real deployment would use live API feeds from NOAA CDO
    and NASA EarthData.
    """
    story.append(Paragraph(method_text, styles["small"]))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


def generate_csv_export(region_name: str, risk_df: pd.DataFrame) -> bytes:
    """
    Returns CSV bytes for the full risk time series of a region.
    """
    cols = [
        "month", "region_name", "primary_crop", "climate_zone",
        "composite_score", "composite_label",
        "drought_score", "drought_label",
        "frost_score", "frost_label",
        "flood_score", "flood_label",
        "ndvi_mean", "ndvi_min",
        "precip_total", "precip_30d_avg",
        "temp_mean_c", "temp_min_c", "temp_max_c",
    ]
    region_ts = risk_df[risk_df["region_name"] == region_name][cols].sort_values("month")
    return region_ts.to_csv(index=False).encode("utf-8")
