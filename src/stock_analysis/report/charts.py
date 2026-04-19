"""Chart generation — matplotlib charts embedded as PNG in reports."""

from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

import structlog

logger = structlog.get_logger(__name__)

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["font.size"] = 10


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _fig_to_file(fig: plt.Figure, path: Path) -> str:
    """Save figure to file and return path."""
    fig.savefig(path, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def chart_revenue_trend(
    historical: list[dict], projected: list[float] | None = None
) -> str:
    """Revenue trend chart — historical + projected."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Historical
    dates = [h.get("period_end", f"Y{i}") for i, h in enumerate(historical)]
    values = [h.get("value", 0) / 1e9 for h in historical]  # Convert to billions
    dates.reverse()
    values.reverse()

    ax.bar(dates, values, color="#2196F3", alpha=0.8, label="Historical")

    # Projected
    if projected:
        proj_labels = [f"P{i+1}" for i in range(len(projected))]
        proj_values = [p / 1e9 for p in projected]
        ax.bar(proj_labels, proj_values, color="#FF9800", alpha=0.7, label="Projected")

    ax.set_title("Revenue Trend ($B)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue ($B)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))

    return _fig_to_base64(fig)


def chart_margin_trend(financials: dict[str, Any]) -> str:
    """Profitability margins over time."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ratios = financials.get("profitability", {})
    metrics = {
        "Gross Margin": ratios.get("gross_margin"),
        "Operating Margin": ratios.get("operating_margin"),
        "Net Margin": ratios.get("net_margin"),
    }

    valid = {k: v for k, v in metrics.items() if v is not None}
    if valid:
        bars = ax.barh(list(valid.keys()), [v * 100 for v in valid.values()], color=["#4CAF50", "#2196F3", "#FF9800"])
        for bar, val in zip(bars, valid.values()):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=10)

    ax.set_title("Profitability Margins", fontsize=13, fontweight="bold")
    ax.set_xlabel("Margin (%)")

    return _fig_to_base64(fig)


def chart_sensitivity_heatmap(sensitivity: dict[str, Any]) -> str:
    """Sensitivity analysis heatmap — WACC × Terminal Growth."""
    fig, ax = plt.subplots(figsize=(8, 5))

    table = sensitivity.get("fair_value_table", [])
    wacc_range = sensitivity.get("wacc_range", [])
    growth_range = sensitivity.get("growth_range", [])

    if not table:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _fig_to_base64(fig)

    # Replace inf with NaN for display
    data = np.array(table, dtype=float)
    data[data > 1e10] = np.nan

    sns.heatmap(
        data,
        xticklabels=[f"{g:.1f}%" for g in growth_range],
        yticklabels=[f"{w:.1f}%" for w in wacc_range],
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        center=sensitivity.get("current_price", 0) or np.nanmean(data),
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title("Sensitivity Analysis: Fair Value per Share ($)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Terminal Growth Rate")
    ax.set_ylabel("WACC")

    return _fig_to_base64(fig)


def chart_peer_comparison(comparables: dict[str, Any], ticker: str) -> str:
    """Peer comparison bar chart — EV/EBITDA."""
    fig, ax = plt.subplots(figsize=(8, 4))

    peers = comparables.get("comp_summary", {}).get("peers", [])
    target_mult = comparables.get("target_multiples", {})

    if not peers:
        ax.text(0.5, 0.5, "No peer data", ha="center", va="center")
        return _fig_to_base64(fig)

    # EV/EBITDA comparison
    names = [p["ticker"] for p in peers if p.get("ev_to_ebitda")]
    values = [p["ev_to_ebitda"] for p in peers if p.get("ev_to_ebitda")]

    # Add target
    target_ev_ebitda = target_mult.get("ev_to_ebitda")
    if target_ev_ebitda:
        names.insert(0, ticker)
        values.insert(0, target_ev_ebitda)

    colors = ["#FF5722" if n == ticker else "#2196F3" for n in names]
    ax.barh(names, values, color=colors)
    ax.set_title("EV/EBITDA — Peer Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("EV/EBITDA")

    # Add median line
    if values:
        median_val = np.median(values)
        ax.axvline(x=median_val, color="red", linestyle="--", alpha=0.7, label=f"Median: {median_val:.1f}x")
        ax.legend()

    return _fig_to_base64(fig)


def chart_revenue_history(history: list[dict[str, Any]], projected: list[float] | None = None) -> str:
    """Revenue bar chart from SEC historical + projected list."""
    fig, ax = plt.subplots(figsize=(8, 4))

    rows = list(reversed([h for h in history if h.get("revenue")]))
    if not rows:
        ax.text(0.5, 0.5, "No revenue history", ha="center", va="center")
        return _fig_to_base64(fig)

    labels = [r["period_end"][:4] for r in rows]
    values = [(r.get("revenue") or 0) / 1e9 for r in rows]
    ax.bar(labels, values, color="#1E3A8A", alpha=0.85, label="Historical")

    if projected:
        plabels = [f"F{i+1}" for i in range(len(projected))]
        pvalues = [p / 1e9 for p in projected]
        ax.bar(plabels, pvalues, color="#D97706", alpha=0.8, label="Projected")

    ax.set_title("Revenue ($B)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue ($B)")
    ax.legend(loc="upper left")
    return _fig_to_base64(fig)


def generate_all_charts(analysis: dict[str, Any]) -> dict[str, str]:
    """Generate all charts for a report. Returns dict of chart_name → base64 PNG.

    Expects the analysis dict from `analyze_ticker` with a `valuation` key holding
    the industry-routed valuation bundle.
    """
    logger.info("generating_charts", ticker=analysis.get("ticker", ""))
    charts: dict[str, str] = {}

    bundle = analysis.get("valuation", {})
    primary = bundle.get("valuation", {}) or {}
    comps = bundle.get("comparables", {}) or {}
    history = bundle.get("financials_history", []) or []

    # Revenue history + projected (FCFF model only — others don't project revenue)
    projected = primary.get("projected_revenue")
    if history:
        charts["revenue_history"] = chart_revenue_history(history, projected)

    # Sensitivity heatmap (FCFF model only)
    sensitivity = primary.get("sensitivity", {})
    if sensitivity:
        charts["sensitivity_heatmap"] = chart_sensitivity_heatmap(sensitivity)

    # Peer comparison
    if comps:
        charts["peer_comparison"] = chart_peer_comparison(comps, analysis.get("ticker", ""))

    return charts
