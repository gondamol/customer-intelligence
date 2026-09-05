"""Presentation primitives: tiles, panels, badges, and number formatting.

Small deliberately. The point is that every page renders the same way, so a
reader learns the layout once.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .theme import BORDER, INK, INK_MUTED, RISK_COLOURS, STATUS

CURRENCY = "MU"


# ------------------------------------------------------------ formatting ----


def money(value: float | None, decimals: int = 0) -> str:
    """Monetary units, abbreviated once the digits stop being informative."""
    if value is None or pd.isna(value):
        return "—"
    v = float(value)
    for cut, suffix in ((1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if abs(v) >= cut:
            return f"{CURRENCY} {v / cut:,.1f}{suffix}"
    return f"{CURRENCY} {v:,.{decimals}f}"


def num(value: float | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def pct(value: float | None, decimals: int = 1, of_one: bool = True) -> str:
    if value is None or pd.isna(value):
        return "—"
    v = float(value) * 100 if of_one else float(value)
    return f"{v:,.{decimals}f}%"


# ----------------------------------------------------------------- blocks ----


def page_header(eyebrow: str, title: str, lead: str = "") -> None:
    st.markdown(
        f'<div class="ci-eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True
    )
    st.markdown(f"# {title}")
    if lead:
        st.markdown(f'<p class="ci-lead">{lead}</p>', unsafe_allow_html=True)


def synthetic_notice(extra: str = "") -> None:
    st.markdown(
        '<div class="ci-synthetic"><strong>Synthetic demonstration.</strong> '
        "Every customer, transaction and score on this page was generated for "
        "this portfolio project. Nothing here describes a real organisation, a "
        "real customer, or a real portfolio, and no output is a financial, "
        "credit, or customer decision." + (f" {extra}" if extra else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<p class="ci-note">{text}</p>', unsafe_allow_html=True)


def tiles(items: list[dict], columns: int | None = None) -> None:
    """A row of stat tiles.

    Each item takes ``label``, ``value`` and an optional ``sub``. No sparkline,
    no delta arrow unless there is a real comparison to make -- a tile that
    invents a trend is worse than one that states a number.
    """
    n = columns or len(items)
    cards = "".join(
        f'<div class="ci-tile">'
        f'<div class="ci-tile-label">{html.escape(str(i["label"]))}</div>'
        f'<div class="ci-tile-value">{i["value"]}</div>'
        + (f'<div class="ci-tile-sub">{i["sub"]}</div>' if i.get("sub") else "")
        + "</div>"
        for i in items
    )
    st.markdown(
        f'<div class="ci-tiles" style="grid-template-columns:repeat({n},minmax(0,1fr));">'
        f"{cards}</div>",
        unsafe_allow_html=True,
    )


def panel(title: str, rows: list[tuple[str, str]]) -> None:
    """A bordered key/value panel."""
    body = "".join(
        f'<div class="ci-kv"><span class="ci-kv-k">{html.escape(str(k))}</span>'
        f'<span class="ci-kv-v">{v}</span></div>'
        for k, v in rows
    )
    st.markdown(
        f'<div class="ci-panel"><h4>{html.escape(title)}</h4>{body}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "neutral") -> str:
    """Status badge. Always carries its own label, never colour alone."""
    palette = {
        "good": (STATUS["good"], "#eaf7ea"), "warning": (STATUS["warning"], "#fff8e6"),
        "serious": (STATUS["serious"], "#fdf0ea"), "critical": (STATUS["critical"], "#fdecec"),
        "neutral": (INK_MUTED, "#f2f4f6"),
    }
    fg, bg = palette.get(kind, palette["neutral"])
    return (f'<span class="ci-badge" style="color:{fg};background:{bg};'
            f'border-color:{fg}33;">{html.escape(str(text))}</span>')


def risk_badge(band_label: str) -> str:
    kind = {"Low": "good", "Moderate": "warning", "Elevated": "serious", "High": "critical"}
    return badge(band_label, kind.get(band_label, "neutral"))


def priority_badge(priority: str) -> str:
    kind = {"Urgent": "critical", "High": "serious", "Standard": "neutral", "Low": "neutral"}
    return badge(priority, kind.get(priority, "neutral"))


def rule() -> None:
    st.markdown('<hr class="ci-rule">', unsafe_allow_html=True)


def section(title: str, lead: str = "") -> None:
    st.markdown(f"## {title}")
    if lead:
        st.markdown(f'<p class="ci-lead">{lead}</p>', unsafe_allow_html=True)
