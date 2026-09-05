"""Chart builders.

Each function does one thing and returns a figure ready to render. The rules
they all follow:

  * one y-axis, never two;
  * a legend whenever two or more series share a panel, and none for one series
    (the heading names it);
  * gridlines on the value axis only, hairline weight, behind the marks;
  * colour assigned by job -- sequential for magnitude, the categorical slots for
    identity, the status ramp only for state;
  * a hover layer on every chart, because a chart the reader cannot interrogate
    is a picture.

Where 50,000 marks would overlap into a single blob, the form changes -- a
density heatmap rather than a scatter -- instead of being drawn anyway at low
opacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import (
    AQUA, AXIS, BLUE, GRID, INK, INK_MUTED, INK_SECONDARY, ORANGE, ORDINAL,
    QUADRANT_COLOURS, RISK_COLOURS, SEQUENTIAL, STATUS, SURFACE, apply,
)


def _hover(fmt: str) -> dict:
    return dict(hovertemplate=fmt + "<extra></extra>")


# ------------------------------------------------------------------- bars ----


def hbar(
    labels: list[str], values: list[float], value_fmt: str = "{:,.0f}",
    colour: str = BLUE, height: int = 300, label_suffix: str = "",
    hover_label: str = "Value",
) -> go.Figure:
    """Horizontal bars, one hue, sorted by the reader's axis.

    Single-hue because the category name is already on the axis: colouring each
    bar differently would encode the same information twice and spend the
    categorical palette on decoration.
    """
    text = [value_fmt.format(v) + label_suffix for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colour, line=dict(width=0)),
        text=text, textposition="outside",
        textfont=dict(size=12, color=INK_SECONDARY),
        cliponaxis=False,
        **_hover("<b>%{y}</b><br>" + hover_label + ": %{text}"),
    ))
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor=GRID, showticklabels=False,
                   linecolor="rgba(0,0,0,0)", ticks=""),
        yaxis=dict(showgrid=False, autorange="reversed",
                   tickfont=dict(size=12, color=INK)),
        margin=dict(l=8, r=64, t=12, b=8), bargap=0.34,
    )
    return apply(fig, height, showlegend=False)


def banded_bar(
    counts: pd.Series, palette: dict[str, str], height: int = 260,
    total: int | None = None,
) -> go.Figure:
    """Counts by an ordered status band. Colour carries state, and the band name
    is always printed beside it -- never colour alone."""
    total = total or int(counts.sum())
    labels = list(counts.index)
    values = list(counts.values)
    text = [f"{v:,} · {v / total:.1%}" for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=[palette.get(l, INK_MUTED) for l in labels], line=dict(width=0)),
        text=text, textposition="outside",
        textfont=dict(size=12, color=INK_SECONDARY), cliponaxis=False,
        **_hover("<b>%{x}</b><br>%{text} of the book"),
    ))
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=12, color=INK)),
        yaxis=dict(showticklabels=False, title=None),
        margin=dict(l=8, r=8, t=28, b=8), bargap=0.42,
    )
    return apply(fig, height, showlegend=False)


# ------------------------------------------------------------ distribution ----


def histogram(
    values: pd.Series, bins: int = 40, colour: str = BLUE, height: int = 260,
    x_title: str = "", cut: float | None = None, cut_label: str = "",
) -> go.Figure:
    """Distribution of one measure, with an optional decision threshold marked."""
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=bins,
        marker=dict(color=colour, line=dict(width=0)),
        **_hover(f"{x_title or 'Value'}: %{{x}}<br>Customers: %{{y:,}}"),
    ))
    if cut is not None:
        fig.add_vline(
            x=cut, line=dict(color=INK, width=1.5, dash="dot"),
            annotation_text=cut_label, annotation_position="top right",
            annotation_font=dict(size=11, color=INK_SECONDARY),
        )
    fig.update_layout(
        xaxis=dict(title=x_title), yaxis=dict(title="Customers"),
        margin=dict(l=8, r=8, t=28, b=8), bargap=0.04,
    )
    return apply(fig, height, showlegend=False)


# ------------------------------------------------------------------ lines ----


def trend_lines(
    df: pd.DataFrame, x: str, series: dict[str, str], height: int = 280,
    y_title: str = "", value_fmt: str = ":,.0f",
) -> go.Figure:
    """One or more series over time, on a single shared axis.

    Two measures of different scale get two charts, never two y-axes -- a dual
    axis lets the author choose where the lines cross.
    """
    fig = go.Figure()
    palette = [BLUE, ORANGE, AQUA]
    for i, (col, name) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], name=name, mode="lines+markers",
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=6, line=dict(width=1.5, color=SURFACE)),
            hovertemplate=f"<b>{name}</b><br>Month %{{x}}: %{{y{value_fmt}}}<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="Month index"), yaxis=dict(title=y_title),
        hovermode="x unified", margin=dict(l=8, r=8, t=34, b=8),
    )
    return apply(fig, height, showlegend=len(series) > 1)


def calibration(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Predicted against observed rate, by decile of predicted probability.

    The diagonal is perfect calibration. Distance from it is the answer to "when
    it says 20%, does 20% happen" -- which ranking metrics like AUC cannot see.
    """
    hi = float(max(df["predicted"].max(), df["observed"].max())) * 1.08
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, hi], y=[0, hi], mode="lines", name="Perfect calibration",
        line=dict(color=AXIS, width=1.5, dash="dash"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["predicted"], y=df["observed"], mode="lines+markers", name="Model",
        line=dict(color=BLUE, width=2),
        marker=dict(size=9, line=dict(width=1.5, color=SURFACE)),
        customdata=df["n"],
        hovertemplate=("Predicted %{x:.1%}<br>Observed %{y:.1%}"
                       "<br>%{customdata:,} customers<extra></extra>"),
    ))
    fig.update_layout(
        xaxis=dict(title="Predicted probability", tickformat=".0%",
                   showgrid=True, gridcolor=GRID),
        yaxis=dict(title="Observed rate", tickformat=".0%"),
        margin=dict(l=8, r=8, t=34, b=8),
    )
    return apply(fig, height, showlegend=True)


# ----------------------------------------------------------------- matrix ----


def risk_opportunity_matrix(
    opportunity: pd.Series, churn: pd.Series,
    opportunity_cut: float = 60.0, risk_cut: float = 0.25, height: int = 420,
) -> go.Figure:
    """The risk/opportunity matrix as a density heatmap.

    Fifty thousand points plotted individually is a blob, not a chart, so the
    form changes: a 2-D density surface with the decision lines drawn on it. The
    quadrant names sit outside the plotting area so they never obscure the data.
    """
    fig = go.Figure(go.Histogram2d(
        x=churn, y=opportunity, nbinsx=44, nbinsy=44,
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        hovertemplate=("Attrition risk %{x:.0%}<br>Opportunity %{y:.0f}"
                       "<br>%{z:,} customers<extra></extra>"),
        colorbar=dict(
            title=dict(text="Customers", font=dict(size=11, color=INK_MUTED)),
            thickness=10, len=0.7, outlinewidth=0,
            tickfont=dict(size=11, color=INK_MUTED),
        ),
    ))
    fig.add_vline(x=risk_cut, line=dict(color=INK, width=1.2, dash="dot"))
    fig.add_hline(y=opportunity_cut, line=dict(color=INK, width=1.2, dash="dot"))

    xmax = float(max(churn.max(), risk_cut * 2))
    for label, (px, py) in {
        "Monitor": (risk_cut + (xmax - risk_cut) / 2, opportunity_cut / 2),
        "Retain": (risk_cut + (xmax - risk_cut) / 2, opportunity_cut + (100 - opportunity_cut) / 2),
        "Develop": (risk_cut / 2, opportunity_cut / 2),
        "Grow": (risk_cut / 2, opportunity_cut + (100 - opportunity_cut) / 2),
    }.items():
        fig.add_annotation(
            x=px, y=py, text=f"<b>{label}</b>", showarrow=False,
            font=dict(size=13, color=QUADRANT_COLOURS[label]),
            bgcolor="rgba(255,255,255,0.82)", borderpad=4,
        )

    fig.update_layout(
        xaxis=dict(title="Attrition risk", tickformat=".0%", showgrid=False, range=[0, xmax]),
        yaxis=dict(title="Relationship opportunity score", showgrid=False, range=[0, 100]),
        margin=dict(l=8, r=8, t=28, b=8),
    )
    return apply(fig, height, showlegend=False)


# ----------------------------------------------------------- explanation ----


def contribution_bars(reasons: list[dict], height: int = 260) -> go.Figure:
    """Why one customer scores as they do.

    Diverging by sign: blue pushes the score down, red pushes it up. Each bar is
    the model's standardised coefficient times this customer's distance from the
    book average -- the actual arithmetic, not an approximation of it.
    """
    labels = [r["label"] for r in reasons][::-1]
    values = [r["contribution"] for r in reasons][::-1]
    colours = [STATUS["critical"] if v > 0 else BLUE for v in values]
    text = [f"{r['value']:,.1f} vs {r['population_mean']:,.1f} avg" for r in reasons][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        customdata=text,
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>Contribution %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_layout(
        xaxis=dict(title="Contribution to the score", showgrid=True, gridcolor=GRID,
                   zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color=INK)),
        margin=dict(l=8, r=8, t=28, b=8), bargap=0.36,
    )
    return apply(fig, height, showlegend=False)


def grouped_profile(
    df: pd.DataFrame, category: str, value: str, height: int = 300,
    value_fmt: str = "{:,.0f}", colour: str = BLUE, hover_label: str = "Value",
) -> go.Figure:
    d = df.sort_values(value, ascending=False)
    return hbar(list(d[category]), list(d[value]), value_fmt, colour, height,
                hover_label=hover_label)


def stacked_share(
    df: pd.DataFrame, index: str, columns: list[str], height: int = 320,
) -> go.Figure:
    """Composition across groups. A 2px surface gap separates the segments so
    adjacent fills never read as one block."""
    palette = [BLUE, ORANGE, AQUA]
    fig = go.Figure()
    for i, col in enumerate(columns):
        fig.add_trace(go.Bar(
            x=df[index], y=df[col], name=col,
            marker=dict(color=palette[i % len(palette)],
                        line=dict(color=SURFACE, width=2)),
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:,.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", yaxis=dict(title="Share (%)"),
        xaxis=dict(tickfont=dict(size=12, color=INK)),
        margin=dict(l=8, r=8, t=34, b=8), bargap=0.34,
    )
    return apply(fig, height, showlegend=True)
