"""One visual language for the whole application.

Colour is assigned by the job it does, never by what is available:

  * sequential blue   — magnitude on a continuous scale
  * blue / orange     — two named series that must be told apart
  * status ramp       — reserved for state (good / warning / serious / critical)
                        and never reused as a series colour

Segment charts are deliberately single-hue: the segment name is already on the
axis, so colouring each bar differently would encode nothing and spend the
categorical palette on decoration. The palette was validated for colour-vision
deficiency against this application's surface before use.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------- surfaces ----
SURFACE = "#ffffff"
PAGE = "#f6f7f9"
INK = "#14171a"
INK_SECONDARY = "#5a6169"
INK_MUTED = "#8b9199"
GRID = "#e8eaed"
BORDER = "#e3e6ea"
AXIS = "#c9ced4"

# ------------------------------------------------------------ categorical --
# Slots 1–3 of the validated order. Three is the cap for all-pairs forms
# (scatter, matrices); beyond that identity folds into "Other" or facets.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE, ORANGE, AQUA = SERIES

# ------------------------------------------------------------- sequential --
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]

# Ordinal steps start no lighter than 250 so the lightest still reads on white.
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

# ------------------------------------------------------------ diverging ----
DIVERGING = [[0.0, "#0d366b"], [0.5, "#f0efec"], [1.0, "#d03b3b"]]

# ---------------------------------------------------------------- status ----
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

RISK_COLOURS = {
    "Low": STATUS["good"], "Moderate": STATUS["warning"],
    "Elevated": STATUS["serious"], "High": STATUS["critical"],
}

QUADRANT_COLOURS = {
    "Retain": STATUS["critical"], "Grow": STATUS["good"],
    "Monitor": STATUS["warning"], "Develop": INK_MUTED,
}

FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif')


def register_template() -> None:
    """Install the house Plotly template and make it the default."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        margin=dict(l=8, r=8, t=28, b=8),
        hoverlabel=dict(
            bgcolor=INK, bordercolor=INK, font=dict(color="#ffffff", size=12, family=FONT),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=12, color=INK_SECONDARY),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, linecolor=AXIS, linewidth=1, ticks="outside",
            ticklen=4, tickcolor=AXIS, tickfont=dict(size=12, color=INK_MUTED),
            title=dict(font=dict(size=12, color=INK_MUTED)),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
            linecolor="rgba(0,0,0,0)", ticks="", tickfont=dict(size=12, color=INK_MUTED),
            title=dict(font=dict(size=12, color=INK_MUTED)),
        ),
        colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]),
    )
    pio.templates["ci"] = template
    pio.templates.default = "ci"


def apply(fig: go.Figure, height: int = 300, showlegend: bool | None = None) -> go.Figure:
    """Final pass every figure goes through before it is rendered."""
    fig.update_layout(height=height, template="ci")
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


CSS = f"""
<style>
  .stApp {{ background: {PAGE}; }}
  .block-container {{ padding-top: 4.6rem; padding-bottom: 4rem; max-width: 1320px; }}

  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.011em; font-weight: 600; }}
  h1 {{ font-size: 1.85rem; margin-bottom: .15rem; }}
  h2 {{ font-size: 1.22rem; margin-top: 2.1rem; margin-bottom: .5rem; }}
  h3 {{ font-size: 1.02rem; }}

  .ci-eyebrow {{
    text-transform: uppercase; letter-spacing: .1em; font-size: .68rem;
    font-weight: 600; color: {INK_MUTED}; margin-bottom: .35rem;
  }}
  .ci-lead {{ color: {INK_SECONDARY}; font-size: .95rem; line-height: 1.55; max-width: 76ch; }}
  .ci-note {{
    color: {INK_MUTED}; font-size: .82rem; line-height: 1.5;
    border-left: 2px solid {BORDER}; padding-left: .75rem; margin: .6rem 0 1.1rem;
    max-width: 82ch;
  }}

  /* --- stat tiles --------------------------------------------------------- */
  .ci-tiles {{ display: grid; gap: .8rem; margin: .4rem 0 .6rem; }}
  .ci-tile {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: .95rem 1.05rem; min-height: 104px;
  }}
  .ci-tile-label {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
    color: {INK_MUTED}; font-weight: 600; margin-bottom: .42rem;
  }}
  .ci-tile-value {{
    font-size: 1.72rem; font-weight: 600; color: {INK}; line-height: 1.05;
    font-variant-numeric: tabular-nums; letter-spacing: -.02em;
  }}
  .ci-tile-sub {{ font-size: .78rem; color: {INK_SECONDARY}; margin-top: .32rem; line-height: 1.4; }}

  /* --- panels ------------------------------------------------------------- */
  .ci-panel {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 1.1rem 1.2rem; margin-bottom: .9rem;
  }}
  .ci-panel h4 {{ margin: 0 0 .5rem; font-size: .95rem; }}

  .ci-kv {{ display: flex; justify-content: space-between; gap: 1rem;
            padding: .38rem 0; border-bottom: 1px solid #f1f2f4; font-size: .87rem; }}
  .ci-kv:last-child {{ border-bottom: none; }}
  .ci-kv-k {{ color: {INK_MUTED}; }}
  .ci-kv-v {{ color: {INK}; font-weight: 500; font-variant-numeric: tabular-nums; text-align: right; }}

  /* --- badges ------------------------------------------------------------- */
  .ci-badge {{
    display: inline-block; padding: .2rem .55rem; border-radius: 4px;
    font-size: .72rem; font-weight: 600; letter-spacing: .02em;
    border: 1px solid transparent;
  }}

  .ci-synthetic {{
    background: #fffbeb; border: 1px solid #fde68a; color: #78500a;
    padding: .55rem .85rem; border-radius: 6px; font-size: .8rem;
    margin-bottom: 1.4rem; line-height: 1.5;
  }}

  .ci-rule {{ border: none; border-top: 1px solid {BORDER}; margin: 1.8rem 0 1.1rem; }}

  [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
  [data-testid="stSidebarNav"] {{ padding-top: .5rem; }}
  section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 6px; }}
</style>
"""
