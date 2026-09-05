"""Risk and opportunity — where the book sits, and where the effort should go."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Risk & opportunity")

from components import charts, data, ui                              # noqa: E402
from components.theme import BLUE, ORANGE, QUADRANT_COLOURS, RISK_COLOURS  # noqa: E402
from customer_intelligence.analytics.opportunity import (            # noqa: E402
    COMPONENT_DEFINITIONS, QUADRANT_MEANING,
)
from customer_intelligence.config import CHURN_BANDS, OPPORTUNITY_WEIGHTS, band  # noqa: E402

df = data.customers()
cards = data.model_cards()

ui.page_header(
    "Risk & opportunity",
    "Which customers are changing, and which are worth the effort",
    "Attrition risk and relationship opportunity are scored separately and on "
    "purpose. Collapsing them into a single ranking would hide the case that "
    "matters most — a valuable relationship that is quietly ending.",
)
ui.synthetic_notice()

# ------------------------------------------------------------------ risk ----
ui.section("Attrition risk across the book")

churn = cards["churn"]
r1, r2 = st.columns([1.3, 1], gap="large")
with r1:
    st.plotly_chart(
        charts.histogram(
            df["churn_probability"], bins=50, x_title="Modelled attrition probability",
            cut=churn["threshold"], cut_label="Operating threshold", height=310,
        ),
        use_container_width=True,
    )
    ui.note(
        f"The dotted line is the operating threshold ({churn['threshold']:.1%}), "
        "set to flag the top 10% of the book rather than at a probability of "
        "0.5. With a base rate near 10%, a 0.5 cut-off would flag almost "
        "nobody. The threshold is a capacity decision — how many customers a "
        "team can actually contact — and is stated as one."
    )
with r2:
    bands = df["churn_probability"].apply(lambda p: band(p, CHURN_BANDS))
    counts = bands.value_counts().reindex([b for b, _ in CHURN_BANDS]).fillna(0).astype(int)
    st.plotly_chart(charts.banded_bar(counts, RISK_COLOURS, height=310, total=len(df)),
                    use_container_width=True)

st.markdown("#### Which segments carry the risk")
risk_by_seg = df.groupby("segment", observed=True).agg(
    customers=("customer_id", "size"),
    mean_risk=("churn_probability", "mean"),
    elevated=("churn_probability", lambda s: (s >= 0.25).sum()),
    balance_at_risk=("avg_monthly_balance",
                     lambda s: s[df.loc[s.index, "churn_probability"] >= 0.25].sum()),
).reset_index().sort_values("mean_risk", ascending=False)

rs1, rs2 = st.columns([1, 1], gap="large")
with rs1:
    st.plotly_chart(
        charts.hbar(list(risk_by_seg["segment"]), list(risk_by_seg["mean_risk"]),
                    value_fmt="{:.1%}", height=320, hover_label="Mean attrition risk"),
        use_container_width=True,
    )
with rs2:
    st.plotly_chart(
        charts.hbar(list(risk_by_seg["segment"]), list(risk_by_seg["balance_at_risk"]),
                    value_fmt="{:,.0f}", height=320, colour=ORANGE,
                    hover_label="Balances held by at-risk customers"),
        use_container_width=True,
    )
ui.note(
    "Left: how likely a segment is to leave. Right: how much balance sits with "
    "the at-risk customers inside it. They rank differently, and the right-hand "
    "chart is the one that should drive where scarce human attention goes."
)

# ----------------------------------------------------------- opportunity ----
ui.section(
    "Relationship opportunity",
    "A published management heuristic, not a fitted model and not a revenue "
    "forecast. Every component is a percentile rank within the book, so the "
    "score is always a relative statement.",
)

o1, o2 = st.columns([1.3, 1], gap="large")
with o1:
    st.plotly_chart(
        charts.histogram(df["opportunity_score"], bins=40, colour=BLUE,
                         x_title="Relationship opportunity score", height=300),
        use_container_width=True,
    )
with o2:
    ui.panel("How the score is built", [
        (name, f"{w:.0%}") for name, w in OPPORTUNITY_WEIGHTS.as_dict().items()
    ])

with st.expander("What each component measures, and why it is in the score"):
    for name, definition in COMPONENT_DEFINITIONS.items():
        st.markdown(f"**{name}** ({OPPORTUNITY_WEIGHTS.as_dict()[name]:.0%}) — {definition}")
    st.markdown(
        "\nThe weights are a stated judgement, published so they can be argued "
        "with and changed deliberately. They were not fitted to anything: there "
        "is no ground truth for 'opportunity', and a model that appeared to "
        "find one would be fitting to whatever the organisation happened to "
        "sell last year."
    )

st.markdown("#### Average component scores by segment")
comp_cols = [c for c in df.columns if c.startswith("opp_")]
comp_means = df.groupby("segment", observed=True)[comp_cols].mean().reset_index()
comp_long = comp_means.melt(id_vars="segment", var_name="component", value_name="score")
comp_long["component"] = comp_long["component"].str.replace("opp_", "").str.replace("_", " ").str.capitalize()
pivot = comp_long.pivot(index="segment", columns="component", values="score").round(1)
st.dataframe(pivot, use_container_width=True)

# -------------------------------------------------------------- matrix -----
ui.section(
    "The risk / opportunity matrix",
    "Every customer placed on both axes at once. The quadrant names are "
    "instructions rather than descriptions — the purpose of the matrix is to "
    "say what to do with each group.",
)

m1, m2 = st.columns([1.5, 1], gap="large")
with m1:
    st.plotly_chart(
        charts.risk_opportunity_matrix(df["opportunity_score"], df["churn_probability"],
                                       height=460),
        use_container_width=True,
    )
    ui.note(
        "Drawn as a density surface, not a scatter: fifty thousand overlapping "
        "points make a blob that reads as one large mass wherever it is "
        "densest, which is the opposite of what a reader needs here."
    )
with m2:
    quad = df.groupby("quadrant", observed=True).agg(
        customers=("customer_id", "size"),
        balances=("avg_monthly_balance", "sum"),
        mean_risk=("churn_probability", "mean"),
        mean_opportunity=("opportunity_score", "mean"),
    ).reset_index()

    for name in ["Retain", "Grow", "Monitor", "Develop"]:
        r = quad[quad["quadrant"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        st.markdown(
            f'<div class="ci-panel" style="padding:.85rem 1rem;margin-bottom:.6rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<strong style="color:{QUADRANT_COLOURS[name]};font-size:1rem;">{name}</strong>'
            f'<span style="color:#5a6169;font-variant-numeric:tabular-nums;">'
            f'{int(r["customers"]):,} · {r["customers"] / len(df):.0%}</span></div>'
            f'<div style="font-size:.8rem;color:#5a6169;margin-top:.35rem;line-height:1.45;">'
            f'{QUADRANT_MEANING[name]}</div>'
            f'<div style="font-size:.78rem;color:#8b9199;margin-top:.45rem;">'
            f'Balances {ui.money(r["balances"])} · mean risk {r["mean_risk"]:.0%} · '
            f'mean opportunity {r["mean_opportunity"]:.0f}</div></div>',
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------- shortlist ----
ui.section(
    "The shortlist",
    "The Retain quadrant, ordered by balances at risk. This is the list a "
    "relationship team would work through first.",
)

shortlist = df[df["quadrant"] == "Retain"].nlargest(50, "avg_monthly_balance")[[
    "customer_id", "segment", "avg_monthly_balance", "churn_probability",
    "opportunity_score", "product_count", "complaints", "action", "priority",
]].copy()
shortlist.columns = [
    "Customer", "Segment", "Avg balance", "Attrition risk", "Opportunity",
    "Products", "Complaints", "Suggested action", "Priority",
]
st.dataframe(
    shortlist.style.format({
        "Avg balance": "{:,.0f}", "Attrition risk": "{:.0%}",
        "Opportunity": "{:.0f}", "Products": "{:.0f}", "Complaints": "{:.0f}",
    }),
    use_container_width=True, hide_index=True, height=420,
)
ui.note(
    "Ordered by balance at risk rather than by probability. The highest-"
    "probability customers are often the smallest relationships, and a list "
    "sorted by risk alone sends a team to spend its scarcest resource on its "
    "least consequential cases."
)
