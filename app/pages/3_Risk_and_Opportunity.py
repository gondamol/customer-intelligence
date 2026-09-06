"""Risk and opportunity — where the book sits, and where effort should go."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Risk & opportunity")

from components import charts, data, ui                                      # noqa: E402
from components.theme import BLUE, ORANGE, QUADRANT_COLOURS                  # noqa: E402
from customer_intelligence.analytics.opportunity import (                    # noqa: E402
    RETAIL_COMPONENT_DEFINITIONS, QUADRANT_MEANING,
)
from customer_intelligence.config import OPPORTUNITY_WEIGHTS                 # noqa: E402

df = data.customers()
cards = data.model_cards()
scored = df[df["lapse_risk"].notna()]

ui.page_header(
    "Risk & opportunity",
    "Which accounts are changing, and which are worth the effort",
    "Lapse risk and relationship opportunity are scored separately and on "
    "purpose. Collapsing them into one ranking would hide the case that matters "
    "most — a valuable account that is quietly going quiet.",
)
ui.source_notice()

ui.section("Lapse risk across the book")
lapse = cards["lapse"]
r1, r2 = st.columns([1.3, 1], gap="large")
with r1:
    st.plotly_chart(
        charts.histogram(scored["lapse_risk"], bins=40,
                         x_title="Modelled probability of no order next quarter",
                         cut=lapse["threshold"], cut_label="Operating threshold",
                         height=310),
        width="stretch",
    )
    ui.note(
        f"The dotted line is the operating threshold ({lapse['threshold']:.1%}), "
        "set to flag the top quarter of scoreable accounts rather than at a "
        "probability of 0.5. The threshold is a capacity decision — how many "
        "accounts a team can actually work — and is stated as one."
    )
with r2:
    ui.panel("What is being predicted", [
        ("Outcome", "No order at all next quarter"),
        ("Population", "Accounts with ≥3 active months"),
        ("Base rate", ui.pct(lapse["base_rate"])),
        ("ROC-AUC", f"{lapse['metrics']['roc_auc']:.3f}"),
        ("Lift at threshold", f"{lapse['metrics']['lift']:.1f}×"),
        ("Scoreable accounts", f"{len(scored):,} of {len(df):,}"),
    ])
    ui.note(
        "This is a <b>behavioural</b> lapse, not a cancelled contract — a "
        "wholesaler cannot close an account, it can only stop ordering. That "
        "distinction is the reason the project also runs against a telco dataset "
        "with a contractual churn flag: see <b>Data quality &amp; governance</b>."
    )

st.markdown("#### Which segments carry the risk")
risk_by_seg = scored.groupby("segment", observed=True).agg(
    accounts=("customer_id", "size"),
    mean_risk=("lapse_risk", "mean"),
    revenue_at_risk=("revenue", lambda s: s[scored.loc[s.index, "lapse_risk"] >= 0.5].sum()),
).reset_index().sort_values("mean_risk", ascending=False)

rs1, rs2 = st.columns(2, gap="large")
with rs1:
    st.plotly_chart(
        charts.hbar(list(risk_by_seg["segment"]), list(risk_by_seg["mean_risk"]),
                    value_fmt="{:.0%}", height=320, hover_label="Mean lapse risk"),
        width="stretch",
    )
with rs2:
    st.plotly_chart(
        charts.hbar(list(risk_by_seg["segment"]), list(risk_by_seg["revenue_at_risk"]),
                    value_fmt="£{:,.0f}", height=320, colour=ORANGE,
                    hover_label="Revenue held by at-risk accounts"),
        width="stretch",
    )
ui.note(
    "Left: how likely a segment is to go quiet. Right: how much revenue sits "
    "with the at-risk accounts inside it. They rank differently, and the "
    "right-hand chart is the one that should decide where scarce account-manager "
    "time goes."
)

ui.section(
    "Relationship opportunity",
    "A published management heuristic, not a fitted model and not a revenue "
    "forecast. Every component is a percentile rank within this book, so the "
    "score is always a relative statement.",
)
o1, o2 = st.columns([1.3, 1], gap="large")
with o1:
    st.plotly_chart(
        charts.histogram(df["opportunity_score"], bins=40,
                         x_title="Relationship opportunity score", height=300),
        width="stretch",
    )
with o2:
    comp_cols = [c for c in df.columns if c.startswith("opp_")]
    names = list(RETAIL_COMPONENT_DEFINITIONS)
    ui.panel("How the score is built",
             list(zip(names, [f"{w:.0%}" for w in OPPORTUNITY_WEIGHTS.as_dict().values()])))

with st.expander("What each component measures, and why it is in the score"):
    for (name, definition), weight in zip(RETAIL_COMPONENT_DEFINITIONS.items(),
                                          OPPORTUNITY_WEIGHTS.as_dict().values()):
        st.markdown(f"**{name}** ({weight:.0%}) — {definition}")
    st.markdown(
        "\nThe weights are a stated judgement, published so they can be argued "
        "with and changed deliberately. They were not fitted to anything: there "
        "is no ground truth for 'opportunity', and a model that appeared to find "
        "one would be fitting to whatever the business happened to sell last year."
    )

st.markdown("#### Average component scores by segment")
comp_means = df.groupby("segment", observed=True)[comp_cols].mean().round(1)
comp_means.columns = [c.replace("opp_", "").replace("_", " ").capitalize()
                      for c in comp_means.columns]
st.dataframe(comp_means, width="stretch")

ui.section(
    "The risk / opportunity matrix",
    "Every scoreable account on both axes at once. The quadrant names are "
    "instructions rather than descriptions.",
)
m1, m2 = st.columns([1.5, 1], gap="large")
with m1:
    st.plotly_chart(
        charts.risk_opportunity_matrix(scored["opportunity_score"], scored["lapse_risk"],
                                       risk_cut=0.5, height=460),
        width="stretch",
    )
with m2:
    quad = df.groupby("quadrant", observed=True).agg(
        accounts=("customer_id", "size"), revenue=("revenue", "sum"),
        mean_risk=("lapse_risk", "mean"), mean_opportunity=("opportunity_score", "mean"),
    ).reset_index()
    for name in ["Retain", "Grow", "Monitor", "Develop", "Not scored"]:
        r = quad[quad["quadrant"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        st.markdown(
            f'<div class="ci-panel" style="padding:.85rem 1rem;margin-bottom:.6rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<strong style="color:{QUADRANT_COLOURS[name]};font-size:1rem;">{name}</strong>'
            f'<span style="color:#5a6169;font-variant-numeric:tabular-nums;">'
            f'{int(r["accounts"]):,} · {r["accounts"]/len(df):.0%}</span></div>'
            f'<div style="font-size:.8rem;color:#5a6169;margin-top:.35rem;line-height:1.45;">'
            f'{QUADRANT_MEANING[name]}</div>'
            f'<div style="font-size:.78rem;color:#8b9199;margin-top:.45rem;">'
            f'Revenue {ui.money(r["revenue"])}'
            + (f' · mean risk {r["mean_risk"]:.0%}' if pd.notna(r["mean_risk"]) else "")
            + f' · mean opportunity {r["mean_opportunity"]:.0f}</div></div>',
            unsafe_allow_html=True,
        )

ui.section(
    "The shortlist",
    "The Retain quadrant, ordered by revenue at risk. This is the list an "
    "account team would work through first.",
)
shortlist = df[df["quadrant"] == "Retain"].nlargest(50, "revenue")[[
    "customer_id", "country", "segment", "revenue", "lapse_risk", "opportunity_score",
    "months_since_last_order", "cadence_overdue", "action", "priority",
]].copy()
shortlist.columns = ["Account", "Country", "Segment", "Revenue", "Lapse risk",
                     "Opportunity", "Months silent", "Past usual gap", "Suggested action",
                     "Priority"]
st.dataframe(
    shortlist.style.format({
        "Revenue": "£{:,.0f}", "Lapse risk": "{:.0%}", "Opportunity": "{:.0f}",
        "Months silent": "{:.0f}", "Past usual gap": "{:.1f}×",
    }),
    width="stretch", hide_index=True, height=420,
)
ui.note(
    "Ordered by revenue at risk rather than by probability. The highest-"
    "probability accounts are often the smallest ones, and a list sorted by risk "
    "alone sends a team to spend its scarcest resource on its least "
    "consequential cases."
)
