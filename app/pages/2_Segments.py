"""Customer segments — RFM, and what it does and does not tell you."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Segments")

from components import charts, data, ui                                      # noqa: E402
from components.theme import AQUA, BLUE, ORANGE                              # noqa: E402
from customer_intelligence.analytics.retail import SEGMENT_DESCRIPTIONS      # noqa: E402

df = data.customers()
profiles = data.load("segment_profiles")

ui.page_header(
    "Customer segments",
    "Who these accounts are",
    "Segmentation exists to make a book of six thousand trading relationships "
    "small enough to reason about. This is RFM — recency, frequency, monetary "
    "value — which is the standard retail rule set, scored as quintiles within "
    "this population.",
)
ui.source_notice()

ui.section("The segments")
o1, o2 = st.columns(2, gap="large")
with o1:
    st.markdown("#### Share of accounts")
    st.plotly_chart(
        charts.hbar(list(profiles["segment"]), list(profiles["share_of_customers"]),
                    value_fmt="{:.1f}", label_suffix="%", height=340,
                    hover_label="Share of accounts"),
        width="stretch",
    )
with o2:
    st.markdown("#### Share of revenue")
    st.plotly_chart(
        charts.hbar(list(profiles["segment"]), list(profiles["share_of_revenue"]),
                    value_fmt="{:.1f}", label_suffix="%", height=340, colour=ORANGE,
                    hover_label="Share of revenue"),
        width="stretch",
    )

champ = profiles[profiles["segment"] == "Champions"]
lost = profiles[profiles["segment"] == "Lost"]
if not champ.empty and not lost.empty:
    ui.note(
        f"<b>Champions</b> are {champ['share_of_customers'].iat[0]:.1f}% of accounts "
        f"and {champ['share_of_revenue'].iat[0]:.1f}% of revenue. <b>Lost</b> are "
        f"{lost['share_of_customers'].iat[0]:.1f}% of accounts and "
        f"{lost['share_of_revenue'].iat[0]:.1f}%. That asymmetry is the entire "
        "argument for segmenting at all — and it is why the account-manager "
        "caseload is built from the left-hand chart weighted by the right-hand one."
    )

ui.section(
    "What each segment looks like",
    "The table an account team actually uses: not cluster centroids, but the "
    "handful of measures a wholesale relationship is managed on.",
)
display = profiles[[
    "segment", "customers", "share_of_customers", "avg_revenue", "avg_orders",
    "avg_order_value", "avg_months_since_order", "active_months",
    "distinct_categories", "avg_cadence", "return_rate",
]].copy()
display.columns = ["Segment", "Accounts", "Share %", "Avg revenue", "Avg orders",
                   "Avg order value", "Months since order", "Active months",
                   "Categories", "Usual gap", "Return rate"]
st.dataframe(
    display.style.format({
        "Accounts": "{:,.0f}", "Share %": "{:.1f}%", "Avg revenue": "£{:,.0f}",
        "Avg orders": "{:.1f}", "Avg order value": "£{:,.0f}",
        "Months since order": "{:.1f}", "Active months": "{:.1f}",
        "Categories": "{:.1f}", "Usual gap": "{:.1f}", "Return rate": "{:.1%}",
    }),
    width="stretch", hide_index=True,
)

with st.expander("What each segment means, in business language"):
    for _, r in profiles.iterrows():
        st.markdown(f"**{r['segment']}** — {r['customers']:,.0f} accounts "
                    f"({r['share_of_customers']:.1f}%). "
                    f"{SEGMENT_DESCRIPTIONS.get(r['segment'], '')}")

ui.section(
    "Why RFM, and not a clustering",
    "The synthetic version of this project ran K-means alongside its rules and "
    "found a silhouette of 0.15 — no natural clusters. The same is true here, "
    "and for the same reason.",
)
st.markdown(
    """
Customers sit on continuous gradients of value, frequency and recency. There is
no seam in the data where one group ends and another begins, so **any**
partition of them is a decision somebody made rather than a structure waiting to
be discovered.

Given that, the question is not "which method finds the true segments" — none
of them will, because there aren't any. The question is which partition is most
useful to operate, and that argues for published rules over a fitted clustering
on three counts:

- **Stability.** An account lands in the same segment next month unless its
  behaviour changed. A re-fitted clustering can move accounts because the
  algorithm re-initialised.
- **Explicability.** "Ordered recently, orders often, spends a lot" is a
  sentence an account manager can act on and argue with. "Cluster 4" is not.
- **Continuity.** The thresholds are quintiles of this population, published in
  `analytics/retail.py`. When the book changes, the definition moves in a way
  that can be read.

RFM is not chosen here because it is sophisticated. It is chosen because it is
the right shape for the decision, and the more sophisticated alternative was
tested and had nothing to add.
"""
)

ui.section("Compare segments on any measure")
metric = st.selectbox(
    "Measure",
    ["revenue", "invoices", "avg_order_value", "months_since_last_order",
     "active_months", "distinct_categories", "distinct_products",
     "opportunity_score", "lapse_risk", "return_rate", "cadence_overdue"],
    format_func=lambda c: {
        "revenue": "Revenue in window", "invoices": "Orders placed",
        "avg_order_value": "Average order value",
        "months_since_last_order": "Months since last order",
        "active_months": "Months with an order",
        "distinct_categories": "Categories bought from",
        "distinct_products": "Distinct products bought",
        "opportunity_score": "Relationship opportunity",
        "lapse_risk": "Lapse risk", "return_rate": "Return rate",
        "cadence_overdue": "Past their usual gap by",
    }[c],
)
by_segment = df.groupby("segment", observed=True)[metric].mean().reset_index()
by_segment = by_segment.sort_values(metric, ascending=False)
fmt = "{:.1%}" if metric in ("lapse_risk", "return_rate") else "{:,.1f}"
st.plotly_chart(
    charts.hbar(list(by_segment["segment"]), list(by_segment[metric]),
                value_fmt=fmt, height=340, hover_label="Segment average"),
    width="stretch",
)
