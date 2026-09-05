"""Customer segments — a rule set, and the clustering that checks it."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Customer segments")

from components import charts, data, ui                     # noqa: E402
from components.theme import AQUA, BLUE, ORANGE             # noqa: E402
from customer_intelligence.analytics.segmentation import SEGMENT_DESCRIPTIONS  # noqa: E402

df = data.customers()
profiles = data.load("segment_profiles")

ui.page_header(
    "Customer segments",
    "Who the customers are",
    "Segmentation exists to make a book of fifty thousand relationships small "
    "enough to reason about. Two methods are built here, because they answer "
    "different questions — and the comparison between them is the deliverable.",
)
ui.synthetic_notice()

# ------------------------------------------------------------- overview -----
ui.section("The segments")

o1, o2 = st.columns([1, 1], gap="large")
with o1:
    st.markdown("#### Share of customers")
    st.plotly_chart(
        charts.hbar(list(profiles["segment"]), list(profiles["share_of_customers"]),
                    value_fmt="{:.1f}", label_suffix="%", height=330,
                    hover_label="Share of customers"),
        use_container_width=True,
    )
with o2:
    st.markdown("#### Share of balances")
    st.plotly_chart(
        charts.hbar(list(profiles["segment"]), list(profiles["share_of_balances"]),
                    value_fmt="{:.1f}", label_suffix="%", height=330, colour=ORANGE,
                    hover_label="Share of balances"),
        use_container_width=True,
    )

top = profiles.loc[profiles["share_of_balances"].idxmax()]
ui.note(
    f"The two charts are the point. <b>{top['segment']}</b> is "
    f"{top['share_of_customers']:.1f}% of customers and "
    f"{top['share_of_balances']:.1f}% of balances. Any segmentation where the "
    "two bars have the same shape has not found anything — it has re-drawn the "
    "customer count twice."
)

# -------------------------------------------------------------- profiles ----
ui.section(
    "What each segment looks like",
    "The table an operating team actually uses: not cluster centroids, but the "
    "handful of measures a relationship is managed on.",
)

display = profiles[[
    "segment", "customers", "share_of_customers", "avg_balance", "avg_products",
    "avg_monthly_transactions", "avg_logins", "digital_share",
    "avg_loan_exposure", "arrears_rate", "avg_tenure",
]].copy()
display.columns = [
    "Segment", "Customers", "Share %", "Avg balance", "Products",
    "Txns / month", "Logins / month", "Digital share", "Loan exposure",
    "Arrears rate", "Tenure (months)",
]
st.dataframe(
    display.style.format({
        "Customers": "{:,.0f}", "Share %": "{:.1f}%", "Avg balance": "{:,.0f}",
        "Products": "{:.1f}", "Txns / month": "{:.1f}", "Logins / month": "{:.1f}",
        "Digital share": "{:.0%}", "Loan exposure": "{:,.0f}",
        "Arrears rate": "{:.1%}", "Tenure (months)": "{:.0f}",
    }),
    use_container_width=True, hide_index=True,
)

with st.expander("What each segment means, in business language"):
    for _, r in profiles.iterrows():
        st.markdown(
            f"**{r['segment']}** — {r['customers']:,.0f} customers "
            f"({r['share_of_customers']:.1f}%). "
            f"{SEGMENT_DESCRIPTIONS.get(r['segment'], '')}"
        )

# ------------------------------------------------------------ comparison ----
ui.section(
    "Rules against clustering",
    "K-means was run on the same customers, over the same behavioural measures, "
    "with no knowledge of the rules.",
)

diagnostics = data.load("cluster_diagnostics")
best = diagnostics.loc[diagnostics["silhouette"].idxmax()]

c1, c2 = st.columns([1, 1.25], gap="large")
with c1:
    st.markdown("#### How well-separated are the clusters?")
    st.plotly_chart(
        charts.trend_lines(diagnostics, "k", {"silhouette": "Silhouette score"},
                           y_title="Silhouette", value_fmt=":.3f", height=270),
        use_container_width=True,
    )
    ui.panel("Clustering diagnostics", [
        ("Best k by silhouette", f"{int(best['k'])}"),
        ("Silhouette at best k", f"{best['silhouette']:.3f}"),
        ("k used in production", "6"),
    ])

with c2:
    st.markdown("#### Where the two methods agree")
    crosstab = data.load("segment_cluster_crosstab").set_index("segment")
    st.dataframe(crosstab, use_container_width=True)

ui.note(
    f"The silhouette score peaks at {best['silhouette']:.3f}. That is weak — a "
    "well-separated clustering scores above about 0.5 — and it is the honest "
    "finding rather than a disappointment: this customer book does not fall "
    "into naturally distinct groups. Customers sit on continuous gradients of "
    "wealth, engagement and credit appetite, and any partition of them is a "
    "decision someone made, not a structure waiting to be discovered. "
    "<b>That is precisely the argument for the rule set.</b> If the segments "
    "are a management choice either way, they should be the choice that is "
    "stable month to month, explicable to the people who act on it, and "
    "unchanged by a re-run — which a clustering is not."
)

# ----------------------------------------------------------- exploration ----
ui.section("Compare segments on any measure")

metric = st.selectbox(
    "Measure",
    ["avg_monthly_balance", "product_count", "avg_monthly_transactions",
     "avg_monthly_logins", "digital_share", "loan_exposure",
     "opportunity_score", "churn_probability", "tenure_months"],
    format_func=lambda c: {
        "avg_monthly_balance": "Average monthly balance",
        "product_count": "Products held",
        "avg_monthly_transactions": "Transactions per month",
        "avg_monthly_logins": "Logins per month",
        "digital_share": "Share of activity digital",
        "loan_exposure": "Loan exposure",
        "opportunity_score": "Relationship opportunity score",
        "churn_probability": "Attrition risk",
        "tenure_months": "Tenure (months)",
    }[c],
)

by_segment = df.groupby("segment", observed=True)[metric].mean().reset_index()
by_segment = by_segment.sort_values(metric, ascending=False)
fmt = "{:.1%}" if metric in ("digital_share", "churn_probability") else "{:,.1f}"
st.plotly_chart(
    charts.hbar(list(by_segment["segment"]), list(by_segment[metric]),
                value_fmt=fmt, height=330, hover_label="Segment average"),
    use_container_width=True,
)
