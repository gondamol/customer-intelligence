"""Customer 360 — everything known about one relationship, on one screen."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Customer 360")

from components import charts, data, ui                                  # noqa: E402
from components.theme import BLUE, ORANGE                                # noqa: E402
from customer_intelligence.config import CHURN_BANDS, OPPORTUNITY_BANDS, PROPENSITY_BANDS, band  # noqa: E402

df = data.customers()
THRESHOLDS = data.run_summary().get("thresholds", {})

ui.page_header(
    "Customer 360",
    "One relationship, end to end",
    "Every attribute, behaviour, score and suggested action for a single "
    "customer — assembled from seven source tables into one analytical view.",
)
ui.synthetic_notice()

# ---------------------------------------------------------------- picker ----
st.markdown("#### Select a customer")
c1, c2, c3 = st.columns([1.1, 1, 1])
with c1:
    segment_filter = st.selectbox(
        "Filter by segment", ["All segments"] + sorted(df["segment"].unique()))
with c2:
    quadrant_filter = st.selectbox(
        "Filter by quadrant", ["All quadrants"] + sorted(df["quadrant"].unique()))
with c3:
    sort_by = st.selectbox(
        "Order by",
        ["Attrition risk (highest)", "Opportunity (highest)",
         "Balance (highest)", "Customer ID"])

pool = df
if segment_filter != "All segments":
    pool = pool[pool["segment"] == segment_filter]
if quadrant_filter != "All quadrants":
    pool = pool[pool["quadrant"] == quadrant_filter]

sort_map = {
    "Attrition risk (highest)": ("churn_probability", False),
    "Opportunity (highest)": ("opportunity_score", False),
    "Balance (highest)": ("avg_monthly_balance", False),
    "Customer ID": ("customer_id", True),
}
col, asc = sort_map[sort_by]
pool = pool.sort_values(col, ascending=asc)

if pool.empty:
    st.info("No customers match that combination of filters.")
    st.stop()

choice = st.selectbox(
    f"Customer ({len(pool):,} match)",
    pool["customer_id"].head(400).tolist(),
    format_func=lambda cid: (
        f"{cid}  ·  {pool.loc[pool.customer_id == cid, 'segment'].iloc[0]}"
        f"  ·  risk {pool.loc[pool.customer_id == cid, 'churn_probability'].iloc[0]:.0%}"
    ),
)
row = df[df["customer_id"] == choice].iloc[0]

ui.rule()

# --------------------------------------------------------------- summary ----
risk_band = band(row["churn_probability"], CHURN_BANDS)
opp_band = band(row["opportunity_score"], OPPORTUNITY_BANDS)

st.markdown(f"### {choice}")
st.markdown(
    f'<div style="margin:-.35rem 0 1rem;">{ui.badge(row["segment"])} '
    f'{ui.risk_badge(risk_band)} {ui.badge(row["quadrant"])} '
    f'{ui.priority_badge(row["priority"])}</div>',
    unsafe_allow_html=True,
)

ui.tiles([
    {"label": "Average monthly balance", "value": ui.money(row["avg_monthly_balance"])},
    {"label": "Attrition risk", "value": f"{row['churn_probability']:.0%}",
     "sub": f"{risk_band} band"},
    {"label": "Relationship opportunity", "value": f"{row['opportunity_score']:.0f}",
     "sub": f"{opp_band} · out of 100"},
    {"label": "Products held", "value": f"{int(row['product_count'])}",
     "sub": f"{int(row['account_count'])} accounts"},
    {"label": "Months active", "value": f"{int(row['active_months'])} of 12",
     "sub": f"Last activity {int(row['months_since_last_activity'])} months ago"},
])

# ----------------------------------------------------------------- panels ---
p1, p2, p3 = st.columns(3, gap="medium")
with p1:
    ui.panel("Profile", [
        ("Age", ui.num(row["age"])),
        ("Gender", row["gender"]),
        ("Region", row["region"]),
        ("Employment", row["employment_type"]),
        ("Income band", row["income_band"]),
        ("Declared monthly income", ui.money(row["monthly_income"])),
        ("Tenure", f"{int(row['tenure_months'])} months"),
    ])
with p2:
    ui.panel("Behaviour in the window", [
        ("Transactions", ui.num(row["transaction_count"])),
        ("Transaction value", ui.money(row["transaction_value"])),
        ("Average transaction", ui.money(row["avg_transaction_value"])),
        ("Logins per month", ui.num(row["avg_monthly_logins"], 1)),
        ("Share of activity digital", ui.pct(row["digital_share"])),
        ("Channels used", ui.num(row["max_channels_used"])),
        ("Outflow against inflow", ui.num(row["outflow_to_inflow"], 2)),
    ])
with p3:
    ui.panel("Holdings and service", [
        ("Loans held", ui.num(row["loan_count"])),
        ("Loan exposure", ui.money(row["loan_exposure"])),
        ("Ever in arrears", "Yes" if row["ever_in_arrears"] else "No"),
        ("Service contacts", ui.num(row["service_interactions"])),
        ("Complaints", ui.num(row["complaints"])),
        ("Unresolved contacts", ui.num(row["unresolved_interactions"])),
        ("Average satisfaction", ui.num(row["avg_satisfaction"], 1)),
    ])

# ----------------------------------------------------------------- trends ---
ui.section(
    "How the relationship has moved",
    "Fifteen months of activity. The scoring window is the most recent twelve; "
    "the model saw nothing outside it.",
)
history = data.customer_history(choice)
t1, t2 = st.columns(2, gap="large")
with t1:
    st.markdown("#### Balance and transaction value")
    st.plotly_chart(
        charts.trend_lines(history, "month_index",
                           {"closing_balance": "Closing balance",
                            "transaction_value": "Transaction value"},
                           y_title="Monetary units"),
        width="stretch",
    )
with t2:
    st.markdown("#### Activity and engagement")
    st.plotly_chart(
        charts.trend_lines(history, "month_index",
                           {"transaction_count": "Transactions",
                            "total_logins": "Logins"},
                           y_title="Count", value_fmt=":,.0f"),
        width="stretch",
    )
ui.note(
    "Balance and transaction value share a chart because they share a unit. "
    "Counts get their own chart rather than a second y-axis — a dual axis lets "
    "whoever drew it choose where the two lines appear to cross."
)

# --------------------------------------------------------------- scoring ----
ui.section("Scores and suggested action")
s1, s2 = st.columns([1, 1.2], gap="large")
with s1:
    inv = row["investment_propensity"]
    lend = row["lending_propensity"]
    ui.panel("Model scores", [
        ("Attrition risk", ui.score_label(row["churn_probability"],
                                          THRESHOLDS.get("churn"), risk_band)),
        ("Investment propensity", ui.score_label(
            inv, THRESHOLDS.get("investment"),
            band(inv, PROPENSITY_BANDS) if not pd.isna(inv) else "")),
        ("Lending propensity", ui.score_label(
            lend, THRESHOLDS.get("lending"),
            band(lend, PROPENSITY_BANDS) if not pd.isna(lend) else "")),
        ("Relationship opportunity", f"{row['opportunity_score']:.0f} / 100 · {opp_band}"),
        ("Quadrant", row["quadrant"]),
    ])
    ui.note(
        "A propensity score reads 'already held' where the customer has the "
        "product. The model was trained only on customers who could take it, so "
        "scoring anyone else would be applying it outside its population."
    )
with s2:
    st.markdown(
        f'<div class="ci-panel">'
        f'<div class="ci-eyebrow">Suggested action</div>'
        f'<h4 style="margin:.15rem 0 .3rem;font-size:1.1rem;">{row["action"]}</h4>'
        f'<div style="font-size:.85rem;color:#5a6169;margin-bottom:.7rem;">'
        f'{row["channel"]} · {row["priority"]} priority · {row["cost"]}</div>'
        f'<div class="ci-eyebrow" style="margin-top:.8rem;">Because</div>'
        f'<div style="font-size:.87rem;color:#14171a;line-height:1.6;">'
        f'{row["conditions"]}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="ci-note">This is a suggestion for a person to weigh, not an '
        "automated decision, and not a financial, credit or product "
        "recommendation. <b>Decision support</b> shows the full reasoning, "
        "including which model inputs drove the risk score.</p>",
        unsafe_allow_html=True,
    )
