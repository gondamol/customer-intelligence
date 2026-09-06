"""Account 360 — everything known about one trading relationship."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Account 360")

from components import charts, data, ui                     # noqa: E402
from components.theme import BLUE, ORANGE                   # noqa: E402

df = data.customers()
run = data.run_summary()
THRESHOLDS = run.get("thresholds", {})

ui.page_header(
    "Account 360",
    "One trading relationship, end to end",
    "Every attribute, behaviour, score and suggested action for a single "
    "account — assembled from a flat ledger of invoice lines into one "
    "analytical view.",
)
ui.source_notice()

st.markdown("#### Select an account")
c1, c2, c3 = st.columns([1.1, 1, 1])
with c1:
    segment_filter = st.selectbox("Segment", ["All segments"] + sorted(df["segment"].unique()))
with c2:
    scored_only = st.checkbox("Scoreable accounts only", value=True)
with c3:
    sort_by = st.selectbox("Order by", ["Revenue (highest)", "Lapse risk (highest)",
                                        "Opportunity (highest)", "Account ID"])

pool = df
if segment_filter != "All segments":
    pool = pool[pool["segment"] == segment_filter]
if scored_only:
    pool = pool[pool["scoreable"]]

sort_map = {"Revenue (highest)": ("revenue", False),
            "Lapse risk (highest)": ("lapse_risk", False),
            "Opportunity (highest)": ("opportunity_score", False),
            "Account ID": ("customer_id", True)}
col, asc = sort_map[sort_by]
pool = pool.sort_values(col, ascending=asc, na_position="last")

if pool.empty:
    st.info("No accounts match that combination of filters.")
    st.stop()

choice = st.selectbox(
    f"Account ({len(pool):,} match)", pool["customer_id"].head(400).tolist(),
    format_func=lambda cid: (
        f"{cid}  ·  {pool.loc[pool.customer_id==cid,'segment'].iloc[0]}"
        f"  ·  {ui.money(pool.loc[pool.customer_id==cid,'revenue'].iloc[0])}"
    ),
)
row = df[df["customer_id"] == choice].iloc[0]
ui.rule()

st.markdown(f"### {choice}")
st.markdown(
    f'<div style="margin:-.35rem 0 1rem;">{ui.badge(row["segment"])} '
    f'{ui.badge(row["quadrant"])} {ui.priority_badge(row["priority"])} '
    f'{ui.badge(row["country"])}</div>',
    unsafe_allow_html=True,
)

ui.tiles([
    {"label": "Revenue in window", "value": ui.money(row["revenue"])},
    {"label": "Orders", "value": f"{int(row['invoices'])}",
     "sub": f"{ui.money(row['avg_order_value'])} average"},
    {"label": "Lapse risk",
     "value": "—" if pd.isna(row["lapse_risk"]) else f"{row['lapse_risk']:.0%}",
     "sub": "Not scored" if pd.isna(row["lapse_risk"]) else "Next-quarter silence"},
    {"label": "Opportunity", "value": f"{row['opportunity_score']:.0f}",
     "sub": "out of 100"},
    {"label": "Months since order", "value": f"{int(row['months_since_last_order'])}",
     "sub": (f"usual gap {row['avg_months_between_orders']:.1f}"
             if pd.notna(row["avg_months_between_orders"]) else "no rhythm yet")},
])

p1, p2, p3 = st.columns(3, gap="medium")
with p1:
    ui.panel("Trading profile", [
        ("Country", row["country"]),
        ("Months on book", f"{int(row['months_on_book'])}"),
        ("Months with an order", f"{int(row['active_months'])} of 12"),
        ("Orders placed", f"{int(row['invoices'])}"),
        ("Average order value", ui.money(row["avg_order_value"])),
        ("Lines per order", ui.num(row["lines_per_order"], 1)),
        ("Average unit price", ui.money(row["avg_unit_price"], 2)),
    ])
with p2:
    ui.panel("Breadth and rhythm", [
        ("Distinct products", ui.num(row["distinct_products"])),
        ("Categories bought", f"{int(row['distinct_categories'])} of 12"),
        ("Usual gap between orders", ui.num(row["avg_months_between_orders"], 1)),
        ("Longest gap", ui.num(row["max_months_between_orders"], 0)),
        ("Consistency (lower is steadier)", ui.num(row["gap_variability"], 2)),
        ("Past their usual gap by", ui.num(row["cadence_overdue"], 1) + "×"),
        ("Revenue trend", ui.num(row["revenue_trend"], 2)),
    ])
with p3:
    ui.panel("Friction and value", [
        ("Units bought", ui.num(row["units"])),
        ("Return lines", ui.num(row["return_lines"])),
        ("Value returned", ui.money(row["return_value"])),
        ("Return rate", ui.pct(row["return_rate"])),
        ("Discount rate", ui.pct(row["discount_rate"])),
        ("Postage paid", ui.money(row["postage"])),
        ("Best month", ui.money(row["peak_monthly_revenue"])),
    ])

# ------------------------------------------------------------- trends -------
ui.section(
    "How the relationship has moved",
    "The full 24-month ledger. The scoring window is the most recent twelve; "
    "the models saw nothing outside it.",
)
history = data.customer_history(choice)
t1, t2 = st.columns(2, gap="large")
with t1:
    st.markdown("#### Revenue by month")
    st.plotly_chart(
        charts.trend_lines(history, "month_index", {"revenue": "Revenue"},
                           y_title="£ per month"),
        width="stretch",
    )
with t2:
    st.markdown("#### Orders and breadth by month")
    st.plotly_chart(
        charts.trend_lines(history, "month_index",
                           {"invoices": "Orders", "distinct_categories": "Categories"},
                           y_title="Count"),
        width="stretch",
    )
ui.note(
    "Revenue and counts are on separate charts rather than a single pair of "
    "axes. A dual axis lets whoever drew it choose where the two lines appear "
    "to cross."
)

# ------------------------------------------------------- category mix -------
ui.section("What this account buys")
shares = {c: row[c] for c in df.columns if c.startswith("share_") and pd.notna(row[c])}
if shares and sum(shares.values()) > 0:
    labels = [c.replace("share_", "").replace("_", " ").capitalize() for c in shares]
    values = [v * 100 for v in shares.values()]
    order = sorted(zip(labels, values), key=lambda x: -x[1])
    st.plotly_chart(
        charts.hbar([o[0] for o in order], [o[1] for o in order],
                    value_fmt="{:.1f}", label_suffix="%", height=330,
                    hover_label="Share of this account's spend"),
        width="stretch",
    )
else:
    st.info("No categorised purchases in the scoring window.")

# -------------------------------------------------------- what to offer -----
ui.section(
    "What to offer next",
    "From an item-to-item recommender over what similar accounts buy — not from "
    "a propensity score, which could rank a customer but never name a product.",
)
nb = data.next_best(choice)
if nb.empty:
    st.info("No recommendation: this account has no purchase history in the scoring window.")
else:
    show = nb[["rank", "description", "category", "affinity"]].copy()
    show.columns = ["Rank", "Product", "Category", "Affinity"]
    st.dataframe(show.style.format({"Affinity": "{:.3f}"}),
                 width="stretch", hide_index=True)
    ui.note(
        "Affinity is the account's similarity-weighted score for a product it "
        "does not currently buy, normalised by how broad its range already is — "
        "so a wide-ranging account does not out-score a narrow one on every "
        "candidate."
    )

# --------------------------------------------------------- the action -------
ui.section("Suggested action")
s1, s2 = st.columns([1, 1.2], gap="large")
with s1:
    ui.panel("Model scores", [
        ("Lapse risk", ui.score_label(row["lapse_risk"], THRESHOLDS.get("lapse"), "")),
        ("Growth propensity", ui.score_label(row["growth_propensity"],
                                             THRESHOLDS.get("growth"), "")),
        ("Relationship opportunity", f"{row['opportunity_score']:.0f} / 100"),
        ("Quadrant", row["quadrant"]),
        ("Scoreable", "Yes" if row["scoreable"] else "No — fewer than 3 active months"),
    ])
with s2:
    st.markdown(
        f'<div class="ci-panel">'
        f'<div class="ci-eyebrow">Suggested action</div>'
        f'<h4 style="margin:.15rem 0 .3rem;font-size:1.1rem;">{row["action"]}</h4>'
        f'<div style="font-size:.85rem;color:#5a6169;margin-bottom:.7rem;">'
        f'{row["channel"]} · {row["priority"]} priority · {row["cost"]}</div>'
        f'<div class="ci-eyebrow" style="margin-top:.8rem;">Because</div>'
        f'<div style="font-size:.87rem;color:#14171a;line-height:1.6;">{row["conditions"]}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="ci-note">A suggestion for a person to weigh, not an automated '
        "decision. <b>Decision support</b> shows the full reasoning, including "
        "which model inputs drove the score.</p>",
        unsafe_allow_html=True,
    )
