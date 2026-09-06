"""Decision support — where evidence becomes a suggestion."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Decision support")

from components import charts, data, ui                                  # noqa: E402
from components.theme import BLUE, INK_MUTED, STATUS                     # noqa: E402
from customer_intelligence.analytics import models as M                  # noqa: E402
from customer_intelligence.analytics import features_retail as FR        # noqa: E402
from customer_intelligence.decision_engine.retail_engine import recommend_retail  # noqa: E402

M.use_contract(FR)

df = data.customers()
cards = data.model_cards()
run = data.run_summary()
THRESHOLDS = run.get("thresholds", {})

ui.page_header(
    "Decision support",
    "From a score to a suggested action",
    "Where the whole system comes together for one account: the evidence the "
    "models produced, what drove it, what to offer, the action the rules "
    "suggest, and the constraints on acting.",
)
ui.source_notice()

c1, c2 = st.columns([1, 2])
with c1:
    view = st.radio("Show me",
                    ["Revenue at risk", "Highest opportunity", "Highest lapse risk",
                     "Search by ID"],
                    label_visibility="collapsed")
with c2:
    if view == "Search by ID":
        choice = st.selectbox("Account", df["customer_id"].tolist())
    else:
        # Score-ranked pools contain only accounts that actually carry a score.
        scoreable = df[df["scoreable"]]
        pool = {
            "Revenue at risk": scoreable[scoreable["quadrant"] == "Retain"].nlargest(60, "revenue"),
            "Highest opportunity": scoreable.nlargest(60, "opportunity_score"),
            "Highest lapse risk": scoreable.nlargest(60, "lapse_risk"),
        }[view]
        if pool.empty:
            pool = scoreable.nlargest(60, "revenue")
        choice = st.selectbox(
            "Account", pool["customer_id"].tolist(),
            format_func=lambda cid: (
                f"{cid}  ·  {df.loc[df.customer_id==cid,'segment'].iloc[0]}"
                f"  ·  {ui.money(df.loc[df.customer_id==cid,'revenue'].iloc[0])}"
            ),
        )

row = df[df["customer_id"] == choice].iloc[0]
ui.rule()

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.markdown(f"### {choice}")
    st.markdown(
        f'<div style="margin:-.3rem 0 .9rem;">{ui.badge(row["segment"])} '
        f'{ui.badge(row["quadrant"])} {ui.badge(row["country"])}</div>',
        unsafe_allow_html=True,
    )
    ui.panel("The evidence", [
        ("Segment", row["segment"]),
        ("Revenue in window", ui.money(row["revenue"])),
        ("Orders / average value",
         f"{int(row['invoices'])} · {ui.money(row['avg_order_value'])}"),
        ("Months with an order", f"{int(row['active_months'])} of 12"),
        ("Months since last order", f"{int(row['months_since_last_order'])}"),
        ("Usual gap between orders", ui.num(row["avg_months_between_orders"], 1)),
        ("Past their usual gap by", ui.num(row["cadence_overdue"], 1) + "×"),
        ("Lapse risk", ui.score_label(row["lapse_risk"], THRESHOLDS.get("lapse"), "")),
        ("Growth propensity", ui.score_label(row["growth_propensity"],
                                             THRESHOLDS.get("growth"), "")),
        ("Relationship opportunity", f"{row['opportunity_score']:.0f} / 100"),
        ("Return rate", ui.pct(row["return_rate"])),
    ])

with right:
    rec = recommend_retail(row, thresholds=THRESHOLDS)
    action = rec.action
    kind = {"Urgent": "critical", "High": "serious"}.get(action.priority, "neutral")
    st.markdown(
        f'<div class="ci-panel" style="border-left:3px solid {STATUS.get(kind, INK_MUTED)};">'
        f'<div class="ci-eyebrow">Suggested action</div>'
        f'<h3 style="margin:.2rem 0 .45rem;">{action.title}</h3>'
        f'<div style="font-size:.86rem;color:#5a6169;margin-bottom:1rem;">'
        f'{action.channel} &nbsp;·&nbsp; {action.priority} priority &nbsp;·&nbsp; {action.cost}</div>'
        f'<div class="ci-eyebrow">Which rule fired</div>'
        + "".join(f'<div style="font-size:.87rem;color:#14171a;padding:.22rem 0;">• {c}</div>'
                  for c in rec.conditions)
        + '<div class="ci-eyebrow" style="margin-top:1rem;">Why this action</div>'
        f'<div style="font-size:.87rem;color:#5a6169;line-height:1.6;">{action.rationale}</div>'
        + (f'<div class="ci-eyebrow" style="margin-top:1rem;">Constraints on acting</div>'
           f'<div style="font-size:.87rem;color:#14171a;line-height:1.6;">{action.guardrail}</div>'
           if action.guardrail else "")
        + f'<div class="ci-eyebrow" style="margin-top:1rem;">Confidence</div>'
        f'<div style="font-size:.87rem;color:#5a6169;">{rec.confidence}</div></div>',
        unsafe_allow_html=True,
    )
    if pd.notna(row.get("next_best_product")):
        st.markdown(
            f'<div class="ci-panel">'
            f'<div class="ci-eyebrow">What to lead with</div>'
            f'<div style="font-size:.98rem;color:#14171a;font-weight:600;margin:.2rem 0 .25rem;">'
            f'{row["next_best_product"]}</div>'
            f'<div style="font-size:.83rem;color:#5a6169;">'
            f'{row["next_best_category"]} · affinity {row["next_best_affinity"]:.3f}</div>'
            f'<div style="font-size:.82rem;color:#8b9199;margin-top:.6rem;line-height:1.5;">'
            f'From an item-to-item recommender over what similar accounts buy. A '
            f'propensity score could rank this account; only a recommender can '
            f'name the product.</div></div>',
            unsafe_allow_html=True,
        )

ui.section(
    "What drove the lapse score",
    "Each bar is the model's standardised coefficient multiplied by how far this "
    "account sits from the book average on that input. That is the arithmetic of "
    "the decision for this row, not an approximation of it.",
)
train_360 = data.load("customer_360_train")
coefficients = data.load("model_lapse_coefficients")


class _Result:
    target = "lapsed"
    coefficients = None


_r = _Result()
_r.coefficients = coefficients
reasons = M.top_reasons(_r, row, train_360, n=6) if row["scoreable"] else []

d1, d2 = st.columns([1.3, 1], gap="large")
with d1:
    if reasons:
        st.plotly_chart(charts.contribution_bars(reasons, height=300), width="stretch")
    else:
        st.info("This account was not scored: fewer than three active months in the "
                "window, so the model has no basis to explain.")
with d2:
    ui.panel("Reading this chart", [
        ("Red bars", "Push lapse risk up"),
        ("Blue bars", "Push lapse risk down"),
        ("Bar length", "Coefficient × distance from average"),
    ])
    ui.note(
        "This is why the production model is a logistic regression. Every "
        "contribution is readable, additive and checkable by hand. Gradient "
        "boosting was trained alongside it and its score is published on "
        "<b>Data quality &amp; governance</b>."
    )

if reasons:
    st.markdown("#### The same reasoning, in words")
    for r in reasons[:4]:
        direction = "raises" if r["contribution"] > 0 else "lowers"
        comparison = "above" if r["z"] > 0 else "below"
        st.markdown(
            f"- **{r['label']}** is {r['value']:,.1f}, {abs(r['z']):.1f} standard "
            f"deviations {comparison} the book average of {r['population_mean']:,.1f} "
            f"— which {direction} the modelled risk."
        )

ui.section(
    "The same engine across the whole book",
    "Applied to every account, so the distribution of recommended effort can be "
    "checked before anyone acts on it.",
)
summary = df.groupby(["action", "priority", "cost"], observed=True).agg(
    accounts=("customer_id", "size"), revenue=("revenue", "sum"),
    mean_risk=("lapse_risk", "mean"), mean_opportunity=("opportunity_score", "mean"),
).reset_index().sort_values("accounts", ascending=False)
summary.columns = ["Suggested action", "Priority", "Effort falls on", "Accounts",
                   "Revenue", "Mean lapse risk", "Mean opportunity"]
st.dataframe(
    summary.style.format({"Accounts": "{:,.0f}", "Revenue": "£{:,.0f}",
                          "Mean lapse risk": "{:.1%}", "Mean opportunity": "{:.0f}"}),
    width="stretch", hide_index=True,
)
human = df[df["cost"] == "Account manager"]
ui.note(
    f"{len(human):,} accounts ({len(human)/len(df):.0%}) are routed to a named "
    f"account manager and they hold {human['revenue'].sum()/df['revenue'].sum():.0%} "
    "of revenue. A recommendation engine that routes most of the book to human "
    "contact has produced a wish list, not a plan; the check belongs on the page."
)
