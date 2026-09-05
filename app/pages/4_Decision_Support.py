"""Decision support — the showcase page, where evidence becomes a suggestion."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Decision support")

from components import charts, data, ui                       # noqa: E402
from components.theme import BLUE, INK_MUTED, STATUS          # noqa: E402
from customer_intelligence.analytics import models as M       # noqa: E402
from customer_intelligence.config import (                    # noqa: E402
    CHURN_BANDS, OPPORTUNITY_BANDS, PROPENSITY_BANDS, band,
)
from customer_intelligence.decision_engine import recommend    # noqa: E402

df = data.customers()
cards = data.model_cards()
run = data.run_summary()
THRESHOLDS = run.get("thresholds", {})

ui.page_header(
    "Decision support",
    "From a score to a suggested action",
    "This is where the whole system comes together for one customer: the "
    "evidence the models produced, what drove it, the action the rules "
    "suggest, and the constraints on acting.",
)
ui.synthetic_notice()

# ----------------------------------------------------------------- picker ---
c1, c2 = st.columns([1, 2])
with c1:
    view = st.radio(
        "Show me",
        ["Highest balance at risk", "Highest opportunity", "Highest attrition risk",
         "Search by ID"],
        label_visibility="collapsed",
    )
with c2:
    if view == "Search by ID":
        choice = st.selectbox("Customer ID", df["customer_id"].tolist())
    else:
        pool = {
            "Highest balance at risk": df[df["quadrant"] == "Retain"].nlargest(
                60, "avg_monthly_balance"),
            "Highest opportunity": df.nlargest(60, "opportunity_score"),
            "Highest attrition risk": df.nlargest(60, "churn_probability"),
        }[view]
        choice = st.selectbox(
            "Customer", pool["customer_id"].tolist(),
            format_func=lambda cid: (
                f"{cid}  ·  {df.loc[df.customer_id == cid, 'segment'].iloc[0]}"
                f"  ·  {ui.money(df.loc[df.customer_id == cid, 'avg_monthly_balance'].iloc[0])}"
            ),
        )

row = df[df["customer_id"] == choice].iloc[0]
risk_band = band(row["churn_probability"], CHURN_BANDS)
opp_band = band(row["opportunity_score"], OPPORTUNITY_BANDS)

ui.rule()

# ------------------------------------------------------------ the summary ---
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown(f"### {choice}")
    st.markdown(
        f'<div style="margin:-.3rem 0 .9rem;">{ui.badge(row["segment"])} '
        f'{ui.risk_badge(risk_band)} {ui.badge(row["quadrant"])}</div>',
        unsafe_allow_html=True,
    )

    inv, lend = row["investment_propensity"], row["lending_propensity"]
    ui.panel("The evidence", [
        ("Segment", row["segment"]),
        ("Attrition risk", ui.score_label(row["churn_probability"],
                                          THRESHOLDS.get("churn"), risk_band)),
        ("Investment propensity", ui.score_label(
            inv, THRESHOLDS.get("investment"),
            band(inv, PROPENSITY_BANDS) if not pd.isna(inv) else "")),
        ("Lending propensity", ui.score_label(
            lend, THRESHOLDS.get("lending"),
            band(lend, PROPENSITY_BANDS) if not pd.isna(lend) else "")),
        ("Relationship opportunity", f"{row['opportunity_score']:.0f} / 100 · {opp_band}"),
        ("Average monthly balance", ui.money(row["avg_monthly_balance"])),
        ("Products held", f"{int(row['product_count'])}"),
        ("Months active of 12", f"{int(row['active_months'])}"),
        ("Complaints in window", f"{int(row['complaints'])}"),
        ("Unresolved contacts", f"{int(row['unresolved_interactions'])}"),
    ])

with right:
    rec = recommend(row, thresholds=THRESHOLDS)
    action = rec.action
    kind = {"Urgent": "critical", "High": "serious"}.get(action.priority, "neutral")
    st.markdown(
        f'<div class="ci-panel" style="border-left:3px solid '
        f'{STATUS.get(kind, INK_MUTED)};">'
        f'<div class="ci-eyebrow">Suggested action</div>'
        f'<h3 style="margin:.2rem 0 .45rem;">{action.title}</h3>'
        f'<div style="font-size:.86rem;color:#5a6169;margin-bottom:1rem;">'
        f'{action.channel} &nbsp;·&nbsp; {action.priority} priority '
        f'&nbsp;·&nbsp; {action.cost}</div>'
        f'<div class="ci-eyebrow">Which rule fired</div>'
        + "".join(
            f'<div style="font-size:.87rem;color:#14171a;padding:.22rem 0;">• {c}</div>'
            for c in rec.conditions
        )
        + f'<div class="ci-eyebrow" style="margin-top:1rem;">Why this action</div>'
        f'<div style="font-size:.87rem;color:#5a6169;line-height:1.6;">'
        f'{action.rationale}</div>'
        + (f'<div class="ci-eyebrow" style="margin-top:1rem;">Constraints on acting</div>'
           f'<div style="font-size:.87rem;color:#14171a;line-height:1.6;">'
           f'{action.guardrail}</div>' if action.guardrail else "")
        + f'<div class="ci-eyebrow" style="margin-top:1rem;">Confidence</div>'
        f'<div style="font-size:.87rem;color:#5a6169;">{rec.confidence}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    ui.note(
        "Confidence reflects how much history exists for this customer, not how "
        "extreme the score is. A decisive-looking score on three months of thin "
        "activity is the case to treat most carefully, not least."
    )

# ------------------------------------------------------- why this score -----
ui.note(
    "A score can read <b>Moderate</b> on the descriptive bands and still be "
    "flagged for outreach. The bands describe where a probability sits; the "
    "threshold is the separate decision about how many customers a team can "
    "actually contact. Both are shown so the score and the action agree."
)

ui.section(
    "What drove the attrition score",
    "Each bar is the model's standardised coefficient multiplied by how far "
    "this customer sits from the book average on that input. That is the "
    "arithmetic of the decision for this row — not an approximation of it.",
)

train_360 = data.load("customer_360_train")
coefficients = data.load("model_churn_coefficients")


class _Result:
    """The minimum `top_reasons` needs, without reloading the fitted model."""
    target = "churned"
    coefficients = None


_r = _Result()
_r.coefficients = coefficients
reasons = M.top_reasons(_r, row, train_360, n=6)

d1, d2 = st.columns([1.3, 1], gap="large")
with d1:
    if reasons:
        st.plotly_chart(charts.contribution_bars(reasons, height=300),
                        width="stretch")
    else:
        st.info("No contributing features could be computed for this customer.")
with d2:
    ui.panel("Reading this chart", [
        ("Red bars", "Push attrition risk up"),
        ("Blue bars", "Push attrition risk down"),
        ("Bar length", "Coefficient × distance from average"),
    ])
    ui.note(
        "This is why the production model is a logistic regression. The "
        "contribution of each input is readable, additive, and checkable by "
        "hand. A gradient boosting model scored marginally better on ROC-AUC "
        "and could not produce this panel honestly — see <b>Data quality &amp; "
        "governance</b> for the full comparison."
    )

if reasons:
    st.markdown("#### The same reasoning, in words")
    for r in reasons[:4]:
        direction = "raises" if r["contribution"] > 0 else "lowers"
        comparison = "above" if r["z"] > 0 else "below"
        st.markdown(
            f"- **{r['label']}** is {r['value']:,.1f}, "
            f"{abs(r['z']):.1f} standard deviations {comparison} the book average "
            f"of {r['population_mean']:,.1f} — which {direction} the modelled risk."
        )

# ------------------------------------------------------------- the book -----
ui.section(
    "The same engine across the whole book",
    "Applied to every customer, so the distribution of recommended effort can "
    "be checked before anyone acts on it.",
)

summary = df.groupby(["action", "priority", "cost"], observed=True).agg(
    customers=("customer_id", "size"),
    balances=("avg_monthly_balance", "sum"),
    mean_risk=("churn_probability", "mean"),
    mean_opportunity=("opportunity_score", "mean"),
).reset_index().sort_values("customers", ascending=False)
summary.columns = ["Suggested action", "Priority", "Effort falls on", "Customers",
                   "Balances", "Mean risk", "Mean opportunity"]
st.dataframe(
    summary.style.format({
        "Customers": "{:,.0f}", "Balances": "{:,.0f}",
        "Mean risk": "{:.1%}", "Mean opportunity": "{:.0f}",
    }),
    width="stretch", hide_index=True,
)

human = df[df["cost"] == "Relationship manager"]
ui.note(
    f"{len(human):,} customers ({len(human) / len(df):.0%}) are routed to a "
    "relationship manager — the scarcest and most expensive channel — and they "
    f"hold {ui.money(human['avg_monthly_balance'].sum())} in balances. A "
    "recommendation engine that routes most of the book to human contact has "
    "produced a wish list, not a plan; the check belongs on the page, not in a "
    "footnote."
)
