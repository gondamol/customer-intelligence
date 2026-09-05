"""Executive view — the landing page.

Four questions in order, and nothing else on the page:

    What is happening?  ->  Why is it happening?
    Where is the opportunity?  ->  What should we consider doing?

The temptation on a page like this is to show everything that exists. The
discipline is to show only what changes a decision, and to send the reader to
the page that answers the next question rather than answering it here badly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import charts, data, ui                       # noqa: E402
from components.theme import BLUE, ORANGE, QUADRANT_COLOURS, RISK_COLOURS, register_template  # noqa: E402

st.set_page_config(
    page_title="Customer Intelligence & Decision Analytics",
    page_icon="◆", layout="wide", initial_sidebar_state="expanded",
)
register_template()
st.markdown(__import__("components.theme", fromlist=["CSS"]).CSS, unsafe_allow_html=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_intelligence.config import CHURN_BANDS, band  # noqa: E402

df = data.customers()
cards = data.model_cards()
quality = data.quality_summary()

# ---------------------------------------------------------------- header ----
ui.page_header(
    "Customer Intelligence & Decision Analytics",
    "From data to decisions",
    "An end-to-end demonstration of how fragmented customer data becomes "
    "customer intelligence, predictive insight, and an explainable suggested "
    "action for a person to weigh. The methodology transfers to any "
    "data-intensive customer environment — financial services, telecoms, "
    "insurance, retail, healthcare.",
)
ui.synthetic_notice()

# ------------------------------------------------- 1. what is happening? ----
ui.section(
    "What is happening?",
    "The state of the book at the close of the observation window.",
)

active = int((df["active_months"] >= 6).sum())
elevated = int((df["churn_probability"] >= 0.25).sum())
high_opp = int((df["opportunity_score"] >= 60).sum())
retain = int((df["quadrant"] == "Retain").sum())

ui.tiles([
    {"label": "Customers", "value": f"{len(df):,}",
     "sub": "Synthetic population under management"},
    {"label": "Consistently active", "value": f"{active / len(df):.0%}",
     "sub": f"{active:,} active in 6+ of the last 12 months"},
    {"label": "Elevated attrition risk", "value": f"{elevated / len(df):.1%}",
     "sub": f"{elevated:,} customers above a 25% modelled probability"},
    {"label": "High opportunity", "value": f"{high_opp / len(df):.0%}",
     "sub": f"{high_opp:,} scoring 60 or above out of 100"},
    {"label": "Retain quadrant", "value": f"{retain:,}",
     "sub": "High opportunity and at risk — the time-critical group"},
])

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("#### Where attrition risk sits")
    bands = df["churn_probability"].apply(lambda p: band(p, CHURN_BANDS))
    order = [b for b, _ in CHURN_BANDS]
    counts = bands.value_counts().reindex(order).fillna(0).astype(int)
    st.plotly_chart(charts.banded_bar(counts, RISK_COLOURS, total=len(df)),
                    use_container_width=True)
    ui.note(
        "Bands are fixed probability cut-offs, chosen once and documented — not "
        "quantiles, which would move every time the book is refreshed and make "
        "two months' reports incomparable."
    )

with right:
    st.markdown("#### Customers by segment")
    profiles = data.load("segment_profiles")
    st.plotly_chart(
        charts.grouped_profile(profiles, "segment", "customers",
                               hover_label="Customers"),
        use_container_width=True,
    )
    ui.note(
        "Segments come from a published rule set rather than a clustering, so "
        "the same customer lands in the same segment next month unless their "
        "behaviour changed. See <b>Customer segments</b> for how a K-means "
        "comparison scores against these rules."
    )

# ------------------------------------------------- 2. why is it happening ---
ui.section(
    "Why is it happening?",
    "What the attrition model actually keys on, ranked by how much the measured "
    "performance depends on each input.",
)

imp = data.load("model_churn_importance").head(8)
c1, c2 = st.columns([1.25, 1], gap="large")
with c1:
    st.plotly_chart(
        charts.hbar(list(imp["label"]), list(imp["importance"]),
                    value_fmt="{:.3f}", height=320, hover_label="Drop in ROC-AUC"),
        use_container_width=True,
    )
    ui.note(
        "Permutation importance: how far ROC-AUC falls when a single input is "
        "shuffled. It measures what the model <i>relies on</i>, which is not the "
        "same as what causes attrition — nothing here licenses a causal claim."
    )
with c2:
    churn = cards["churn"]
    ui.panel("Attrition model", [
        ("Discrimination (ROC-AUC)", f"{churn['metrics']['roc_auc']:.3f}"),
        ("Precision-recall AUC", f"{churn['metrics']['pr_auc']:.3f}"),
        ("Recall at operating point", ui.pct(churn["metrics"]["recall"])),
        ("Lift over the base rate", f"{churn['metrics']['lift']:.1f}×"),
        ("Base rate in training", ui.pct(churn["base_rate"])),
        ("Model in production", churn["chosen"]),
    ])
    ui.note(
        "The operating threshold flags the top 10% of the book, not everything "
        "above a probability of 0.5. Outreach capacity is the binding "
        "constraint, so the threshold is set to it."
    )

# --------------------------------------------- 3. where is the opportunity --
ui.section(
    "Where is the opportunity?",
    "Attrition risk against relationship opportunity. The two are scored "
    "separately and deliberately: a customer can be high on both, and that "
    "combination is the one worth acting on first.",
)

m1, m2 = st.columns([1.4, 1], gap="large")
with m1:
    st.plotly_chart(
        charts.risk_opportunity_matrix(df["opportunity_score"], df["churn_probability"]),
        use_container_width=True,
    )
with m2:
    st.markdown("#### The four groups")
    from customer_intelligence.analytics.opportunity import QUADRANT_MEANING
    quad_counts = df["quadrant"].value_counts()
    for name in ["Retain", "Grow", "Monitor", "Develop"]:
        n = int(quad_counts.get(name, 0))
        st.markdown(
            f'<div class="ci-panel" style="padding:.75rem .95rem;margin-bottom:.55rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<strong style="color:{QUADRANT_COLOURS[name]};">{name}</strong>'
            f'<span style="font-variant-numeric:tabular-nums;color:#5a6169;">'
            f"{n:,} · {n / len(df):.0%}</span></div>"
            f'<div style="font-size:.82rem;color:#5a6169;margin-top:.3rem;line-height:1.45;">'
            f"{QUADRANT_MEANING[name]}</div></div>",
            unsafe_allow_html=True,
        )

# --------------------------------------- 4. what should we consider doing? --
ui.section(
    "What should we consider doing?",
    "The decision engine turns the scores into a suggested action per customer. "
    "These are suggestions for a person to weigh, not automated decisions.",
)

actions = df["action"].value_counts().reset_index()
actions.columns = ["action", "customers"]
a1, a2 = st.columns([1.3, 1], gap="large")
with a1:
    st.plotly_chart(
        charts.grouped_profile(actions, "action", "customers", height=340,
                               hover_label="Customers"),
        use_container_width=True,
    )
with a2:
    cost = df["cost"].value_counts()
    ui.panel("Where the effort would fall", [
        (k, f"{int(v):,} · {v / len(df):.0%}") for k, v in cost.items()
    ])
    urgent = int((df["priority"] == "Urgent").sum())
    ui.panel("Priority mix", [
        (k, f"{int(v):,}") for k, v in df["priority"].value_counts().items()
    ])
    ui.note(
        f"{urgent:,} customers carry an urgent action. Any recommendation "
        "engine that marks most of the book urgent has told the reader nothing, "
        "so the rules put credit position and unresolved service failure ahead "
        "of every commercial action."
    )

# ------------------------------------------------------------ foundations ---
ui.rule()
ui.section(
    "Can we trust the data underneath?",
    "Every number above rests on data that was deliberately damaged before it "
    "was measured, to prove the quality framework detects what it claims to.",
)

ui.tiles([
    {"label": "Record-level conformance", "value": f"{quality['score']:.2f}%",
     "sub": "Severity-weighted pass rate across all rows"},
    {"label": "Rules passing", "value": f"{quality['checks_passed']} of {quality['checks_run']}",
     "sub": "Automated checks across seven quality dimensions"},
    {"label": "Customers affected", "value": f"{quality['customers_affected_pct']:.0f}%",
     "sub": f"{quality['customers_affected']:,} customers touched by at least one defect"},
    {"label": "Defects detected", "value": f"{quality['defects_caught']} of {quality['defects_injected']}",
     "sub": "Injected defect types the checks caught"},
])
ui.note(
    "The first and third tiles are the same data and disagree completely. "
    "Conformance is a row-weighted average, so a defect rate under one percent "
    "always reads as a healthy number; but defects are not spread evenly, and "
    "most customers are touched by at least one. Reporting only the first "
    "figure would be technically true and materially misleading — which is why "
    "both are on the page. <b>Data quality &amp; governance</b> has the detail."
)

st.markdown(
    '<p class="ci-note" style="margin-top:2.4rem;">'
    "Built by Nichodemus Amollo as a portfolio demonstration. Synthetic data "
    "throughout; no real organisation, customer, or portfolio is represented. "
    "The domain may change — the analytical problem does not.</p>",
    unsafe_allow_html=True,
)
