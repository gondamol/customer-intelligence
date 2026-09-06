"""Executive view — the landing page.

Four questions in order, and nothing else:

    What is happening?  ->  Why is it happening?
    Where is the opportunity?  ->  What should we consider doing?

The temptation on a page like this is to show everything that exists. The
discipline is to show only what changes a decision, and to send the reader to
the page that answers the next question rather than answering it here badly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components.page import setup                                          # noqa: E402

setup("Executive view", root=True)

from components import charts, data, ui                                    # noqa: E402
from components.theme import BLUE, ORANGE, QUADRANT_COLOURS, RISK_COLOURS  # noqa: E402
from customer_intelligence.analytics.opportunity import QUADRANT_MEANING   # noqa: E402

df = data.customers()
cards = data.model_cards()
quality = data.quality_summary()
run = data.run_summary()

ui.page_header(
    "Customer Intelligence & Decision Analytics",
    "From data to decisions",
    "An end-to-end analytics product built on a real, openly published "
    "transaction ledger: two years of invoices from a UK giftware wholesaler, "
    "turned into account intelligence, predictive scores, and an explainable "
    "suggested action for a person to weigh. The method transfers to any "
    "data-intensive customer environment.",
)
ui.source_notice()

# ------------------------------------------------- 1. what is happening? ----
ui.section("What is happening?",
           "The state of the account book at the close of the observation window.")

scoreable = int(df["scoreable"].sum())
at_risk = int((df["lapse_risk"] >= 0.5).sum())
high_opp = int((df["opportunity_score"] >= 60).sum())
revenue = float(df["revenue"].sum())
champions = int((df["segment"] == "Champions").sum())

ui.tiles([
    {"label": "Accounts", "value": f"{len(df):,}",
     "sub": "Identified customers in the ledger"},
    {"label": "Revenue in window", "value": ui.money(revenue),
     "sub": "12 months of attributed sales"},
    {"label": "Scoreable", "value": f"{scoreable:,}",
     "sub": f"{scoreable/len(df):.0%} with an established ordering rhythm"},
    {"label": "Elevated lapse risk", "value": f"{at_risk:,}",
     "sub": "Above a 50% modelled probability of not ordering next quarter"},
    {"label": "Champions", "value": f"{champions:,}",
     "sub": f"{df.loc[df.segment=='Champions','revenue'].sum()/revenue:.0%} of revenue"},
])
ui.note(
    f"Only {scoreable:,} of {len(df):,} accounts carry a score. The models were "
    "fitted on accounts that ordered in at least three months of a twelve-month "
    "window, and an account outside that population is one they have never seen. "
    "Scoring it anyway would be inventing a judgement, so those accounts show "
    "no score and say why."
)

left, right = st.columns(2, gap="large")
with left:
    st.markdown("#### Revenue by segment")
    profiles = data.load("segment_profiles")
    st.plotly_chart(
        charts.hbar(list(profiles["segment"]), list(profiles["share_of_revenue"]),
                    value_fmt="{:.1f}", label_suffix="%", height=330,
                    hover_label="Share of revenue"),
        width="stretch",
    )
with right:
    st.markdown("#### Accounts by segment")
    st.plotly_chart(
        charts.hbar(list(profiles["segment"]), list(profiles["share_of_customers"]),
                    value_fmt="{:.1f}", label_suffix="%", height=330, colour=ORANGE,
                    hover_label="Share of accounts"),
        width="stretch",
    )
top = profiles.loc[profiles["share_of_revenue"].idxmax()]
ui.note(
    f"The two charts are the finding. <b>{top['segment']}</b> is "
    f"{top['share_of_customers']:.1f}% of accounts and {top['share_of_revenue']:.1f}% "
    "of revenue. A segmentation where the two have the same shape has not found "
    "anything — it has drawn the account count twice."
)

# ------------------------------------------------- 2. why is it happening ---
ui.section(
    "Why is it happening?",
    "What the lapse model keys on, ranked by how much its measured performance "
    "depends on each input.",
)
imp = data.load("model_lapse_importance").head(8)
c1, c2 = st.columns([1.25, 1], gap="large")
with c1:
    st.plotly_chart(
        charts.hbar(list(imp["label"]), list(imp["importance"]),
                    value_fmt="{:.3f}", height=320, hover_label="Drop in ROC-AUC"),
        width="stretch",
    )
    ui.note(
        "The strongest input is <b>how far past its own usual gap</b> an account "
        "has drifted — not how long since it last ordered. A wholesaler ordering "
        "quarterly and one ordering weekly can both be six weeks silent and be in "
        "completely different states. That feature had to be engineered from the "
        "ordering rhythm; it is not in the source data."
    )
with c2:
    lapse = cards["lapse"]
    m = lapse["metrics"]
    ui.panel("Lapse model", [
        ("Cross-validated ROC-AUC",
         f"{m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f}"),
        ("Single held-out split", f"{m['roc_auc']:.3f}"),
        ("Precision-recall AUC", f"{m['pr_auc']:.3f}"),
        ("Recall at operating point", ui.pct(m["recall"])),
        ("Lift over the base rate", f"{m['lift']:.1f}×"),
        ("Base rate", ui.pct(lapse["base_rate"])),
        ("Mean predicted / observed", f"{m['mean_predicted']:.3f} / {m['observed_rate']:.3f}"),
        ("Eligible accounts", f"{lapse['eligible']:,}"),
    ])
    ui.note(
        f"The quoted figure is the <b>cross-validated</b> one. A single split on "
        f"{cards['lapse']['n_test']:,} held-out rows put it at "
        f"{m['roc_auc']:.3f} — about {m['roc_auc'] - m['cv_roc_auc_mean']:.02f} "
        "higher, and the same was true of the growth model. On samples this size "
        "a single split is one draw, not a measurement, and reporting the "
        "flattering draw is how a model gets oversold."
    )

# --------------------------------------------- 3. where is the opportunity --
ui.section(
    "Where is the opportunity?",
    "Lapse risk against relationship opportunity. The two are scored separately "
    "and deliberately: a valuable account that is quietly going quiet is the "
    "case a single combined ranking would hide.",
)
m1, m2 = st.columns([1.4, 1], gap="large")
with m1:
    plot = df[df["lapse_risk"].notna()]
    st.plotly_chart(
        charts.risk_opportunity_matrix(plot["opportunity_score"], plot["lapse_risk"],
                                       risk_cut=0.5, height=420),
        width="stretch",
    )
    ui.note(
        f"Plotted for the {len(plot):,} scoreable accounts only. The remaining "
        f"{len(df) - len(plot):,} are shown as <b>Not scored</b> rather than "
        "placed somewhere convenient — an account the model declined to score "
        "cannot be put in a risk quadrant without inventing the number that "
        "would put it there."
    )
with m2:
    st.markdown("#### The four groups")
    quad = df["quadrant"].value_counts()
    for name in ["Retain", "Grow", "Monitor", "Develop", "Not scored"]:
        n = int(quad.get(name, 0))
        st.markdown(
            f'<div class="ci-panel" style="padding:.75rem .95rem;margin-bottom:.55rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<strong style="color:{QUADRANT_COLOURS[name]};">{name}</strong>'
            f'<span style="font-variant-numeric:tabular-nums;color:#5a6169;">'
            f"{n:,} · {n/len(df):.0%}</span></div>"
            f'<div style="font-size:.82rem;color:#5a6169;margin-top:.3rem;line-height:1.45;">'
            f"{QUADRANT_MEANING[name]}</div></div>",
            unsafe_allow_html=True,
        )

# --------------------------------------- 4. what should we consider doing? --
ui.section(
    "What should we consider doing?",
    "The decision engine turns the scores into a suggested action per account. "
    "These are suggestions for a person to weigh, not automated decisions.",
)
actions = df["action"].value_counts().reset_index()
actions.columns = ["action", "accounts"]
a1, a2 = st.columns([1.3, 1], gap="large")
with a1:
    st.plotly_chart(
        charts.hbar(list(actions["action"]), list(actions["accounts"]),
                    value_fmt="{:,.0f}", height=340, hover_label="Accounts"),
        width="stretch",
    )
with a2:
    cost = df["cost"].value_counts()
    ui.panel("Where the effort would fall",
             [(k, f"{int(v):,} · {v/len(df):.0%}") for k, v in cost.items()])
    am = df[df["cost"] == "Account manager"]
    ui.panel("Account-manager caseload", [
        ("Accounts", f"{len(am):,}"),
        ("Share of the book", f"{len(am)/len(df):.1%}"),
        ("Revenue represented", ui.money(am["revenue"].sum())),
        ("Share of revenue", f"{am['revenue'].sum()/revenue:.0%}"),
    ])
    ui.note(
        f"{len(am)/len(df):.0%} of accounts reach a named account manager, and "
        f"they hold {am['revenue'].sum()/revenue:.0%} of revenue. An engine that "
        "routes most of the book to human contact has produced a wish list, not "
        "a plan, so the check belongs on the page rather than in a footnote."
    )

# ------------------------------------------------------------ foundations ---
ui.rule()
ui.section(
    "Can we trust the data underneath?",
    "This is a real, widely used, published dataset. Its defects are real too.",
)
ui.tiles([
    {"label": "Record-level conformance", "value": f"{quality['score']:.2f}%",
     "sub": "Severity-weighted pass rate across all lines"},
    {"label": "Rules passing", "value": f"{quality['checks_passed']} of {quality['checks_run']}",
     "sub": "Automated checks across seven dimensions"},
    {"label": "Lines with no customer", "value": "235,143",
     "sub": "22.6% of the ledger cannot be attributed to anyone"},
    {"label": "Exact duplicate lines", "value": "34,058",
     "sub": "Re-run of a load, not repeat purchases"},
    {"label": "Accounts affected", "value": f"{quality['customers_affected_pct']:.0f}%",
     "sub": f"{quality['customers_affected']:,} touched by at least one defect"},
])
ui.note(
    "None of this was injected. A fifth of the ledger carries no customer "
    "identifier, 34,058 lines are exact duplicates, 3,427 negative quantities "
    "sit on invoices that are not credit notes, and seventeen rows are labelled "
    '"This is a test product". Every decision taken about them is logged on '
    "<b>Data quality &amp; governance</b>, because a pipeline that quietly drops "
    "a fifth of its input has changed every number downstream and nobody can tell."
)

st.markdown(
    '<p class="ci-note" style="margin-top:2.4rem;">'
    "Built by Nichodemus Amollo. Data: Chen, D. (2012), <em>Online Retail II</em>, "
    "UCI Machine Learning Repository, CC BY 4.0. "
    "The domain may change — the analytical problem does not.</p>",
    unsafe_allow_html=True,
)
