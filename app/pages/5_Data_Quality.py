"""Data quality & governance — whether any of the rest of it can be trusted."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Data quality & governance")

from components import charts, data, ui                    # noqa: E402
from components.theme import BLUE, ORANGE, STATUS          # noqa: E402
from customer_intelligence.data_quality.report import DIMENSION_DEFINITIONS  # noqa: E402

summary = data.quality_summary()
results = data.load("quality_results")
dimensions = data.load("quality_dimensions")
impact = data.load("quality_impact")
reconciliation = data.load("quality_reconciliation")
conformance = data.load("conformance_log")
cards = data.model_cards()

ui.page_header(
    "Data quality & governance",
    "Can we trust what is underneath?",
    "The raw data in this project was deliberately damaged before it was "
    "measured. That is the only way to know whether a quality framework detects "
    "what it claims to — a set of checks that has never been shown a defect "
    "proves nothing.",
)
ui.synthetic_notice()

# --------------------------------------------------------- the headline -----
ui.section("The headline, and why one number is not enough")

ui.tiles([
    {"label": "Record-level conformance", "value": f"{summary['score']:.2f}%",
     "sub": "Severity-weighted pass rate across all rows"},
    {"label": "Rules passing", "value": f"{summary['checks_passed']} of {summary['checks_run']}",
     "sub": "Automated checks across seven dimensions"},
    {"label": "Customers affected", "value": f"{summary['customers_affected_pct']:.1f}%",
     "sub": f"{summary['customers_affected']:,} touched by at least one defect"},
    {"label": "Rows affected", "value": f"{summary['rows_affected']:,}",
     "sub": "Across every source table"},
    {"label": "Defect types caught", "value": f"{summary['defects_caught']} of {summary['defects_injected']}",
     "sub": "Checked against the injection manifest"},
])

ui.note(
    f"<b>These tiles describe the same data and tell opposite stories.</b> "
    f"Conformance is {summary['score']:.2f}% because it is a row-weighted "
    f"average and defect rates are under one percent per rule — at that scale "
    f"the number is always near 100 and always reassuring. But defects are not "
    f"spread evenly: one damaged transaction row belongs to a customer, and "
    f"{summary['customers_affected_pct']:.0f}% of customers carry at least one. "
    "Reporting the first figure alone would be technically accurate and "
    "materially misleading, which is the most common failure of a quality "
    "dashboard — so both are shown, and the second is the one that should drive "
    "remediation."
)

# ------------------------------------------------------------ dimensions ----
ui.section("The seven dimensions")

d1, d2 = st.columns([1.2, 1], gap="large")
with d1:
    st.markdown("#### Shortfall from full conformance")
    dims = dimensions.copy()
    dims["shortfall"] = 100.0 - dims["score"]
    dims = dims.sort_values("shortfall", ascending=False)
    st.plotly_chart(
        charts.hbar(list(dims["dimension"]), list(dims["shortfall"]),
                    value_fmt="{:.2f}", label_suffix="%", height=320,
                    colour=ORANGE, hover_label="Shortfall from 100%"),
        width="stretch",
    )
    ui.note(
        "Plotted as the shortfall from 100%, not the pass rate itself. Every "
        "dimension scores between 98.4% and 99.9%, so a bar chart of the scores "
        "is seven identical bars that hide the very differences it is there to "
        "show. Inverting it puts the largest problem at the top, which is also "
        "the order remediation would work in."
    )
with d2:
    st.markdown("#### What each dimension asks")
    for name, definition in DIMENSION_DEFINITIONS.items():
        st.markdown(
            f'<div style="padding:.32rem 0;font-size:.85rem;">'
            f'<strong style="color:#14171a;">{name}</strong> '
            f'<span style="color:#5a6169;">— {definition}</span></div>',
            unsafe_allow_html=True,
        )

# ------------------------------------------------------- customer impact ----
ui.section(
    "How many customers each defect actually touches",
    "The question an owner asks, which a row-level pass rate cannot answer.",
)
top_impact = impact.head(10)
st.plotly_chart(
    charts.hbar(list(top_impact["rule"]), list(top_impact["customers_affected"]),
                value_fmt="{:,.0f}", height=380, colour=ORANGE,
                hover_label="Customers affected"),
    width="stretch",
)

# ----------------------------------------------------------- every check ----
ui.section("Every check, and what it found")

show = results[["dimension", "table", "column", "rule", "severity", "failing",
                "total", "pass_rate", "status"]].copy()
show.columns = ["Dimension", "Table", "Column", "Rule", "Severity",
                "Failing rows", "Total rows", "Pass rate", "Status"]
st.dataframe(
    show.style.format({
        "Failing rows": "{:,.0f}", "Total rows": "{:,.0f}", "Pass rate": "{:.4%}",
    }),
    width="stretch", hide_index=True, height=440,
)

# --------------------------------------------------------- reconciliation ---
ui.section(
    "Grading the checks against a known answer",
    "Because the defects were injected deliberately, exactly how many of each "
    "were introduced is known. Detected against injected is the test of the "
    "framework itself.",
)

rec = reconciliation.copy()
rec["Result"] = rec["caught"].map({True: "Caught", False: "Missed"})
rec_display = rec[["table", "defect", "dimension", "injected", "detected", "Result"]]
rec_display.columns = ["Table", "Injected defect", "Dimension", "Injected",
                       "Detected", "Result"]
st.dataframe(
    rec_display.style.format({"Injected": "{:,.0f}", "Detected": "{:,.0f}"}),
    width="stretch", hide_index=True,
)

over = rec[rec["detected"] > rec["injected"]]
if not over.empty:
    example = over.iloc[0]
    ui.note(
        f"Detected exceeds injected in {len(over)} case"
        f"{'s' if len(over) > 1 else ''} — for instance "
        f"<b>{example['defect'].lower()}</b>, where "
        f"{int(example['detected']):,} were found against {int(example['injected']):,} "
        "introduced. That is not a false positive: nulling a customer's "
        "identifier orphans every account beneath it, so one injected defect "
        "manufactures others. Defects cascade across a relational model, and a "
        "framework that only counted what it expected would have missed the "
        "downstream damage entirely."
    )

# ------------------------------------------------------------ conformance ---
ui.section(
    "What the cleanse did about it",
    "Nothing is dropped silently. Every row that does not survive into the "
    "conformed layer is accounted for here, with the decision that removed it.",
)
conf = conformance.copy()
conf.columns = ["Table", "Rule", "Rows", "Decision taken"]
st.dataframe(
    conf.style.format({"Rows": "{:,.0f}"}),
    width="stretch", hide_index=True, height=440,
)
ui.note(
    "Two decisions are worth arguing with. Anomalous transaction amounts are "
    "<b>quarantined rather than capped</b>: a capped value is a fabricated one "
    "that then flows into every downstream average. And impossible ages are set "
    "to <b>null rather than imputed</b>, so a consumer sees missing data and "
    "decides what to do, rather than inheriting a guess it cannot distinguish "
    "from a measurement."
)

# ----------------------------------------------------- model governance -----
ui.rule()
ui.section(
    "Model governance",
    "Three models are in production here. Each was chosen, thresholded and "
    "bounded deliberately.",
)

for slug, name in [("churn", "Attrition risk"), ("investment", "Investment propensity"),
                   ("lending", "Lending propensity")]:
    card = cards[slug]
    m = card["metrics"]
    with st.expander(f"{name} — ROC-AUC {m['roc_auc']:.3f}, lift {m['lift']:.1f}×",
                     expanded=(slug == "churn")):
        g1, g2 = st.columns([1, 1.2], gap="large")
        with g1:
            ui.panel("Performance on held-out data", [
                ("Discrimination (ROC-AUC)", f"{m['roc_auc']:.3f}"),
                ("Precision-recall AUC", f"{m['pr_auc']:.3f}"),
                ("Brier score", f"{m['brier']:.4f}"),
                ("Precision at threshold", f"{m['precision']:.1%}"),
                ("Recall at threshold", f"{m['recall']:.1%}"),
                ("F1 at threshold", f"{m['f1']:.3f}"),
                ("Lift over base rate", f"{m['lift']:.1f}×"),
                ("Base rate", f"{card['base_rate']:.2%}"),
                ("Operating threshold", f"{card['threshold']:.3f}"),
                ("Eligible population", f"{card['eligible']:,}"),
                ("Train / test", f"{card['n_train']:,} / {card['n_test']:,}"),
            ])
        with g2:
            st.markdown("**Calibration — when it says 20%, does 20% happen?**")
            st.plotly_chart(
                charts.calibration(data.load(f"model_{slug}_calibration"), height=290),
                width="stretch",
            )
            st.markdown("**Model comparison**")
            comp = data.load(f"model_{slug}_comparison")
            st.dataframe(
                comp.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "Brier": "{:.4f}"}),
                width="stretch", hide_index=True,
            )
            tm = confusion = card["confusion"]
            st.markdown(
                f"**Confusion matrix at the operating threshold** — "
                f"true negatives {confusion[0][0]:,}, false positives {confusion[0][1]:,}, "
                f"false negatives {confusion[1][0]:,}, true positives {confusion[1][1]:,}."
            )

        st.markdown("**What the model relies on**")
        imp = data.load(f"model_{slug}_importance").head(10)
        st.plotly_chart(
            charts.hbar(list(imp["label"]), list(imp["importance"]),
                        value_fmt="{:.4f}", height=330,
                        hover_label="Drop in ROC-AUC when shuffled"),
            width="stretch",
        )

ui.note(
    "Logistic regression is in production for all three despite gradient "
    "boosting scoring marginally higher on two of them. The comparison table is "
    "shown rather than hidden so the trade-off is visible: a small gain in "
    "discrimination did not justify losing the per-customer explanation on the "
    "<b>Decision support</b> page. Where the gap were large, that judgement "
    "should be revisited — which is why the number is on the page."
)

# ---------------------------------------------------------- the principles --
ui.section("Governance principles applied here")

principles = [
    ("Data minimisation",
     "The generated population carries no names, addresses, or contact details. "
     "None are needed to demonstrate any of the analytics, so none exist — the "
     "cheapest privacy control is data that was never collected."),
    ("Leakage control",
     "Training features come from months 1–12 and the outcome is observed in "
     "13–15. Scoring features come from months 4–15 and the outcome is the "
     "unobserved future. Both snapshots are produced by the same SQL with a "
     "different window, so the definitions cannot drift apart."),
    ("Population eligibility",
     "Each model is trained only on customers who could experience its outcome, "
     "and refuses to score anyone outside that population. A propensity model "
     "trained on the whole book learns eligibility, not propensity."),
    ("Auditability",
     "Every transformation is SQL in version control; every dropped row is "
     "counted in the conformance log; every threshold and weight is a constant "
     "in a configuration file, not a number embedded in a calculation."),
    ("Lineage",
     "Raw → conformed → monthly panel → 360 view → features → models → decision "
     "engine, each stage a separate artefact on disk. Any number on any page "
     "can be traced back to the rows that produced it."),
    ("Human oversight",
     "The system produces suggested actions with the reasoning attached and the "
     "constraints stated. It decides nothing. Every recommendation names the "
     "rule that fired so the person acting on it can disagree."),
    ("Model monitoring",
     "What would be tracked in operation: population stability across the "
     "feature distributions, calibration drift by decile, precision at the "
     "operating threshold, and the share of the book flagged. A model that "
     "starts flagging 30% of customers has changed, whatever its AUC says."),
    ("Known limitations",
     "The data is synthetic and generated from a known process, so relationships "
     "are cleaner than reality. Performance figures here are a property of that "
     "process and are not a forecast of performance on a real portfolio."),
]
for title, body in principles:
    st.markdown(
        f'<div class="ci-panel" style="padding:.85rem 1rem;margin-bottom:.55rem;">'
        f'<strong style="font-size:.92rem;">{title}</strong>'
        f'<div style="font-size:.86rem;color:#5a6169;margin-top:.3rem;line-height:1.6;">'
        f"{body}</div></div>",
        unsafe_allow_html=True,
    )
