"""Data quality & governance — whether any of the rest of it can be trusted."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.page import setup

setup("Data quality & governance")

from components import charts, data, ui                                        # noqa: E402
from components.theme import BLUE, ORANGE, STATUS                              # noqa: E402
from customer_intelligence.data_quality.report import DIMENSION_DEFINITIONS    # noqa: E402

summary = data.quality_summary()
results = data.load("quality_results")
dimensions = data.load("quality_dimensions")
impact = data.load("quality_impact")
conformance = data.load("conformance_log")
cards = data.model_cards()
rec_card = data.recommender_card()
sources = data.sources()
run = data.run_summary()

ui.page_header(
    "Data quality & governance",
    "Can we trust what is underneath?",
    "This is a real, published, widely used dataset. Its defects are real too — "
    "none of them were injected, and every decision taken about them is on this "
    "page.",
)
ui.source_notice()

# ------------------------------------------------------------- sources -----
ui.section("Where the data came from")
attribution = pd.DataFrame(sources["attribution"])
st.dataframe(attribution, width="stretch", hide_index=True)

prov = sources.get("provenance", {})
if prov:
    rows = [(v.get("name", k), f"{v.get('bytes', 0)/1e6:,.1f} MB",
             v.get("sha256", "")[:16] + "…", v.get("licence", ""))
            for k, v in prov.items()]
    st.markdown("#### What was actually downloaded")
    st.dataframe(
        pd.DataFrame(rows, columns=["Dataset", "Size", "SHA-256", "Licence"]),
        width="stretch", hide_index=True,
    )
ui.note(
    "Nothing is redistributed in the repository. Each file is fetched from its "
    "original publisher at build time and cached by content hash, so a source "
    "that changes underneath the project is noticed rather than silently "
    "changing every figure downstream. CC BY 4.0 requires attribution as a "
    "condition of use, which is why the credit is on every page and not in an "
    "about box."
)

# ------------------------------------------------------------ headline -----
ui.section("The headline, and why one number is not enough")
ui.tiles([
    {"label": "Record-level conformance", "value": f"{summary['score']:.2f}%",
     "sub": "Severity-weighted pass rate across all lines"},
    {"label": "Rules passing", "value": f"{summary['checks_passed']} of {summary['checks_run']}",
     "sub": "Automated checks across seven dimensions"},
    {"label": "Lines affected", "value": f"{summary['rows_affected']:,}",
     "sub": "Across 1,041,845 invoice lines"},
    {"label": "Accounts affected", "value": f"{summary['customers_affected_pct']:.1f}%",
     "sub": f"{summary['customers_affected']:,} touched by at least one defect"},
])
ui.note(
    f"Conformance reads {summary['score']:.2f}% because it is a row-weighted "
    "average, and at these defect rates such a number is arithmetically bound to "
    "look reassuring. It is the wrong number to lead with. The one that changes "
    "what the analysis can claim is the next section: <b>a fifth of the ledger "
    "cannot be attributed to any customer at all</b>."
)

# -------------------------------------------------------- what is wrong -----
ui.section("What is actually wrong with this data")
failing = results[results["failing"] > 0].sort_values("failing", ascending=False)
st.plotly_chart(
    charts.hbar(list(failing["rule"].head(10)), list(failing["failing"].head(10)),
                value_fmt="{:,.0f}", height=400, colour=ORANGE,
                hover_label="Failing rows"),
    width="stretch",
)
st.markdown(
    """
Four of these are worth stopping on, because they are the kind of thing that
quietly invalidates an analysis rather than breaking it:

- **235,143 lines carry no customer identifier** — 22.6% of the ledger. That
  revenue is real and it is in the totals, but it belongs to nobody, so no
  customer-level figure on this site describes it. Every per-account number here
  is computed on the 77% that can be attributed, and saying so is the difference
  between a measurement and a claim.
- **3,427 negative quantities sit on invoices that are not credit notes.**
  Returns booked through the sales table without a credit note. Netting them off
  would make a customer who buys £10,000 and returns £9,000 look identical to one
  who buys £1,000 and returns nothing.
- **1,215 stock codes carry more than one description.** The same product,
  described differently on different lines — which is why the product taxonomy is
  built on the modal description per code rather than on whatever arrived first.
- **Seventeen lines are literally described as "This is a test product".**
  Alongside an Amazon commission of −£260,764 and a bad-debt write-off of
  −£147,614, both booked as ordinary invoice lines.
"""
)

ui.section("The seven dimensions")
d1, d2 = st.columns([1.2, 1], gap="large")
with d1:
    st.markdown("#### Shortfall from full conformance")
    dims = dimensions.copy()
    dims["shortfall"] = 100.0 - dims["score"]
    dims = dims.sort_values("shortfall", ascending=False)
    st.plotly_chart(
        charts.hbar(list(dims["dimension"]), list(dims["shortfall"]),
                    value_fmt="{:.2f}", label_suffix="%", height=320, colour=ORANGE,
                    hover_label="Shortfall from 100%"),
        width="stretch",
    )
    ui.note(
        "Plotted as the shortfall, not the pass rate. Every dimension scores "
        "above 95%, so a chart of the scores themselves is seven near-identical "
        "bars that hide the differences it exists to show."
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

ui.section("Every check, and what it found")
show = results[["dimension", "table", "column", "rule", "severity", "failing",
                "total", "pass_rate", "status"]].copy()
show.columns = ["Dimension", "Table", "Column", "Rule", "Severity", "Failing rows",
                "Total rows", "Pass rate", "Status"]
st.dataframe(
    show.style.format({"Failing rows": "{:,.0f}", "Total rows": "{:,.0f}",
                       "Pass rate": "{:.4%}"}),
    width="stretch", hide_index=True, height=440,
)

# ------------------------------------------------ validating the checks -----
ui.section(
    "How do we know the checks work?",
    "On real data the answer is not known in advance, so the checks cannot be "
    "graded against it. They are graded somewhere else instead.",
)
st.markdown(
    """
The framework is validated against **synthetic control data** with sixteen
deliberately injected defect types in known quantities — missing keys, duplicate
records, impossible ages, orphaned rows, invalid statuses, future dates,
inconsistent codes. The reconciliation of *injected* against *detected* is a
test, and the build fails if any injected defect type escapes.

The complementary test matters just as much: the same rules are run over the
**undamaged** synthetic data and the injected-defect rules must all pass.
Without it, a check that always fires would look like a working detector.

```bash
make validate-checks     # 16 defect types injected, 16 detected
```

That is the order of operations: prove the instrument reads correctly against a
known answer, then point it at data where the answer is unknown.
"""
)

# ----------------------------------------------------------- conformance ----
ui.section(
    "What the cleanse did about it",
    "Nothing is dropped silently. Every line that does not survive into the "
    "conformed layer is accounted for here, with the decision that removed it.",
)
conf = conformance.copy()
conf.columns = ["Table", "Rule", "Rows", "Decision taken"]
st.dataframe(conf.style.format({"Rows": "{:,.0f}"}),
             width="stretch", hide_index=True, height=460)

# ------------------------------------------------------ model governance ----
ui.rule()
ui.section("Model governance",
           "Two classifiers and one recommender. Each was chosen, thresholded "
           "and bounded deliberately.")

for slug, name in [("lapse", "Lapse risk"), ("growth", "Growth propensity")]:
    card = cards[slug]
    m = card["metrics"]
    with st.expander(
            f"{name} — cross-validated ROC-AUC {m['cv_roc_auc_mean']:.3f} "
            f"± {m['cv_roc_auc_std']:.3f}, lift {m['lift']:.1f}×",
            expanded=(slug == "lapse")):
        g1, g2 = st.columns([1, 1.2], gap="large")
        with g1:
            ui.panel("Performance", [
                ("Cross-validated ROC-AUC",
                 f"{m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f}"),
                ("— folds", f"{m['cv_folds']} (5-fold, 3 repeats)"),
                ("Single held-out split", f"{m['roc_auc']:.3f}"),
                ("Precision-recall AUC", f"{m['pr_auc']:.3f}"),
                ("Brier score", f"{m['brier']:.4f}"),
                ("Mean predicted", f"{m['mean_predicted']:.4f}"),
                ("Observed rate", f"{m['observed_rate']:.4f}"),
                ("Precision at threshold", f"{m['precision']:.1%}"),
                ("Recall at threshold", f"{m['recall']:.1%}"),
                ("Lift over base rate", f"{m['lift']:.1f}×"),
                ("Operating threshold", f"{card['threshold']:.3f}"),
                ("Eligible accounts", f"{card['eligible']:,}"),
                ("Train / test", f"{card['n_train']:,} / {card['n_test']:,}"),
            ])
        with g2:
            st.markdown("**Calibration — when it says 40%, does 40% happen?**")
            st.plotly_chart(charts.calibration(data.load(f"model_{slug}_calibration"),
                                               height=290), width="stretch")
            st.markdown("**Model comparison**")
            st.dataframe(
                data.load(f"model_{slug}_comparison").style.format(
                    {"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "Brier": "{:.4f}"}),
                width="stretch", hide_index=True,
            )
        st.markdown("**What the model relies on**")
        imp = data.load(f"model_{slug}_importance").head(10)
        st.plotly_chart(
            charts.hbar(list(imp["label"]), list(imp["importance"]),
                        value_fmt="{:.4f}", height=330,
                        hover_label="Drop in ROC-AUC when shuffled"),
            width="stretch",
        )

with st.expander("Next-best-product recommender — and the model it replaced"):
    prod = rec_card["product_level"]["5"] if "5" in rec_card["product_level"] else \
           list(rec_card["product_level"].values())[0]
    cat = rec_card["category_level"]
    r1, r2 = st.columns(2, gap="large")
    with r1:
        ui.panel("Product level — where the problem is real", [
            ("Products considered", f"{rec_card['products_considered']:,}"),
            ("Accounts evaluated", f"{prod['accounts_evaluated']:,}"),
            ("Hit rate @5", f"{prod['hit_rate_at_k']:.3f}"),
            ("Popularity baseline @5", f"{prod['popularity_baseline']:.3f}"),
            ("Lift over baseline",
             f"{prod['hit_rate_at_k']/max(prod['popularity_baseline'],1e-9):.2f}×"),
        ])
    with r2:
        ui.panel("Category level — where it is not", [
            ("Categories", "12"),
            ("Accounts evaluated", f"{cat.get('accounts_evaluated', 0):,}"),
            ("Hit rate @3", f"{cat.get('hit_rate_at_k', 0):.3f}"),
            ("Popularity baseline @3", f"{cat.get('popularity_baseline', 0):.3f}"),
            ("Verdict", "No better than popularity"),
        ])
    st.markdown(f"\n{rec_card['why']}")
    st.markdown(
        "\nA **category-expansion classifier** was also built and scored ROC-AUC "
        "0.550 — barely above chance. It is not shipped. The reason it failed is "
        "the same reason the category recommender failed, and it is structural "
        "rather than a modelling problem: there is very little left to predict "
        "when the median account already buys nine of twelve categories. The "
        "answer was to change the question, not to reach for a bigger model."
    )

ui.note(
    "Both models are quoted on a <b>cross-validated</b> ROC-AUC with its spread, "
    "not on a single held-out split. The single split flattered both by "
    "0.03–0.04, and on a sample of this size that is roughly one standard error "
    "— enough to report as an improvement something that is only a different "
    "shuffle. The spread is on the page so the reader can judge how much of any "
    "future change is real."
)

ui.section("Governance principles applied here")
principles = [
    ("Attribution", "CC BY 4.0 requires credit as a condition of use. The source, "
     "publisher, licence and citation are shown on every page and carried in code "
     "in sources/registry.py, so they cannot drift out of date."),
    ("Provenance", "Each file is downloaded from its original publisher at build "
     "time and cached by SHA-256. A source that changes upstream is reported, not "
     "silently absorbed."),
    ("Leakage control", "Training features come from months 1–12 with the outcome "
     "in 13–15; scoring features from months 13–24 with the outcome unobserved. "
     "Both snapshots are produced by the same SQL with a different window. A test "
     "greps the SQL for any month literal reaching into the outcome window."),
    ("Seasonal alignment", "The windows were chosen so both outcome periods fall "
     "in December–February. This business takes three times as much revenue in "
     "November as in February; a model trained on one season and applied to "
     "another has learned the wrong base rate."),
    ("Population eligibility", "The models were fitted only on accounts with an "
     "established ordering rhythm and refuse to score anyone else. Ineligible "
     "accounts get null, not zero — zero is a prediction, null is an admission."),
    ("Nothing dropped silently", "Every line removed by the cleanse is counted "
     "with the rule and the reasoning, including the 235,143 that cannot be "
     "attributed to a customer."),
    ("Negative results published", "The category recommender and the "
     "category-expansion classifier both failed. Both are reported, with the "
     "diagnosis, because the diagnosis is the useful part."),
    ("Human oversight", "The system produces suggested actions with the reasoning "
     "attached and the constraints stated. It decides nothing."),
]
for title, body in principles:
    st.markdown(
        f'<div class="ci-panel" style="padding:.85rem 1rem;margin-bottom:.55rem;">'
        f'<strong style="font-size:.92rem;">{title}</strong>'
        f'<div style="font-size:.86rem;color:#5a6169;margin-top:.3rem;line-height:1.6;">'
        f"{body}</div></div>",
        unsafe_allow_html=True,
    )

ui.section("Limitations")
st.markdown(
    """
- **One business, two years, one country of origin.** A UK giftware wholesaler
  selling largely to trade customers. Nothing here generalises to retail
  consumers, to other sectors, or to other periods without being re-fitted.
- **The lapse label is behavioural, not contractual.** A wholesale customer
  cannot cancel; they can only stop ordering. "No order in a quarter" is a
  reasonable proxy and it is not the same thing, which is why the second domain
  exists.
- **A fifth of the ledger is unattributable**, so every per-account figure
  describes the 77% that can be attributed.
- **The product taxonomy is derived**, by keyword rules over description text.
  It covers 87.7% of revenue in eleven named categories. It is a judgement,
  published in `sources/retail.py`, not a fact about the data.
- **1,752 accounts train the models.** That is a small sample by modern
  standards, and it is why the reported figures carry the uncertainty they do.
- **No causal claim is supported anywhere.** Permutation importance measures
  what a model relies on, not what causes an account to lapse.
"""
)
