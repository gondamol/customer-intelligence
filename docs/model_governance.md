# Model governance

Two classifiers and one recommender are in production. This document states, for
each, what it is for, what it was trained on, how it performs, where it should
not be used, and what would have to be monitored if it were real.

> All figures come from **Online Retail II** (Chen, D., 2012; UCI Machine
> Learning Repository; CC BY 4.0) — one UK giftware wholesaler, two years. The
> outputs are decision **support**. Nothing here is an automated decision.

---

## 1. Purpose, and what each is not

| Model | Predicts | Explicitly not |
|---|---|---|
| **Lapse risk** | Probability an established account places no order next quarter | A statement that the account has left, or a licence to act without asking why |
| **Growth propensity** | Probability its next-quarter spend exceeds the same quarter a year earlier | A revenue forecast, or a target |
| **Next best product** | Which product to lead with | A prediction that they will buy it |

## 2. Training data

- **Source:** Online Retail II, 1,067,371 invoice lines, 24 whole months.
- **Feature window:** months 1–12 (Dec 2009 – Nov 2010).
- **Outcome window:** months 13–15 (Dec 2010 – Feb 2011).
- **Preprocessing:** median imputation and standardisation for numeric inputs;
  most-frequent imputation and one-hot encoding for country, categories below 1%
  frequency grouped, unseen categories ignored at scoring time.

Features are declared in one place — `analytics/features_retail.py` — so "what
went into this model?" is answered by reading a single list.

## 3. Target definitions

**Lapse.** An established account placed no order at all across the whole
three-month outcome window.

The word "established" is doing real work. Applied to every account that ordered
once, the apparent lapse rate is **67%** — which is not attrition, it is a
wholesaler's ordinary order cadence. Restricting to accounts that ordered in at
least three distinct months of the twelve-month window brings it to **45.5%**,
where a silent quarter is genuinely informative.

This is a **behavioural** definition. A wholesale customer cannot cancel a
contract; they can only stop ordering. The figures below are not comparable to a
churn model built on account closure, and should not be read as though they were.

**Growth.** Outcome-quarter revenue exceeded **the same quarter a year earlier**.

The obvious target — beating the account's own annual quarterly run-rate — is
wrong in a seasonal business. The outcome window is December–February, the low
season, so most accounts fall below their annual average whatever they do, and
the model spends its capacity learning the calendar. Comparing like quarter with
like quarter moved ROC-AUC from **0.62 to 0.68** on identical features. The cost
is a smaller eligible population: an account must have traded in the prior-year
quarter to have anything to be measured against.

## 4. Eligible populations

Each model is trained only on accounts that could have experienced its outcome,
and **refuses to score anyone else**.

| Model | Eligible | Why |
|---|---|---|
| Lapse | ≥3 active months in the window (1,752) | Below that, a quiet quarter is normal trading |
| Growth | The above, and traded in the prior-year quarter (1,168) | Nothing to compare against otherwise |

Only **1,703 of 5,914** accounts are scoreable in the scoring window. The rest
carry **null, not zero** — zero is a prediction, null is an admission — sit in a
"Not scored" quadrant, and their recommendation says why.

This is not a technicality. Before it was enforced, the account-manager workload
was 45% of the book, most of it accounts the models had never seen anything like.
It is now 15%, and those accounts hold 48% of revenue.

## 5. Performance

Regenerated on every build into `models/model_cards.json`; the application reads
them there rather than reproducing them by hand.

| Model | Cross-validated ROC-AUC | Single split | Base rate | Lift | Calibration (predicted / observed) |
|---|---|---|---|---|---|
| Lapse risk | **0.735 ± 0.035** | 0.760 | 45.5% | 1.70× | 0.453 / 0.454 |
| Growth propensity | **0.680 ± 0.033** | 0.717 | 24.1% | 1.66× | 0.240 / 0.240 |

**On why the cross-validated figure is the one quoted.** Re-running this project
with the rows in a different order moved a held-out AUC by 0.05. On ~440 test
rows that is roughly two standard errors — enough to report as an improvement
something that is only a different shuffle. Both models are quoted as a
repeated-stratified-CV mean (5 folds × 3 repeats) with its spread. The single
split flattered both by 0.03–0.04, in the same direction, which is exactly what
one would expect and exactly why it should not be the headline.

**On the absolute level.** 0.735 is a real result on real data, and it is well
below the 0.93 the synthetic version of this project reached. The synthetic
number was higher because the process that produced the data was knowable.

**On calibration.** A model used to decide who gets contacted must be right about
the *level*, not only the order. Both are calibrated in the large to three
decimal places; reliability by decile is shown in the application.

## 6. The recommender, and the two models it replaced

A **category-expansion classifier** was built first: will an account buy from a
category it has never bought? ROC-AUC **0.550** — barely above chance.

A **category-level recommender** was built next. Hit rate at 3 of **0.927**
against a popularity baseline of **0.936** — it lost to recommending whatever is
popular.

Both fail for the same structural reason, and finding it mattered more than
either model: **the median account already buys 9 of the 12 categories**. "The
top three you don't buy" is most of what is left, so there is almost nothing to
predict and the evaluation is close to degenerate.

Reframed to **product** level — 400 products, median account buys 18 — the
problem is real:

| k | Recommender | Popularity baseline | Lift |
|---|---|---|---|
| 5 | 0.314 | 0.211 | **1.49×** |
| 10 | 0.457 | 0.325 | 1.41× |
| 20 | 0.615 | 0.475 | 1.30× |

Item-to-item cosine similarity over the account × product matrix, normalised by
how broad the account's range already is so a wide-ranging account does not
out-score a narrow one on every candidate. Evaluated as hit-rate at k against
what accounts actually bought next quarter, **against a popularity baseline** —
the benchmark any recommender has to beat to have earned its complexity.

The answer was to change the question, not to reach for a bigger model.

## 7. Model selection

Logistic regression, random forest and gradient boosting are trained on every
target and compared on held-out data. The comparison is published in the
application.

**Logistic regression ships.** Every recommendation carries a per-account
explanation built from standardised coefficients times the account's distance
from the book average — readable, additive, and checkable by hand. A small gain
in discrimination did not justify losing it. The comparison table is published
precisely so that judgement can be made on evidence rather than taste.

## 8. Explainability

| Method | Where | What it answers |
|---|---|---|
| Standardised coefficients | Model card | Which inputs move the score, and how, across the population |
| Permutation importance | Model card | What the model *relies on* |
| Coefficient × z-score | Per account | Why *this* account scored as it did |

**Importance is not causation.** Permutation importance measures what the model
depends on, not what causes an account to lapse. Nothing here licenses a causal
claim.

The strongest input to the lapse model is **`cadence_overdue`** — how far past
its own usual gap an account has drifted — ahead of raw recency. A wholesaler
ordering quarterly and one ordering weekly can both be six weeks silent and be in
entirely different states. That feature is engineered from the ordering rhythm;
it is not in the source data.

## 9. Leakage controls

1. Training and scoring features are produced by **the same SQL file** with a
   different `analysis_window`.
2. Both outcome windows fall in the same calendar months, so the model is applied
   to the season it was trained on. Enforced by test.
3. `segment` is excluded: RFM is a rule over features the model already sees, and
   including it would make the explanations circular.
4. The growth model's prior-year comparison is drawn from months *before* the
   feature window closes, never from the outcome window.
5. `tests/test_retail.py` builds the 360 over two windows and asserts the results
   differ in the right direction, and greps the SQL for month literals reaching
   into the outcome window.

## 10. Thresholds

Not 0.5. Thresholds are set so each model flags a fixed share of the eligible
book — 25% for each classifier — which is a **capacity decision**, declared in
`analytics/models.py::CAPACITY_SHARE`.

The decision engine consumes those thresholds rather than the descriptive
probability bands, so the capacity assumption is made once and the engine cannot
recommend outreach to more accounts than the models were thresholded to flag.

## 11. Monitoring, if this were real

| Signal | Why | Action if it moves |
|---|---|---|
| Feature distribution stability | Inputs drift before outputs do | Investigate the source system first |
| Calibration by decile | A model can keep its ranking and lose its level | Recalibrate; ranking metrics will not show this |
| Share of the book scoreable | If eligibility collapses, the models cover less of the business | Re-examine the eligibility rule, not the model |
| Precision at the operating threshold | Determines whether outreach is worth doing | Re-examine the capacity assumption |
| Account-manager caseload | Rules can shift effort without any model changing | Review the rule ordering |
| Recommender hit-rate vs popularity | If it stops beating the baseline, it has stopped earning its place | Retire it and use popularity |

## 12. Limitations

1. **One business, two years, one country of origin.** A UK giftware wholesaler
   selling largely to trade. Nothing generalises without re-fitting.
2. **The lapse label is behavioural, not contractual.**
3. **22.6% of the ledger is unattributable**, so every per-account figure
   describes the 77% that can be attributed.
4. **1,752 accounts train the lapse model, 1,168 the growth model.** Small, and
   the reason the reported spreads are what they are.
5. **The product taxonomy is derived** by keyword rules — a published judgement.
6. **Two years is two Christmases.** Enough to align the windows seasonally, not
   enough to model seasonality itself.
7. **No causal claim is supported anywhere.**

## 13. Human oversight

The system produces suggested actions. It decides nothing.

Every recommendation names the rule that fired, the evidence beneath it, the
constraints on acting, and a confidence grade based on how much *trading history*
exists — not on how extreme the score is. Doing nothing is a reachable outcome
and is recorded as a decision, because an engine that always finds something to
do is generating contact rather than insight.
