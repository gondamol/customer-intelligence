# Customer Intelligence & Decision Analytics
### From data to decisions

An end-to-end analytics product that turns fragmented customer data into
customer intelligence, predictive insight, and explainable decision support.

**Synthetic demonstration.** Every customer, transaction and score in this
project was generated. Nothing here describes a real organisation, customer, or
portfolio, and no output is a financial, credit, or customer decision. The
methodology transfers to any data-intensive customer environment — financial
services, telecoms, insurance, retail, healthcare.

---

## The problem

Customer data is fragmented by construction. A core ledger holds balances, a
switch holds card activity, a digital channel holds logins, a contact centre
holds complaints. Every one is authoritative about its own domain and none of
them knows who the customer is.

The consequence is not that reporting is hard. It is that the questions worth
answering cannot be asked:

- Which relationships are quietly ending, while the balance still looks healthy?
- Where is there genuine headroom, as opposed to a large existing balance?
- Which customers should be contacted this week, by whom, at what cost?

This project builds the layers required to answer them, and stops at a
**suggested action with its reasoning attached** — not at a chart.

## Why it matters

The distance between a dashboard and a decision is where most analytics work
stops. Closing it takes four things that are usually treated as separate
disciplines, and this project does all four deliberately:

1. **Data you can defend.** Quality measured before anything is built on it, with
   every cleansing decision logged.
2. **A model of the customer, not of the tables.** One analytical view, bounded
   by an explicit time window.
3. **Predictions that are honest about their limits.** Calibration reported, weak
   results published, thresholds set by capacity rather than convention.
4. **A decision layer someone can argue with.** Transparent rules, stated policy
   ordering, constraints attached to every recommendation.

## Architecture

```
                          SOURCE SYSTEMS
        ┌───────────────┬───────┴────────┬────────────────┐
        ▼               ▼                ▼                ▼
    customers       accounts        transactions      engagement
    products        balances        loans             interactions
        └───────────────┴───────┬────────┴────────────────┘
                                ▼
                          RAW LAYER            as landed, defects intact
                                │
                   22 checks · 7 dimensions
                                ▼
                       CONFORMED LAYER         every dropped row logged
                                ▼
                        MONTHLY PANEL          customer × month, full spine
                                ▼
                         CUSTOMER 360          window-parameterised
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        SEGMENTATION       PREDICTION        OPPORTUNITY
              └─────────────────┼─────────────────┘
                                ▼
                        DECISION ENGINE        transparent, ordered rules
                                ▼
                       STREAMLIT APPLICATION   reads artefacts, computes nothing
```

Full detail in [`docs/architecture.md`](docs/architecture.md).

## The data

50,000 synthetic customers over 15 months, across eight related source tables.

Nothing is drawn independently. Each customer has latent traits — affluence,
digital affinity, credit appetite, relationship stability — and every observable
quantity is a noisy function of those traits. That is what makes segmentation
find real structure and churn predictable but not trivially so. The traits are
never written to disk; models see only what an analyst would see.

Behaviour that follows from this rather than being hard-coded:

- income band rises monotonically with balances, transaction value, product
  holdings and digital engagement;
- digital engagement falls monotonically with age;
- customers heading for attrition show declining balances, falling transaction
  frequency, reduced logins **and** rising complaints, together;
- about a third of declines **recover**, and a further group leaves abruptly
  inside the outcome window — the irreducible error that keeps the achievable
  performance honest.

### Deliberate defects

Sixteen defect types are injected in known quantities: missing values, duplicate
customers and transactions, invalid dates, impossible ages, negative balances,
invalid statuses, orphaned records, inconsistent product codes, missing keys, and
anomalous amounts.

This is the point of the exercise. A quality framework that has never been shown
a defect proves nothing.

## Analytics

| Layer | Approach | Result |
|---|---|---|
| **Segmentation** | Published rules, checked against K-means | Silhouette 0.14 — no natural clusters exist. Reported as a finding, and as the argument for rules |
| **Attrition** | Logistic regression | ROC-AUC ~0.92, PR-AUC ~0.75, 6.6× lift |
| **Investment propensity** | Logistic regression, eligible population only | ROC-AUC ~0.83, 3.8× lift |
| **Lending propensity** | Logistic regression, eligible population only | ROC-AUC ~0.67, 2.0× lift — modest, and published rather than dropped |
| **Opportunity** | Five-component published heuristic | 0–100 relative score. Not a revenue figure |

Random forest and gradient boosting are trained alongside every model and their
scores published. **Logistic regression ships** — a marginal gain in
discrimination did not justify losing the per-customer explanation that makes a
recommendation arguable. The comparison table is shown so the trade-off can be
judged on evidence.

## The decision engine

A transparent rule set, not a model. A recommendation has to be arguable; the
rules encode policy, which should be changed deliberately; and learning actions
from historical outcomes learns the historical policy, including its mistakes.

The ordering is the policy:

```
1. Credit position       →  nobody in arrears is sold anything
2. Unresolved service    →  fix the failure before any commercial conversation
3. Retention             →  graded by what the relationship is worth
4. Activation            →  early-tenure customers with no established pattern
5. Growth                →  only on a stable relationship
6. Deliberately nothing  →  a reachable outcome, recorded as a decision
```

Every recommendation carries the rule that fired, the model reasons beneath it,
the constraint on acting, and a confidence grade based on how much history exists
for that customer — not on how extreme the score is.

## Governance

Controls are properties of the code, enforced by tests.

- **Leakage.** Training features from months 1–12, outcome from 13–15; scoring
  features from months 4–15, outcome unobserved. Both produced by the *same SQL*
  with a different window. A test greps the SQL for any month literal reaching
  into the outcome window.
- **Protected attributes.** Gender is in the source data, shown in the profile,
  and is **not** an input to any model. A test fails the build if that changes.
- **Eligibility.** Each model scores only the population it was trained on.
  Ineligible customers get null, not zero.
- **Data minimisation.** No names, addresses, contact details or identifiers are
  ever generated. A test enforces it.
- **Nothing dropped silently.** Every removed row is counted with its reason.

Detail in [`docs/data_governance.md`](docs/data_governance.md) and
[`docs/model_governance.md`](docs/model_governance.md).

### One result worth pausing on

> **99.4% of records conform. 73% of customers are affected.**

Both describe the same data. A row-weighted average is arithmetically bound to
look reassuring at these defect rates; defects are not spread evenly across
customers. Reporting only the first figure would be technically accurate and
materially misleading — so both are on the application's landing page, and the
second is the one that drives remediation.

## Technology

| | |
|---|---|
| Language | Python 3.11 |
| Data | pandas, NumPy, PyArrow, Parquet |
| Query engine | DuckDB (default), PostgreSQL-compatible via `CI_POSTGRES_DSN` |
| Modelling | scikit-learn |
| Application | Streamlit |
| Charts | Plotly |
| Testing | pytest |

DuckDB is the default because it needs no server, so the project runs on a laptop
with nothing installed. The SQL is ordinary analytical SQL and avoids
engine-specific syntax — the analysis window is a one-row table rather than a
DuckDB variable for exactly that reason.

## Running locally

```bash
git clone https://github.com/gondamol/customer-intelligence.git
cd customer-intelligence

make setup      # create the environment (Python 3.11 via uv)
make all        # generate, assess quality, build, model and score
make run        # open the application
```

`make all` takes roughly seven minutes on 50,000 customers. For a quicker look:

```bash
make demo       # the same pipeline on 8,000 customers
```

Individual stages:

```bash
make generate   # synthetic source data + defect injection
make quality    # 22 checks, scored and reconciled against the manifest
make build      # conformed layer, monthly panel, Customer 360
make train      # models, scores, opportunity, recommendations
make test       # the test suite
```

To run against PostgreSQL instead of local parquet:

```bash
export CI_POSTGRES_DSN="postgresql://user:pass@host:5432/dbname"
make build
```

## The application

| Page | Answers |
|---|---|
| **Executive view** | What is happening, why, where the opportunity is, what to consider doing |
| **Customer 360** | Everything known about one relationship, on one screen |
| **Customer segments** | Who the customers are — rules against clustering |
| **Risk & opportunity** | Which relationships are changing, and which are worth the effort |
| **Decision support** | The evidence, the drivers, the suggested action, the constraints |
| **Data quality & governance** | Whether any of the rest can be trusted |

The application **reads artefacts and computes nothing**. Every page is
interactive because the work already happened, and the number a reader saw
yesterday can be reproduced today because it is a file rather than the output of
a fit that ran while they were looking.

## Testing

```bash
make test
```

Seven suites covering schema and referential integrity, the quality framework
graded against the injection manifest, leakage controls, feature generation,
model behaviour, the decision rules, and the SQL layer.

The tests worth reading first:

- `test_data_quality.py::test_every_injected_defect_is_detected` — grades the
  framework against a known answer.
- `test_data_quality.py::test_clean_data_passes_the_checks_it_should` — without
  it, a check that always fires would look like a working detector.
- `test_leakage.py::test_customer_360_never_reads_outside_its_window` — builds
  the 360 over two windows and asserts the results differ in the right direction.
- `test_leakage.py::test_gender_is_not_a_model_feature` — fails the build if a
  protected attribute re-enters the model.
- `test_decision_engine.py::test_stable_unremarkable_customer_gets_no_action` —
  doing nothing must be reachable, or the engine generates contact rather than
  insight.

## Limitations

Stated plainly, because a demonstration that hides its boundaries is worth less
than one that names them.

1. **The data is synthetic**, generated from a known process. Relationships are
   cleaner than reality; there are no seasonal effects, no macroeconomic shocks,
   no mid-panel data migrations.
2. **Performance figures are a property of that process** and are not a forecast
   of performance on a real portfolio.
3. **The attrition label is behavioural** ("went quiet"), not commercial
   ("closed the account"). Behaviour that begins inside the feature window and
   continues into the outcome window is easier to predict, which is the main
   reason the reported AUC is high. It is the label that is easy, not the model
   that is exceptional.
4. **Fifteen months is a short panel.** No model here has seen a full annual
   cycle, so nothing seasonal could have been learned.
5. **The opportunity score is a heuristic**, not a fitted model. There is no
   ground truth for "opportunity"; a model that appeared to find one would be
   fitting to last year's sales.
6. **The decision engine encodes one set of policy assumptions.** A different
   organisation would order the rules differently, and should.
7. **No causal claim is supported anywhere.** Every relationship here is
   associational. Permutation importance measures what the model relies on, not
   what causes attrition.

## What this demonstrates

| Capability | Where |
|---|---|
| Analytical data architecture | Layered model, window-parameterised portable SQL |
| Data engineering | Reproducible pipeline, one command per stage |
| Data quality & governance | 22 checks graded against 16 injected defects |
| Statistical modelling | Three models, calibration reported, weakest result published |
| Decision science | Thresholds set by capacity; risk and opportunity kept separate |
| Data products | An application that reads artefacts, so numbers are reproducible |
| Stakeholder communication | An executive view that answers four questions and no more |

```
Data management  →  Analytics  →  Statistical modelling
                 →  Decision support  →  Analytics products
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Layers, the two-snapshot design, repository layout |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every field: meaning, type, allowed values, sensitivity, validation |
| [`docs/data_governance.md`](docs/data_governance.md) | Quality dimensions, cleansing decisions, access model, lineage |
| [`docs/model_governance.md`](docs/model_governance.md) | Purpose, targets, eligibility, performance, fairness, monitoring, limitations |
| [`docs/presentation.md`](docs/presentation.md) | Six-slide executive walkthrough |

---

Built by **Nichodemus Amollo** as a portfolio demonstration.
[Portfolio](https://gondamol.github.io) · [GitHub](https://github.com/gondamol)

> The domain may change. The analytical problem remains: turning complex data
> into better decisions.
