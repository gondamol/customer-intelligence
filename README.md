# Customer Intelligence & Decision Analytics
### From data to decisions

An end-to-end analytics product built on a **real, openly published transaction
ledger**: two years of invoices from a UK giftware wholesaler, turned into
account intelligence, predictive scores, and an explainable suggested action for
a person to weigh.

### ▶ **[Open the live dashboard](https://nic-customer-intelligence.streamlit.app/)**

**[Architecture](docs/architecture.md)** · **[Model governance](docs/model_governance.md)** · **[Data governance](docs/data_governance.md)** · **[Data dictionary](docs/data_dictionary.md)**

> **Data source.** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii),
> UCI Machine Learning Repository — Chen, D. (2012), used under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 1,067,371 real
> invoice lines. Scores and recommendations are this project's own work and are
> decision **support**: nothing here is an automated decision.

---

## The problem

Customer data arrives as a flat ledger of transactions, and almost everything
interesting about it is a problem. A fifth of the lines belong to nobody.
Returns are booked through the sales table. The same product is described three
different ways. Somewhere in there is a row that says *"This is a test product"*.

The questions a business actually needs answered cannot be asked of that ledger
directly:

- Which accounts have quietly stopped ordering — as distinct from the ones that
  order quarterly and are simply between orders?
- Which are likely to grow, in a business where a third of the year's revenue
  arrives in two months?
- Which deserve a phone call this week, and which should be served by a catalogue?

This project builds the layers required to answer them and stops at a **suggested
action with its reasoning attached**.

## Why real data changed the project

An earlier version of this ran on data it generated itself. The architecture is
unchanged; what the numbers mean is not.

| | Synthetic version | On Online Retail II |
|---|---|---|
| Attrition model | ROC-AUC 0.93 | **0.734 ± 0.034** |
| Data quality defects | 16 types, injected | Real, and worse |
| Unattributable revenue | none | **22.6% of the ledger** |
| Seasonality | none | 3× between February and November |

The synthetic figure was higher because the process that produced the data was
knowable. The real figure is the honest one, and every awkward property below is
something the invented data did not have.

## The data engineering is the substance

```
Invoice / StockCode mix integers with alphanumeric codes
  → read as numbers, every credit note and admin row silently disappears

0.57% of stock codes are not products
  → postage, carriage, manual adjustments, discounts, samples, bank charges,
    an Amazon commission of −£260,764, bad-debt write-offs, gift vouchers,
    and 17 rows described as "This is a test product"

`M` and `m` are the same manual-adjustment code
  → unified for admin codes only; product codes are case-significant
    (84031A and 84031B are different items)

22,950 negative quantities
  → returns SEPARATED, not netted off. An account that buys £10,000 and returns
    £9,000 is not the same relationship as one that buys £1,000 and returns none

5,305 stock codes, no categories
  → a derived keyword taxonomy carries 87.7% of revenue into 11 named
    categories. A published judgement, not a fact about the data
```

## Architecture

```
              Online Retail II (UCI, CC BY 4.0)
                  downloaded at build time, cached by SHA-256
                                ▼
                          RAW LAYER              as published, defects intact
                                │
                    22 checks · 7 dimensions
                                ▼
                       CONFORMED LAYER           every dropped line logged
                                ▼
                        MONTHLY PANEL            account × month, silence recorded
                                ▼
                         CUSTOMER 360            window-parameterised
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
         RFM SEGMENTS      LAPSE / GROWTH    OPPORTUNITY + RECOMMENDER
              └─────────────────┼─────────────────┘
                                ▼
                        DECISION ENGINE          transparent, ordered rules
                                ▼
                       STREAMLIT APPLICATION     reads artefacts, computes nothing
```

## Windows chosen for season, not convenience

This business takes **84,711 invoice lines in November 2011 and 27,707 in
February**. That single fact determines the experimental design:

```
train   features months  1–12  (Dec 09 – Nov 10) → outcome 13–15 (Dec 10 – Feb 11)
score   features months 13–24  (Dec 10 – Nov 11) → outcome 25–27 (Dec 11 – Feb 12, unobserved)
```

Both feature windows are twelve whole months, so seasonality averages out inside
them. **Both outcome windows are December–February**, so the model is applied to
the season it was trained on. A test enforces it.

The same reasoning fixed the growth target. "Beats its own annual quarterly
run-rate" is wrong in a seasonal business — in a low quarter most accounts fall
below their annual average whatever they do, and the model spends its capacity
learning the calendar. Comparing **the same quarter a year earlier** moved
ROC-AUC from 0.62 to 0.68 on identical features.

## What the models do

| | Question | Result |
|---|---|---|
| **Lapse risk** | Will an established account place no order next quarter? | Cross-validated ROC-AUC **0.734 ± 0.034**, lift 1.66× |
| **Growth propensity** | Will it beat the same quarter a year earlier? | Cross-validated ROC-AUC **0.684 ± 0.030**, lift 2.00× |
| **Next best product** | What should we lead with? | hit@5 **0.314** vs 0.211 popularity baseline — **1.49× lift** |

Both classifiers are quoted **cross-validated with a spread**, not on a single
split. On this sample a single split flattered both by 0.03–0.04 — about one
standard error — which is enough to report as an improvement something that is
only a different shuffle.

Both are calibrated in the large to three decimal places (predicted 0.453 vs
observed 0.454; 0.239 vs 0.240).

### One defect a ranking metric could not see

An account that had ordered the previous month, and was active in 8 of 12 months,
was scored at a **97.6% probability of lapsing**. Its largest month was fourteen
standard deviations above the book mean and its returned value twenty — so the
linear model was extrapolating far outside anything it had been trained on and
the sigmoid had saturated.

**ROC-AUC did not move.** It is a ranking metric, the ranking was fine, and only
the scores were nonsense — the same blind spot that hid a calibration failure
earlier in this project.

The fix is standard: `log1p` the heavy-tailed money and count features before
standardising, which is monotone, defined at zero, and already applied before
K-means in this codebase for the same reason. Afterwards that account scores
**0.112**, no account exceeds 0.90, and discrimination is unchanged at 0.734.
`tests/test_retail.py::test_scores_do_not_saturate_on_outliers` fails the build
if it returns.

### Two negative results, published

A **category-expansion classifier** scored ROC-AUC 0.550. A **category-level
recommender** failed to beat a popularity baseline (0.927 vs 0.936). Both fail
for the same structural reason, which took a diagnosis rather than a bigger
model: the median account already buys **9 of the 12 categories**, so "the top 3
you don't buy" is most of what is left.

Reframed to **product** level — where the median account buys 18 of the top 400 —
the problem is real, item-to-item collaborative filtering beats popularity by
1.49×, and it can name the product an account manager should lead with. A
propensity score can rank a customer; only a recommender can name a product.

## The decision engine

A transparent rule set, not a model. A recommendation has to be arguable by the
account manager who receives it; the rules encode commercial policy, which should
be changed deliberately; and learning actions from last year learns last year's
policy, mistakes included.

**The ordering is the policy:**

```
1. Service failure       →  a 20%+ return rate is not a sales opportunity
2. Too new to judge      →  onboarding, not a low-value label
3. Not scored            →  says so, rather than implying a judgement never made
4. Retention             →  graded by what the account is worth and how far gone
5. Growth                →  only on an account that is not going anywhere
6. Deliberately nothing  →  a reachable outcome, recorded as a decision
```

**17% of accounts reach a named account manager**, and they hold **53% of the
revenue**. An engine routing much more than that to human contact has produced a
wish list, not a plan, so the check is on the page rather than in a footnote.

### Accounts the models refuse to score

Only **1,703 of 5,914** accounts have an established ordering rhythm. The rest
carry no score, sit in a **"Not scored"** quadrant, and their recommendation says
why. Filling their risk with 1.0 to make the arithmetic work would put them in
the Retain quadrant — flagged as high risk on the strength of a number the model
explicitly declined to produce.

## Data quality: the defects are real

| Finding | Count |
|---|---|
| Lines with no customer identifier | **235,143** (22.6%) |
| Exact duplicate lines | **34,058** |
| Non-positive unit price on a sale | 6,153 |
| Negative quantity on a non-credit-note invoice | 3,427 |
| Stock codes with conflicting descriptions | 1,215 |
| Rows described as "This is a test product" | 17 |

None of it was injected. **A fifth of the ledger cannot be attributed to any
customer**, so every per-account figure in this project describes the 77% that
can — and says so.

### How do we know the checks work?

On real data the answer is not known in advance, so the checks cannot be graded
against it. They are graded somewhere else:

```bash
make validate-checks     # 16 defect types injected in known quantities, 16 detected
```

The synthetic generator is retained for exactly this. It injects known defects,
and the reconciliation of *injected* against *detected* is a test the build must
pass. The complementary test matters as much: the same rules run over the
**undamaged** data must all pass, or a check that always fires would look like a
working detector.

Prove the instrument reads correctly against a known answer, then point it at
data where the answer is unknown.

## Technology

| | |
|---|---|
| Language | Python 3.11 |
| Data | pandas, NumPy, PyArrow, Parquet |
| Query engine | DuckDB (default), PostgreSQL-compatible via `CI_POSTGRES_DSN` |
| Modelling | scikit-learn |
| Application | Streamlit + Plotly |
| Testing | pytest — 123 tests |

Transformations live in SQL because that is where this work belongs and where a
colleague can change it. DuckDB needs no server, so the project runs on a laptop
with nothing installed; the SQL avoids engine-specific syntax and the analysis
window is a one-row table rather than a DuckDB variable for exactly that reason.

## Running it

```bash
git clone https://github.com/gondamol/customer-intelligence.git
cd customer-intelligence

make setup      # create the environment (Python 3.11 via uv)
make run        # open the application — the derived artefacts are committed
```

The application runs immediately: the artefacts it reads (3.4 MB) are in the
repository. To rebuild everything from the published source:

```bash
make all        # fetch, land, assess quality, build, train, score
```

That downloads Online Retail II from UCI on first run and takes a few minutes.
Individual stages: `make fetch land quality build train`.

Against PostgreSQL instead of local parquet:

```bash
export CI_POSTGRES_DSN="postgresql://user:pass@host:5432/dbname"
make build
```

### Deploying

The repository is deployment-ready for **Streamlit Community Cloud** — the
derived artefacts are committed, so there is no build step and no data download
at start-up.

[**Deploy this app**](https://share.streamlit.io/deploy?repository=gondamol%2Fcustomer-intelligence&branch=main&mainModule=app%2FHome.py)

| Setting | Value |
|---|---|
| Repository | `gondamol/customer-intelligence` |
| Branch | `main` |
| Main file path | `app/Home.py` |
| Python version | 3.11 |

## The application

| Page | Answers |
|---|---|
| **Executive view** | What is happening, why, where the opportunity is, what to consider doing |
| **Account 360** | Everything known about one trading relationship |
| **Segments** | Who these accounts are — RFM, and why not a clustering |
| **Risk & opportunity** | Which accounts are changing, and which are worth the effort |
| **Decision support** | The evidence, the drivers, what to offer, the constraints |
| **Data quality & governance** | Whether any of the rest can be trusted |

The application **reads artefacts and computes nothing**. Every page is
interactive because the work already happened, and yesterday's number can be
reproduced today because it is a file rather than the output of a fit that ran
while someone was looking.

## Screenshots

> **Live:** [https://nic-customer-intelligence.streamlit.app/](https://nic-customer-intelligence.streamlit.app/)

**Executive view** — four questions in order. Champions are 18.3% of accounts and
66.9% of revenue; Lost are 27.4% of accounts and 0.0%.

![Executive view](assets/01-executive-view.png)

**Decision support** — the evidence, what drove it, what to lead with, and the
constraints on acting.

![Decision support](assets/05-decision-support.png)

**Data quality & governance** — real defects, real decisions, and how the checks
themselves were validated.

![Data quality](assets/06-data-quality.png)

**Risk & opportunity** — the matrix, drawn as a density surface.

![Risk and opportunity](assets/04-risk-opportunity.png)

## Testing

```bash
make test      # 123 tests
```

The ones worth reading first:

- `test_retail.py::test_both_outcome_windows_fall_in_the_same_season` — enforces
  the seasonal alignment the whole design rests on.
- `test_retail.py::test_recommender_beats_a_popularity_baseline` — the benchmark
  a recommender must clear to have earned its complexity.
- `test_retail.py::test_mixed_type_columns_are_read_as_strings` — read as
  numbers, every credit note disappears silently.
- `test_data_quality.py::test_every_injected_defect_is_detected` — grades the
  framework against a known answer.
- `test_data_quality.py::test_clean_data_passes_the_checks_it_should` — without
  it, a check that always fires would look like a detector.
- `test_features_and_models.py::test_predicted_probabilities_match_the_observed_rate`
  — catches class weighting applied by reflex, which leaves ranking metrics
  untouched while invalidating every probability downstream.

## Limitations

- **One business, two years, one country.** A UK giftware wholesaler selling
  largely to trade customers. Nothing generalises without re-fitting.
- **The lapse label is behavioural, not contractual.** A wholesale customer
  cannot cancel; they can only stop ordering.
- **A fifth of the ledger is unattributable**, so every per-account figure
  describes the 77% that can be attributed.
- **The product taxonomy is derived** by keyword rules. 87.7% of revenue, eleven
  categories, published in `sources/retail.py`.
- **1,752 accounts train the models** — small, and the reason the reported
  spreads are what they are.
- **No causal claim is supported anywhere.** Permutation importance measures what
  a model relies on, not what causes an account to lapse.

## What this demonstrates

| Capability | Where |
|---|---|
| Data acquisition & provenance | Publisher downloads, SHA-256 caching, licence-aware attribution |
| Data engineering | Mixed types, admin codes, returns, a derived taxonomy — on real mess |
| Analytical architecture | Layered model, window-parameterised portable SQL |
| Data quality & governance | 22 rules on real defects, validated against a synthetic control |
| Statistical modelling | Cross-validated with spreads, calibrated, negative results published |
| Decision science | Thresholds set by capacity; risk and opportunity kept separate |
| Data products | An application that reads artefacts, so numbers are reproducible |

```
Data management → Analytics → Statistical modelling
                → Decision support → Analytics products
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Layers, the two-snapshot design, repository layout |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every field: meaning, type, validation, sensitivity |
| [`docs/data_governance.md`](docs/data_governance.md) | Quality dimensions, cleansing decisions, lineage |
| [`docs/model_governance.md`](docs/model_governance.md) | Targets, eligibility, performance, limitations |
| [`docs/presentation.md`](docs/presentation.md) | Six-slide executive walkthrough |

---

Built by **Nichodemus Amollo**.
[Portfolio](https://gondamol.github.io) · [GitHub](https://github.com/gondamol)

Data: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D — CC BY 4.0.

> The domain may change. The analytical problem remains: turning complex data
> into better decisions.
