# From Data to Decisions
### Customer Intelligence & Decision Analytics

*A six-slide walkthrough. Built on Online Retail II — Chen, D. (2012), UCI
Machine Learning Repository, CC BY 4.0.*

**Live dashboard: [https://nic-customer-intelligence.streamlit.app/](https://nic-customer-intelligence.streamlit.app/)**

---

## 1 — The problem

**A transaction ledger knows about transactions. It knows nothing about
customers.**

The source is 1,067,371 invoice lines from a UK giftware wholesaler. Almost
everything interesting about it is a problem: a fifth of the lines belong to
nobody, returns are booked through the sales table, the same product is described
three ways, and somewhere in there is a row that says *"This is a test product"*.

The questions the business needs answered cannot be asked of that ledger:

- Which accounts have **stopped** ordering, as against those that order quarterly
  and are simply between orders?
- Which are likely to **grow**, in a business that takes three times the revenue
  in November that it takes in February?
- Which deserve a **phone call** this week, and which should be served by a
  catalogue?

The gap is not between data and reporting. It is between reporting and a decision
someone can be held to.

---

## 2 — The data foundation

**Nothing above this layer is worth anything if this layer is wrong.**

| Real defect | Count |
|---|---|
| Lines with no customer identifier | **235,143** (22.6%) |
| Exact duplicate lines | **34,058** |
| Negative quantity on a non-credit-note invoice | 3,427 |
| Stock codes with conflicting descriptions | 1,215 |
| Rows described as "This is a test product" | 17 |

> **Conformance reads 97.3%. A fifth of the ledger belongs to nobody.**
>
> Both describe the same data. The first is a row-weighted average and is
> arithmetically bound to look reassuring. The second changes what every
> per-account figure in this project can claim to describe — and it is the one
> that should be led with.

Every cleansing decision is logged. Returns are separated rather than netted off,
because an account that buys £10,000 and returns £9,000 is not the same
relationship as one that buys £1,000 and returns nothing.

**The checks themselves are validated** against synthetic control data with
sixteen injected defects in known quantities. Prove the instrument reads
correctly against a known answer, then point it at data where the answer is
unknown.

---

## 3 — The intelligence layer

**The design decision that mattered most was not a model. It was a window.**

This business takes 84,711 invoice lines in November and 27,707 in February. So:

```
train   features months  1–12 → outcome 13–15   (Dec – Feb)
score   features months 13–24 → outcome 25–27   (Dec – Feb, unobserved)
```

Both feature windows are twelve whole months, so seasonality averages out inside
them. **Both outcome windows are the same three calendar months**, so the model
is applied to the season it was trained on. A test enforces it.

The same reasoning fixed the growth target: "beats its own annual run-rate" is
wrong in a low quarter, where most accounts fall below their average whatever they
do. Comparing **the same quarter a year earlier** moved ROC-AUC from 0.62 to 0.68
on identical features.

| Model | Result |
|---|---|
| Lapse risk | Cross-validated ROC-AUC **0.734 ± 0.034**, lift 1.66× |
| Growth propensity | Cross-validated ROC-AUC **0.684 ± 0.030**, lift 2.00× |
| Next best product | hit@5 **0.314** vs 0.211 popularity — **1.49× lift** |

Quoted cross-validated, with the spread. A single split flattered both by
0.03–0.04 — about one standard error, and enough to report as an improvement
something that is only a different shuffle.

---

## 4 — From prediction to action

**A probability is not a decision.**

The decision engine is a transparent rule set, not a model: a recommendation has
to be arguable by the account manager who receives it, the rules encode
commercial policy that should change deliberately, and learning actions from last
year learns last year's policy including its mistakes.

```
1. Service failure       →  a 20%+ return rate is not a sales opportunity
2. Too new to judge      →  onboarding, not a low-value label
3. Not scored            →  says so, rather than implying a judgement never made
4. Retention             →  graded by what the account is worth and how far gone
5. Growth                →  only on an account that is not going anywhere
6. Deliberately nothing  →  a reachable outcome, recorded as a decision
```

Only **1,703 of 5,914** accounts have an established ordering rhythm. The rest
carry no score and say why. Filling their risk with 1.0 to make the arithmetic
work would flag them as high-risk on the strength of a number the model
explicitly declined to produce.

**17% of accounts reach a named account manager**, holding **53% of revenue**.
An engine routing much more than that to human contact has produced a wish list,
not a plan.

---

## 5 — Governance

**The controls are properties of the code, enforced by tests.**

- **Attribution.** CC BY 4.0 makes credit a condition of use. Source, licence and
  citation are carried in code and shown on every page.
- **Provenance.** Downloaded from the publisher, cached by SHA-256. A source that
  changes upstream is reported, not absorbed.
- **Leakage.** Training and scoring features come from the *same SQL* with a
  different window. A test greps for month literals reaching forward.
- **Eligibility.** Models refuse to score outside the population they were fitted
  on. Ineligible accounts get null, not zero.
- **Calibration.** Both models calibrated in the large to three decimal places.
- **Negative results published.** A category classifier scored 0.550 and a
  category recommender lost to a popularity baseline. Both are reported with the
  diagnosis — the median account already buys 9 of 12 categories — because the
  diagnosis is the useful part.

---

## 6 — What this demonstrates

| Capability | Where it shows |
|---|---|
| **Data acquisition & provenance** | Publisher downloads, hash caching, licence-aware attribution |
| **Data engineering** | Mixed types, admin codes, returns, a derived taxonomy — on real mess |
| **Analytical architecture** | Layered model, window-parameterised portable SQL |
| **Data quality & governance** | 22 rules on real defects, validated against a synthetic control |
| **Statistical modelling** | Cross-validated with spreads, calibrated, negative results published |
| **Decision science** | Thresholds set by capacity; risk and opportunity kept separate |
| **Data products** | An application that reads artefacts, so numbers are reproducible |

```
Data management  →  Analytics  →  Statistical modelling
                 →  Decision support  →  Analytics products
```

> **The domain may change. The analytical problem remains: turning complex data
> into better decisions.**
