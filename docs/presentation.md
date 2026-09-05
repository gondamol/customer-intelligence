# From Data to Decisions
### Customer Intelligence & Decision Analytics

*A six-slide walkthrough. Synthetic demonstration throughout — no real
organisation, customer, or portfolio is represented.*

---

## 1 — The problem

**Customer data is fragmented by construction, and reporting alone does not fix it.**

Every source system is correct about its own domain and none of them knows who
the customer is. A core ledger holds balances. A switch holds card activity. A
digital channel holds logins. A contact centre holds complaints. Each is
authoritative and none is sufficient.

The consequence is not that reporting is hard. The consequence is that the
questions an organisation actually needs answered cannot be asked at all:

- Which relationships are quietly ending, while the balance still looks healthy?
- Where is there genuine headroom, as opposed to a large existing balance?
- Which customers should be contacted this week — and by whom, at what cost?

A dashboard that reports what happened last month answers none of these. The gap
is not between data and reporting. It is between reporting and **a decision
someone can act on and be held to**.

---

## 2 — The data foundation

**Nothing above this layer is worth anything if this layer is wrong.**

Seven source tables — customers, accounts, balances, transactions, loans,
products, digital activity, service interactions — assembled into one analytical
customer view over a bounded time window.

The data was **deliberately damaged first**: sixteen defect types injected in
known quantities, then twenty-two automated checks run against the result and
reconciled against the injection manifest. The framework is graded against a
known answer rather than judged by eye.

> **99.4% of records conform. 73% of customers are affected.**
>
> Both figures describe the same data. A row-weighted average is arithmetically
> bound to look reassuring at these defect rates; defects are not spread evenly
> across customers. Reporting only the first would be technically accurate and
> materially misleading — which is the most common failure of a quality
> dashboard.

Every cleansing decision is logged with the count of rows it moved. Anomalous
amounts are quarantined rather than capped, because a capped value is a
fabricated one. Impossible ages are nulled rather than imputed, because a
consumer should see missing data rather than inherit a guess.

---

## 3 — The intelligence layer

**Four questions, four methods, each chosen for the question rather than for the technique.**

| Question | Method | Why this one |
|---|---|---|
| Who are these customers? | Published rule set, checked against K-means | The clustering scored a silhouette of 0.15 — the book has no natural clusters. If the segments are a management choice either way, they should be the stable, explicable choice |
| Which relationships are ending? | Logistic regression, ROC-AUC 0.93, 6.7× lift | A behavioural label observed after the feature window. High, and the reason is stated |
| Who will take what? | Two propensity models on eligible populations only | Trained on the whole book, a propensity model learns eligibility rather than propensity |
| Where is the headroom? | Published five-component heuristic | There is no ground truth for "opportunity". A model that appeared to find one would be fitting to last year's sales |

Attrition risk and opportunity are scored **separately and deliberately**.
Collapsing them into one ranking would hide the case that matters most: a
valuable relationship that is quietly ending.

---

## 4 — From prediction to action

**A probability is not a decision. The gap between them is where most analytics projects stop.**

The decision engine is a transparent rule set, not a model — for three reasons.
A recommendation has to be arguable, so the person acting on it can see which
condition fired and say it is wrong. The rules encode policy, and policy should
be written down and changed deliberately. And learning actions from historical
outcomes learns the historical policy, including its mistakes.

The ordering *is* the policy:

```
1. Credit position        →  nobody in arrears is sold anything
2. Unresolved service     →  fix the failure before any commercial conversation
3. Retention              →  graded by what the relationship is worth
4. Activation             →  early-tenure customers who never established a pattern
5. Growth                 →  only on a stable relationship
6. Deliberately nothing   →  a reachable outcome, recorded as a decision
```

Every recommendation carries the rule that fired, the model reasons beneath it,
the constraint on acting, and a confidence grade based on **how much history
exists** — not on how extreme the score is. A decisive-looking score on three
months of thin activity is the case to treat most carefully.

Routing is checked as an output: what share of the book reaches a relationship
manager, the scarcest channel. An engine that routes most of the book to human
contact has produced a wish list, not a plan.

---

## 5 — Governance

**The controls are properties of the code, enforced by tests — not a policy appended to a finished system.**

- **Leakage control.** Training features come from months 1–12, the outcome from
  13–15; scoring features from months 4–15, outcome unobserved. Both snapshots
  come from the *same SQL* with a different window, so the definitions cannot
  drift apart. A test greps the SQL for any month literal reaching forward.
- **Protected attributes.** Gender is in the source data and shown in the
  profile. It is not an input to any model, and a test fails the build if that
  changes. "It was in the data" is not a justification for targeting on it.
- **Population eligibility.** Each model scores only customers it could have been
  trained on. Ineligible customers get null, not zero — zero is a prediction,
  null is an admission.
- **Data minimisation.** No names, addresses, contact details or identifiers were
  ever generated. A test enforces it.
- **Calibration, not just discrimination.** A model used to decide who gets
  contacted must be right about the level, not only the order.
- **Auditability.** Every threshold and weight is a named constant. Every dropped
  row is counted. The whole build is reproducible from a fixed seed.
- **Human oversight.** The system suggests. It decides nothing.

Limitations are stated rather than buried: the data is synthetic, the panel is
fifteen months, the attrition label is behavioural and therefore easier than a
commercial one, and no causal claim is supported anywhere.

---

## 6 — What this demonstrates

| Capability | Where it shows |
|---|---|
| **Analytical data architecture** | Layered model, window-parameterised SQL, portable to PostgreSQL |
| **Data engineering** | Reproducible pipeline, one command per stage, artefacts on disk |
| **Data quality & governance** | 22 checks graded against 16 injected defects; every cleansing decision logged |
| **Statistical modelling** | Three models, honestly evaluated, calibration reported, weakest result published rather than dropped |
| **Decision science** | Thresholds set by capacity, not convention; risk and opportunity kept separate |
| **Data products** | An application that reads artefacts and computes nothing, so yesterday's number is reproducible |
| **Stakeholder communication** | An executive view that answers four questions and sends the reader elsewhere for the fifth |

The progression this represents:

```
Data management  →  Analytics  →  Statistical modelling
                 →  Decision support  →  Analytics products
```

> **The domain may change. The analytical problem remains: turning complex data
> into better decisions.**
