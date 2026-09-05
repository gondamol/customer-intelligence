# Data governance

Governance in this project is not a policy document bolted onto a finished
system. It is a set of properties the code holds, most of them enforced by tests,
and this file explains what they are and why each was chosen.

---

## The principle

> A quality framework that has never been shown a defect proves nothing.

So the raw data here is deliberately damaged before it is measured. Sixteen
defect types are injected in known, counted quantities; twenty-two automated
checks run against the result; and the two are **reconciled** — detected against
injected — so the framework is graded against a known answer rather than judged
by eye. `tests/test_data_quality.py` fails the build if any injected defect type
escapes detection.

The complementary test matters just as much: the same checks are run over the
*undamaged* data, and the injected-defect rules must all pass. Without it, a
check that always fires would look like a working detector.

## The seven dimensions

| Dimension | The question | Example rule here |
|---|---|---|
| **Completeness** | Are the values that should be present actually present? | A transaction without an amount cannot be measured |
| **Uniqueness** | Does each real-world thing appear exactly once? | A transaction must be posted exactly once |
| **Validity** | Do values fall inside their permitted range or type? | Age must fall within 18–100 |
| **Consistency** | Do values agree with the controlled vocabulary and each other? | Status must be Active, Dormant or Closed |
| **Integrity** | Do foreign keys resolve to a real parent record? | Every account must belong to a known customer |
| **Timeliness** | Do dates fall inside the reporting window? | No transaction dated after the reporting close |
| **Accuracy** | Are values plausible against the distribution they belong to? | Amounts orders of magnitude beyond the population |

## The number that matters, and the number that misleads

The build reports two figures from the same data:

- **Record-level conformance: ~99.4%**
- **Customers affected by at least one defect: ~73%**

Both are correct. Conformance is a row-weighted average, and with per-rule defect
rates under one percent it is arithmetically bound to sit near 100 — it will
always look reassuring, whatever is actually wrong. But defects are not spread
evenly. One damaged transaction row belongs to a customer, and most customers
carry at least one problem somewhere in their record.

Reporting only the first figure would be technically accurate and materially
misleading. It is the most common failure of a data quality dashboard, and both
numbers appear side by side on the application's landing page for that reason.
The second one is what should drive remediation.

## Defects cascade

Nulling 100 customer identifiers produced 443 intended orphaned accounts — and
the checks found 697. That is not a false positive. An account whose parent
customer lost its key is orphaned too, so one injected defect manufactured
others.

This is worth stating because it is the argument for referential checks running
across the whole model rather than table by table. A framework that only counted
what it expected to find would have missed the downstream damage entirely.

## Cleansing decisions, and why each was made

Every rule in `sql/01_conform.sql` is a decision about what to do with a defect,
and each is logged with the count of rows it affected.

| Defect | Decision | Reasoning |
|---|---|---|
| Missing customer identifier | **Drop the row** | A row with no key cannot be joined or attributed to anyone |
| Duplicate customer records | **Keep the first** | A re-landed batch, not two customers |
| Impossible age | **Set to null** | Not imputed. A consumer should see missing data and decide, rather than inherit a guess it cannot distinguish from a measurement |
| Negative savings balance | **Set to null** | A sign error, not a real overdraft. Excluding it is honest; treating it as debt is not |
| Non-canonical status | **Map to the vocabulary**, else `Unknown` | Recoverable meaning; `Unknown` where it is not |
| Orphaned account | **Drop the row** | Cannot be attributed to a relationship |
| Duplicate transaction | **Keep the earliest** | A retry without idempotency |
| Missing / non-positive amount | **Drop the row** | Contributes no measurable value |
| Anomalous amount | **Quarantine, do not cap** | A capped value is a fabricated one that then flows into every downstream average |
| Future-dated transaction | **Drop the row** | Outside the reporting window |
| Non-canonical product code | **Map to the vocabulary** | Same product, different source spelling |

The two decisions most worth arguing with are marked in the application: **null
rather than impute**, and **quarantine rather than cap**. Both trade a
lower-looking data volume for the guarantee that no number downstream was
invented.

## Nothing is dropped silently

Every row that fails to reach the conformed layer is counted in the conformance
log, with the rule that removed it and the reasoning. That log is a page in the
application, not an appendix.

The distinction is between *cleaning* data and *quietly losing* it. A pipeline
that drops 3% of transactions and says nothing has changed every number
downstream, and nobody can tell.

## Data minimisation

The generated population carries **no names, addresses, contact details, national
identifiers or dates of birth**. None are needed to demonstrate any of the
analytics here, so none exist.

This is not incidental. The cheapest and most reliable privacy control is data
that was never collected, and a demonstration project is exactly where that
discipline should be visible.
`tests/test_schema.py::test_no_personal_identifiers_generated` enforces it: a
future change that added a `name` column would fail the suite.

## Access, if this were real

The demonstration runs locally with no access control, because there is nothing
to protect. The classification work has still been done, in
`docs/data_dictionary.md`, because deciding sensitivity *before* building is what
stops a system from quietly becoming one that should not exist.

| Role | Would see | Would not see |
|---|---|---|
| Analyst | Aggregates, segments, model performance, quality reports | Individual customer records |
| Relationship manager | The customers assigned to them, in full | Anyone else's book |
| Data engineer | Schemas, quality metrics, lineage | Sensitive field values in the clear |
| Model owner | Training data, model cards, monitoring | Production customer contact data |

Under that model, the pages in this application divide cleanly: the executive
view, segments, risk and opportunity, and data quality are all aggregate.
Customer 360 and Decision support are the two that expose an individual, and are
the two that would sit behind a per-customer access check.

## Lineage

```
raw/*.parquet
   → conformed_*            (sql/01_conform.sql,      logged)
   → monthly_customer_metrics (sql/02_monthly_metrics.sql)
   → customer_360_train     (sql/03, window 1–12)
   → customer_360_score     (sql/03, window 4–15)
   → models/*.joblib        (analytics/models.py)
   → customer_360.parquet   (scored, segmented, recommended)
   → the application
```

Each stage is a separate artefact on disk. Any number on any page can be traced
back to the rows that produced it, and rebuilt from the seed in `config.py`.

## Auditability

- Every transformation is SQL in version control.
- Every threshold, weight and band is a named constant in `config.py`, not a
  number embedded in a calculation.
- Every dropped row is counted.
- The whole pipeline is reproducible from a fixed seed: `make all` twice produces
  the same figures.
- The application computes nothing. It reads artefacts, so the number a reader
  saw yesterday can be reproduced today.
