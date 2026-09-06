# Data governance

Governance here is a set of properties the code holds, most of them enforced by
tests, rather than a policy appended to a finished system.

---

## Provenance and licence

| | |
|---|---|
| **Dataset** | Online Retail II |
| **Publisher** | UCI Machine Learning Repository, dataset 502 |
| **Creator** | Chen, D. (2012) |
| **Licence** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Citation** | Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D |

Secondary domains: **Telco Customer Churn** (IBM sample data) and **Bank
Marketing** (Moro, Rita & Cortez, 2014; UCI 222; CC BY 4.0).

Three properties follow from CC BY 4.0 being an **attribution** licence:

1. **Credit is a condition of use, not a courtesy.** The source, publisher,
   licence and citation appear on every page of the application and are carried
   in code, in `sources/registry.py`, so they cannot drift out of date.
2. **Nothing is redistributed.** Each file is fetched from its original publisher
   at build time and cached under `data/external/`, which is never committed. The
   licence travels with the data rather than being assumed.
3. **Derived artefacts may be shared, with attribution.** The 3.4 MB of parquet
   in `data/processed/` are adaptations. They are committed so the application
   can be deployed without a build step, and every page carries the credit.

Downloads are cached by **SHA-256**. A source that changes upstream is reported,
not silently absorbed into every figure downstream.

## The defects are real

The earlier synthetic version of this project injected sixteen defect types to
have something to detect. That is no longer necessary.

| Finding | Count | Share |
|---|---|---|
| Lines with no customer identifier | 235,143 | 22.6% |
| Exact duplicate lines | 34,058 | 3.3% |
| Non-positive unit price on a sale line | 6,153 | 0.6% |
| Lines with no description | 4,367 | 0.4% |
| Negative quantity on a non-credit-note invoice | 3,427 | 0.3% |
| Stock codes with conflicting descriptions | 1,215 | 23% of codes |
| Write-offs and commissions booked as invoice lines | 148 | — |
| Rows described as "This is a test product" | 17 | — |
| Accounts invoicing to more than one country | 13 | — |

Including an Amazon commission of **−£260,764** and a bad-debt write-off of
**−£147,614**, both booked as ordinary invoice lines.

## The number that misleads, and the one that matters

Record-level conformance reads **97.3%**. It is a row-weighted average and at
these defect rates it is arithmetically bound to look reassuring. It is the wrong
number to lead with.

The one that changes what the analysis can claim is this: **22.6% of the ledger
cannot be attributed to any customer at all.** That revenue is real and it is in
the totals, but it belongs to nobody, so no per-account figure in this project
describes it. Every account-level number here is computed on the 77% that can be
attributed, and saying so is the difference between a measurement and a claim.

Both numbers are on the application's landing page for that reason.

## How the checks themselves were validated

On real data the answer is not known in advance, so the checks cannot be graded
against it. They are graded elsewhere.

```bash
make validate-checks
```

The synthetic generator is retained for exactly this purpose. It injects sixteen
defect types in **known quantities**, and the reconciliation of *injected*
against *detected* is a test the build must pass. The complementary test matters
as much: the same rules run over the **undamaged** data must all pass, or a check
that always fires would look like a working detector.

Prove the instrument reads correctly against a known answer, then point it at
data where the answer is unknown.

One finding from that harness is worth keeping: nulling 100 customer identifiers
was designed to orphan 443 accounts, and the checks found 697. Not a false
positive — an account whose parent lost its key is orphaned too. **Defects
cascade**, which is the argument for integrity checks that run across the whole
model rather than table by table.

## The seven dimensions

| Dimension | The question | Example rule here |
|---|---|---|
| **Completeness** | Are values that should be present, present? | Every line should be attributable to a customer |
| **Uniqueness** | Does each real thing appear once? | An identical line should not be posted twice to an invoice |
| **Validity** | Are values inside their permitted range? | A sale line must carry a positive unit price |
| **Consistency** | Do values agree with the vocabulary and each other? | Test rows must not be in a production extract |
| **Integrity** | Do foreign keys resolve? | Every product line must reference a known stock code |
| **Timeliness** | Do dates fall inside the window? | Every line must fall inside the 24-month panel |
| **Accuracy** | Are values plausible against their distribution? | Line value beyond \|£50,000\| |

## Cleansing decisions, and why each was made

Every rule in `retail_01_conform.sql` is a decision, logged with the count of
lines it affected.

| Defect | Decision | Reasoning |
|---|---|---|
| No customer identifier | **Drop from the account layer, keep in revenue** | Real money that belongs to nobody. It cannot enter a customer-level model without being invented onto someone |
| Exact duplicate line | **Keep one** | A re-run of a load. A customer does not buy the same item twice in the same second at the same price — those would share an invoice but differ in line |
| Non-positive unit price | **Drop** | Stock adjustments and write-offs booked through the sales table |
| No description | **Drop** | Cannot be classified to a product or category |
| Negative quantity | **Separate as a return, do not net off** | Netting makes a £10,000-buy/£9,000-return account identical to a £1,000-buy/no-return one |
| Administrative stock codes | **Classify and flag, keep** | Postage and commission are real money, but they are not products |
| `M` vs `m` | **Upper-case administrative codes only** | Product codes are case-significant: 84031A and 84031B are different items |
| Multi-country account | **Flag, do not resolve** | With no source of truth, picking one would be a guess |
| Partial final month | **Exclude** | A third of a month reads as a collapse in demand |

The two most arguable are marked in the application: **separate rather than net**,
and **flag rather than resolve**. Both trade a tidier-looking dataset for the
guarantee that nothing downstream was invented.

## Nothing is dropped silently

Every line that fails to reach the conformed layer is counted in the conformance
log, with the rule that removed it and the reasoning. That log is a page in the
application, not an appendix.

The distinction is between *cleaning* data and *quietly losing* it. A pipeline
that drops a fifth of its input and says nothing has changed every number
downstream and nobody can tell.

## Data minimisation

Online Retail II contains **no names, addresses, contact details or national
identifiers**. Customers are five-digit pseudonymous identifiers. Nothing further
is needed to demonstrate any of the analytics here, and nothing further is
derived. `tests/test_schema.py::test_no_personal_identifiers_generated` enforces
the equivalent property on the synthetic control data.

## Access, if this were real

The demonstration runs with no access control because there is nothing to
protect. The classification work is still done, in `docs/data_dictionary.md`,
because deciding sensitivity *before* building is what stops a system quietly
becoming one that should not exist.

Under a real model the pages divide cleanly: the executive view, segments, risk
and opportunity, and data quality are aggregate. **Account 360** and **Decision
support** are the two that expose an individual, and the two that would sit
behind a per-account access check.

## Lineage

```
data/external/online_retail_ii.zip        (publisher, SHA-256 pinned)
   → data/raw/*.parquet                   (sources/retail.py — landed, defects intact)
   → conformed_*                          (retail_01_conform.sql — logged)
   → monthly_customer_metrics             (retail_02 — account × month spine)
   → customer_360_train                   (retail_03, window 1–12)
   → customer_360_score                   (retail_03, window 13–24)
   → models/*.joblib                      (analytics/models.py)
   → data/processed/customer_360.parquet  (scored, segmented, recommended)
   → the application
```

Each stage is a separate artefact on disk. Any number on any page can be traced
back to the lines that produced it and rebuilt with `make all`.

## Auditability

- Every transformation is SQL in version control.
- Every threshold, weight and cut-off is a named constant in `config.py` or
  `domains.py`, not a number embedded in a calculation.
- Every dropped line is counted.
- Every download is hash-pinned.
- The application computes nothing, so yesterday's number is reproducible.
- **Negative results are published**: the category classifier and the
  category-level recommender both failed, and both are reported with the
  diagnosis, because the diagnosis is the useful part.
