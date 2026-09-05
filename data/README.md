# Data

Nothing in this directory is committed. The pipeline rebuilds all of it from a
fixed seed in a few minutes, and committing a few hundred megabytes of synthetic
parquet would help nobody.

```bash
make all        # 50,000 customers, the full build
make demo       # 8,000 customers, if you only want to look around
```

## `raw/` — as landed, defects intact

The eight source tables, written exactly as a set of source-system extracts would
arrive, **including sixteen deliberately injected defect types**. This layer is
not cleaned and is not meant to be trusted; it exists so the quality framework
has something real to detect.

| Table | Grain |
|---|---|
| `customers` | one row per customer |
| `accounts` | one row per account |
| `account_monthly_balances` | one row per account per month |
| `transactions` | one row per posting |
| `loans` | one row per facility |
| `products` | one row per holding |
| `digital_activity` | one row per customer per month |
| `service_interactions` | one row per contact |

`defect_manifest.json` records exactly what was injected and how much of it,
which is what allows the checks to be graded against a known answer.

## `processed/` — analysis-ready

| Artefact | What it is |
|---|---|
| `customer_360.parquet` | **The main artefact.** One row per customer: features, scores, segment, opportunity, quadrant, recommended action |
| `customer_360_train.parquet` | The training snapshot (months 1–12) |
| `customer_360_score.parquet` | The scoring snapshot (months 4–15) |
| `monthly_customer_metrics.parquet` | The customer × month panel, silent months included |
| `outcomes.parquet` | The three modelled labels and their eligibility flags |
| `quality_*.parquet` | Check results, dimension scores, customer impact, reconciliation |
| `conformance_log.parquet` | Every row the cleanse removed, and why |
| `segment_profiles.parquet` | Segment characteristics in business terms |
| `model_*_*.parquet` | Per-model comparison, importance, coefficients, calibration |
| `quality_summary.json`, `run_summary.json` | Headline figures the application reads |

## Synthetic data notice

All of it is generated. There are no real customers, no real transactions, and no
real portfolio. The generator produces **no names, addresses, contact details,
national identifiers or dates of birth** — none are needed to demonstrate any of
the analytics, and the cheapest privacy control is data that was never collected.
