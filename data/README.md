# Data

## `external/` — the published source data (never committed)

Downloaded from the original publisher at build time and cached by SHA-256.
Nothing here is redistributed in this repository; the licence travels with the
data rather than being assumed.

| Dataset | Publisher | Licence |
|---|---|---|
| Online Retail II | UCI ML Repository (502) — Chen, D. (2012) | CC BY 4.0 |
| Telco Customer Churn | IBM sample data | Apache-2.0 (repository) |
| Bank Marketing | UCI ML Repository (222) — Moro, Rita & Cortez (2014) | CC BY 4.0 |

```bash
make fetch      # download and cache
```

`sources.json` records what was retrieved, when, its size and its hash.

## `raw/` — the landed source tables (never committed)

Online Retail II reshaped into `transactions`, `products`, `customers` and
`invoices`, **with every defect intact**. This layer is not cleaned and is not
meant to be trusted; it exists so the quality framework has something real to
detect.

## `processed/` — the derived artefacts (committed)

About 3.4 MB. These are what the deployed application reads, which is why they
are in the repository — under CC BY 4.0 an adaptation may be redistributed with
attribution, and every page of the application carries it.

| Artefact | What it is |
|---|---|
| `customer_360.parquet` | **The main artefact.** One row per account: features, scores, segment, opportunity, quadrant, recommended product and action |
| `customer_360_train/_score.parquet` | The two snapshots (months 1–12 and 13–24) |
| `monthly_customer_metrics.parquet` | The account × month panel, silent months included |
| `outcomes.parquet` | Modelled labels and eligibility flags |
| `next_best_product.parquet` | Top five recommendations per account |
| `quality_*.parquet` | Check results, dimension scores, account impact |
| `conformance_log.parquet` | Every line the cleanse removed, and why |
| `segment_profiles.parquet` | Segment characteristics in business terms |
| `model_*_*.parquet` | Per-model comparison, importance, coefficients, calibration |
| `*.json` | Headline figures, provenance and attribution the app reads |

Rebuild everything from source with `make all`.
