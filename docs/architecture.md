# Architecture

## The shape of the problem

An organisation holds customer data in systems that were each built for a
different purpose: a core banking ledger, a card switch, a digital channel, a
contact centre. Every one is correct about its own domain and none of them knows
who the customer is. The work is not analysis. The work is assembling a
defensible view of a customer *before* any analysis is possible, and then keeping
the line from that view to a recommended action short enough that someone will
actually act on it.

```
                          SOURCE SYSTEMS
                                │
        ┌───────────────┬───────┴────────┬────────────────┐
        ▼               ▼                ▼                ▼
    customers       accounts        transactions      engagement
    products        balances        loans             interactions
        │               │                │                │
        └───────────────┴───────┬────────┴────────────────┘
                                ▼
                       ╔═════════════════╗
                       ║   RAW LAYER     ║   as landed, defects intact
                       ╚═════════════════╝
                                │
                    22 automated checks across
                    7 quality dimensions
                                ▼
                       ╔═════════════════╗
                       ║ CONFORMED LAYER ║   sql/01_conform.sql
                       ╚═════════════════╝   every dropped row logged
                                │
                                ▼
                       ╔═════════════════╗
                       ║  MONTHLY PANEL  ║   sql/02_monthly_metrics.sql
                       ╚═════════════════╝   customer × month, full spine
                                │
                                ▼
                       ╔═════════════════╗
                       ║  CUSTOMER 360   ║   sql/03_customer_360.sql
                       ╚═════════════════╝   window-parameterised
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        SEGMENTATION       PREDICTION        OPPORTUNITY
        rules + k-means    churn             5 components
                           propensity ×2     published weights
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                       ╔═════════════════╗
                       ║ DECISION ENGINE ║   transparent rules
                       ╚═════════════════╝   ordered by policy
                                │
                                ▼
                       ╔═════════════════╗
                       ║ STREAMLIT APP   ║   reads artefacts, computes nothing
                       ╚═════════════════╝
```

## Layers, and what each is responsible for

| Layer | Built by | Responsible for | Explicitly not responsible for |
|---|---|---|---|
| **Raw** | `data_generation/` | Landing source data unchanged, defects included | Judging or fixing anything |
| **Quality** | `data_quality/` | Measuring conformance; naming what is wrong | Changing the data |
| **Conformed** | `sql/01_conform.sql` | Deciding what to do about each defect, and logging it | Aggregation or business logic |
| **Panel** | `sql/02_monthly_metrics.sql` | One row per customer per month, silent months included | Feature engineering |
| **360** | `sql/03_customer_360.sql` | Customer-level features over a bounded window | Modelling |
| **Models** | `analytics/models.py` | Estimating probabilities, honestly evaluated | Deciding anything |
| **Opportunity** | `analytics/opportunity.py` | A published heuristic for headroom | Claiming to be a revenue figure |
| **Decision** | `decision_engine/` | Turning evidence into a suggested action | Acting |
| **Application** | `app/` | Presenting artefacts | Computing them |

The separation that matters most is the last one. The application reads parquet
files the pipeline produced and trains nothing. That keeps every page
interactive, and it means the number a reader saw yesterday can be reproduced
today, because it is a file rather than the output of a fit that happened to run
while they were looking.

## Why DuckDB, and why the SQL is portable

The transformations are SQL because SQL is where this work belongs: it is the
language the warehouse speaks, it is what a data engineer would be handed and
asked to change, and it keeps the logic out of a notebook that nobody else can
run.

DuckDB is the default engine because it needs no server, so `make all` works on
a laptop with nothing installed. The SQL itself is ordinary analytical SQL —
CTEs, window functions, left joins — and deliberately avoids engine-specific
syntax. The analysis window is a one-row table rather than a DuckDB variable for
exactly this reason. Set `CI_POSTGRES_DSN` and `io.connect()` attaches a
PostgreSQL database instead, running the same statements against the same tables.

## The two-snapshot design

This is the single most important structural decision in the project, and the one
most worth interrogating.

```
month:   1 ─────────────────────────── 12 │ 13 ── 15
         └────── training features ─────┘ └ outcome ┘

month:        4 ──────────────────────────────── 15 │ 16 ── 18
              └────────── scoring features ───────┘ └ future ┘
```

Training features are computed over months 1–12 and the outcome is observed in
13–15. Scoring features are computed over months 4–15 and the outcome is the
unobserved future. Both snapshots come out of **the same SQL file** with a
different `analysis_window`, which is what guarantees the two feature definitions
cannot drift apart — the commonest way a model that validated well goes wrong in
production.

`tests/test_leakage.py` enforces the property structurally: it builds the 360
over two different windows and asserts the results differ in the right direction,
and it greps the SQL for any month literal that reaches into the outcome window.
A future edit that quietly reached forward would fail a test rather than
producing a suspiciously good AUC.

## Repository layout

```
customer-intelligence/
├── app/                     Streamlit application (reads artefacts only)
│   ├── Home.py              Executive view
│   ├── pages/               Customer 360, Segments, Risk & Opportunity,
│   │                        Decision Support, Data Quality & Governance
│   └── components/          Theme, charts, UI primitives, data loaders
├── src/customer_intelligence/
│   ├── config.py            Every tunable constant, in one auditable place
│   ├── io.py                Parquet + DuckDB access; PostgreSQL switch
│   ├── pipeline.py          The four stages, and the CLI
│   ├── data_generation/     Synthetic population, and deliberate defects
│   ├── data_quality/        22 checks, scoring, customer-impact analysis
│   ├── analytics/           Conformance, 360, features, models, segmentation,
│   │                        opportunity scoring
│   └── decision_engine/     The next-best-action rule set
├── sql/                     The transformation layer
├── tests/                   Schema, quality, leakage, features, models,
│                            decision rules, SQL
├── docs/                    This file, data dictionary, governance, model
│                            governance, executive presentation
├── data/                    raw/ and processed/ (generated; not committed)
└── models/                  Fitted estimators and model cards (generated)
```
