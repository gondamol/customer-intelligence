# Architecture

## The shape of the problem

The source is one flat sheet of invoice lines. It knows about transactions; it
knows nothing about customers, products, seasons, or relationships. Everything
this project does is the work of getting from that to a defensible view of an
account, and then keeping the line from that view to a recommended action short
enough that somebody will act on it.

```
        Online Retail II — UCI dataset 502, CC BY 4.0 (Chen 2012)
        1,067,371 invoice lines · Dec 2009 – Dec 2011 · 43 countries
                                │
                downloaded at build time, cached by SHA-256
                                ▼
                       ╔═════════════════╗
                       ║   RAW LAYER     ║   as published, defects intact
                       ╚═════════════════╝
                                │
                    22 checks · 7 dimensions
                                ▼
                       ╔═════════════════╗
                       ║ CONFORMED LAYER ║   retail_01_conform.sql
                       ╚═════════════════╝   every dropped line logged
                     sales │ returns │ charges
                                ▼
                       ╔═════════════════╗
                       ║  MONTHLY PANEL  ║   retail_02_monthly_metrics.sql
                       ╚═════════════════╝   account × month, silence recorded
                                ▼
                       ╔═════════════════╗
                       ║  CUSTOMER 360   ║   retail_03_customer_360.sql
                       ╚═════════════════╝   window-parameterised
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        RFM SEGMENTS      LAPSE / GROWTH     OPPORTUNITY
                          (classifiers)      + RECOMMENDER
              └─────────────────┼─────────────────┘
                                ▼
                       ╔═════════════════╗
                       ║ DECISION ENGINE ║   transparent, ordered rules
                       ╚═════════════════╝
                                ▼
                       ╔═════════════════╗
                       ║ STREAMLIT APP   ║   reads artefacts, computes nothing
                       ╚═════════════════╝
```

## Layers, and what each is responsible for

| Layer | Built by | Responsible for | Explicitly not |
|---|---|---|---|
| **Acquisition** | `sources/` | Fetching from the publisher, caching by hash, carrying the licence | Interpreting anything |
| **Raw** | `sources/retail.py` | Landing the ledger unchanged, defects included | Judging or fixing |
| **Quality** | `data_quality/` | Measuring conformance; naming what is wrong | Changing the data |
| **Conformed** | `retail_01_conform.sql` | Deciding what to do about each defect, and logging it | Aggregation |
| **Panel** | `retail_02_monthly_metrics.sql` | One row per account per month, silent months included | Feature engineering |
| **360** | `retail_03_customer_360.sql` | Account-level features over a bounded window | Modelling |
| **Models** | `analytics/models.py` | Estimating probabilities, honestly evaluated | Deciding |
| **Recommender** | `analytics/recommender.py` | Naming what to offer next | Ranking customers |
| **Opportunity** | `analytics/opportunity.py` | A published heuristic for headroom | Claiming to be revenue |
| **Decision** | `decision_engine/` | Turning evidence into a suggested action | Acting |
| **Application** | `app/` | Presenting artefacts | Computing them |

The last separation matters most. The application reads parquet the pipeline
produced and trains nothing. That keeps every page interactive, and it means a
number seen yesterday can be reproduced today because it is a file rather than
the output of a fit that happened to run while someone was looking.

## The source adapter, and why it is not trivial

`sources/retail.py` is where most of the real work sits. The published sheet has
properties that break a naive load:

| Property | Consequence if ignored |
|---|---|
| `Invoice` and `StockCode` mix integers with alphanumeric codes | Read as numbers, every credit note and administrative row disappears |
| 0.57% of stock codes are not products | Basket counts inflate; every per-product measure is corrupted |
| `M` and `m` are the same manual-adjustment code | The same concept splits into two categories |
| 22,950 negative quantities | Netted off, a £10,000-buy/£9,000-return account looks like a £1,000 one |
| 243,007 lines with no customer | Either 22.6% of revenue is invented onto customers, or it is dropped silently |
| No product categories at all | 5,305 stock codes is not a modellable target space |
| December 2011 is a third of a month | Reads as a collapse in demand rather than the end of a file |

Each is handled explicitly, and each decision is counted in the conformance log
the application shows.

## The derived taxonomy

The dataset ships no categories. Eleven are derived by ordered keyword rules over
the description text, published in `sources/retail.py`, covering **87.7% of
revenue**; the remainder is an explicit `Other giftware` bucket rather than a
silent gap.

Order matters: seasonal and occasion rules are tested before material and form
rules, so a Christmas candle is Christmas stock rather than candle stock —
because that is how it is bought.

This is a **judgement, not a fact about the data**, which is why the rules are in
one readable list and the coverage is reported rather than assumed.

## The two-snapshot design, and why the windows are where they are

```
month:   1 ─────────────────────────── 12 │ 13 ── 15
         └────── training features ─────┘ └ outcome ┘
         Dec 09              Nov 10        Dec 10  Feb 11

month:        13 ─────────────────────────── 24 │ 25 ── 27
              └────── scoring features ───────┘ └ future ┘
              Dec 10                     Nov 11   Dec 11  Feb 12
```

Two properties are enforced:

1. **Training features never touch the outcome window.** Both snapshots come out
   of *the same SQL file* with a different `analysis_window`, which is what stops
   the two feature definitions drifting apart — the commonest way a model that
   validated well goes wrong in production.

2. **Both outcome windows are December–February.** This business takes 84,711
   invoice lines in November 2011 and 27,707 in February. A model trained to
   predict a low-season outcome and applied to predict a peak-season one has
   learned the wrong base rate. `tests/test_retail.py` asserts the two outcome
   windows share calendar months.

`tests/test_leakage.py` and `tests/test_retail.py` also build the 360 over two
different windows and assert the results differ in the right direction, and grep
the SQL for any month literal reaching into the outcome window. A future edit
that quietly reached forward would fail a test rather than producing a
suspiciously good AUC.

## Why DuckDB, and why the SQL is portable

The transformations are SQL because that is the language the warehouse speaks and
what a colleague would be handed and asked to change.

DuckDB is the default because it needs no server, so `make all` works on a laptop
with nothing installed. The SQL is ordinary analytical SQL — CTEs, window
functions, left joins — and deliberately avoids engine-specific syntax. The
analysis window is a one-row table rather than a DuckDB variable for exactly this
reason. Set `CI_POSTGRES_DSN` and `io.connect()` attaches PostgreSQL instead,
running the same statements.

## Repository layout

```
customer-intelligence/
├── app/                     Streamlit application (reads artefacts only)
│   ├── Home.py              Executive view
│   ├── pages/               Account 360, Segments, Risk & Opportunity,
│   │                        Decision Support, Data Quality & Governance
│   └── components/          Theme, charts, UI primitives, data loaders
├── src/customer_intelligence/
│   ├── domains.py           Per-domain windows and outcome definitions
│   ├── config.py            Tunable constants, in one auditable place
│   ├── io.py                Parquet + DuckDB access; PostgreSQL switch
│   ├── pipeline_retail.py   The primary build
│   ├── pipeline.py          The synthetic quality-validation harness
│   ├── sources/             Registry, downloader, retail adapter
│   ├── data_generation/     Synthetic control data with known defects
│   ├── data_quality/        Framework + retail rules + synthetic rules
│   ├── analytics/           Conformance, 360, features, models,
│   │                        segmentation, opportunity, recommender
│   └── decision_engine/     The next-best-action rule sets
├── sql/                     The transformation layer
├── tests/                   123 tests
├── docs/                    This file and the governance documents
├── data/
│   ├── external/            Downloaded source data (never committed)
│   ├── raw/                 Landed source tables (never committed)
│   └── processed/           Derived artefacts (committed — the app reads these)
└── models/                  Model cards (committed); fitted estimators (not)
```
