"""Shared fixtures.

The tests build a small world from scratch rather than reading the artefacts in
data/. A test that depends on a build someone ran last week is testing the disk,
not the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from customer_intelligence import io                                     # noqa: E402
from customer_intelligence.analytics import (                            # noqa: E402
    build_analytical_layer, build_customer_360, build_outcomes,
)
from customer_intelligence.config import (                               # noqa: E402
    SCORE_FEATURE_MONTHS, TRAIN_FEATURE_MONTHS, TRAIN_OUTCOME_MONTHS,
)
from customer_intelligence.data_generation import generate_all, inject_defects  # noqa: E402

N_TEST_CUSTOMERS = 1200
TEST_SEED = 4242


@pytest.fixture(scope="session")
def clean_tables():
    return generate_all(n_customers=N_TEST_CUSTOMERS, seed=TEST_SEED)


@pytest.fixture(scope="session")
def raw_and_manifest(clean_tables):
    return inject_defects(clean_tables, seed=99)


@pytest.fixture(scope="session")
def raw_tables(raw_and_manifest):
    return raw_and_manifest[0]


@pytest.fixture(scope="session")
def manifest(raw_and_manifest):
    return raw_and_manifest[1]


@pytest.fixture(scope="session")
def con(raw_tables, tmp_path_factory, monkeypatch_session):
    """A DuckDB connection over the raw tables, written to a temp directory."""
    import duckdb

    tmp = tmp_path_factory.mktemp("raw")
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar = false")
    for name, df in raw_tables.items():
        path = tmp / f"{name}.parquet"
        df.to_parquet(path, index=False)
        connection.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{path.as_posix()}')"
        )
    return connection


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def analytical(con):
    build_analytical_layer(con)
    return con


@pytest.fixture(scope="session")
def train_360(analytical):
    return build_customer_360(
        analytical, min(TRAIN_FEATURE_MONTHS), max(TRAIN_FEATURE_MONTHS)
    )


@pytest.fixture(scope="session")
def outcomes(analytical):
    return build_outcomes(analytical, TRAIN_FEATURE_MONTHS, TRAIN_OUTCOME_MONTHS)
