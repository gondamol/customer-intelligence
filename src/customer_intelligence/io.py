"""Storage and query access.

Parquet on disk is the store; DuckDB is the query engine over it. There is no
server to install, which is the point -- but the SQL in ``sql/`` is ordinary
analytical SQL, and ``connect()`` accepts a PostgreSQL DSN for the same tables
when one is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, SQL_DIR

RAW_TABLES = (
    "customers", "accounts", "account_monthly_balances", "transactions",
    "loans", "products", "digital_activity", "service_interactions",
)

PROCESSED_TABLES = (
    "customer_360", "monthly_customer_metrics", "segment_profiles",
    "quality_results", "quality_impact", "conformance_log",
)


def _path(layer: str, table: str) -> Path:
    base = RAW_DIR if layer == "raw" else PROCESSED_DIR
    return base / f"{table}.parquet"


def write_table(df: pd.DataFrame, table: str, layer: str = "raw") -> Path:
    path = _path(layer, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_table(table: str, layer: str = "raw") -> pd.DataFrame:
    path = _path(layer, table)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make generate-data` (or "
            f"`python -m customer_intelligence.pipeline generate`) first."
        )
    return pd.read_parquet(path)


def table_exists(table: str, layer: str = "raw") -> bool:
    return _path(layer, table).exists()


def connect(layer: str = "raw", tables: tuple[str, ...] | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with the layer's parquet files as views.

    Set ``CI_POSTGRES_DSN`` to attach a PostgreSQL database instead; the views
    are then created over the ``ci_pg`` attachment so the same SQL runs against
    either engine.
    """
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    dsn = os.environ.get("CI_POSTGRES_DSN")
    names = tables or (RAW_TABLES if layer == "raw" else PROCESSED_TABLES)

    if dsn:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{dsn}' AS ci_pg (TYPE POSTGRES, READ_ONLY);")
        schema = os.environ.get("CI_POSTGRES_SCHEMA", "public")
        for table in names:
            con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM ci_pg.{schema}.{table}")
        return con

    for table in names:
        path = _path(layer, table)
        if path.exists():
            con.execute(
                f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')"
            )
    return con


def run_sql_file(con: duckdb.DuckDBPyConnection, filename: str) -> None:
    """Execute every statement in a file from ``sql/``."""
    text = (SQL_DIR / filename).read_text()
    for statement in _split_statements(text):
        con.execute(statement)


def query_sql_file(con: duckdb.DuckDBPyConnection, filename: str) -> pd.DataFrame:
    """Execute a file and return the final statement's result."""
    statements = _split_statements((SQL_DIR / filename).read_text())
    for statement in statements[:-1]:
        con.execute(statement)
    return con.execute(statements[-1]).df()


def _split_statements(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
