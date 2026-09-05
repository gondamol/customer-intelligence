"""Loading the artefacts the pipeline produced.

The application reads; it never computes. Everything expensive -- generation,
conformance, training, scoring -- happened in the pipeline, and its outputs are
on disk. That is what keeps every page interactive, and it is also the honest
separation: a dashboard that fits a model on page load has no reproducible
answer to "what was this number when I looked at it yesterday?".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from customer_intelligence.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402

MISSING_DATA_MESSAGE = """
### The analytical artefacts have not been built yet

This application reads what the pipeline produced. Build it once:

```bash
make setup          # create the environment
make all            # generate, assess quality, model, score
make run            # open this application
```

The full build takes a few minutes on 50,000 customers.
"""


def _require(path: Path) -> None:
    if not path.exists():
        st.markdown(MISSING_DATA_MESSAGE)
        st.stop()


@st.cache_data(show_spinner=False)
def load(table: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{table}.parquet"
    _require(path)
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_json(name: str, where: str = "processed") -> dict:
    path = (PROCESSED_DIR if where == "processed" else MODELS_DIR) / name
    _require(path)
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def customers() -> pd.DataFrame:
    """The scored book: one row per customer, every score and recommendation."""
    return load("customer_360")


@st.cache_data(show_spinner=False)
def monthly() -> pd.DataFrame:
    return load("monthly_customer_metrics")


@st.cache_data(show_spinner=False)
def customer_history(customer_id: str) -> pd.DataFrame:
    m = monthly()
    return m[m["customer_id"] == customer_id].sort_values("month_index")


@st.cache_data(show_spinner=False)
def model_cards() -> dict:
    return load_json("model_cards.json", where="models")


@st.cache_data(show_spinner=False)
def quality_summary() -> dict:
    return load_json("quality_summary.json")


@st.cache_data(show_spinner=False)
def run_summary() -> dict:
    return load_json("run_summary.json")
