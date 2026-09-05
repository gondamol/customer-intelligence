"""Boilerplate every page repeats: config, theme, CSS, and the src path."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
for path in (str(APP_DIR), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from components.theme import CSS, register_template  # noqa: E402


def setup(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} · Customer Intelligence",
        page_icon="◆", layout="wide", initial_sidebar_state="expanded",
    )
    register_template()
    st.markdown(CSS, unsafe_allow_html=True)
