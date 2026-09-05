"""Schema and referential integrity of the generated source tables."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_intelligence.io import RAW_TABLES

EXPECTED_COLUMNS = {
    "customers": {"customer_id", "age", "gender", "region", "employment_type",
                  "income_band", "monthly_income", "tenure_months",
                  "customer_segment", "join_date"},
    "accounts": {"account_id", "customer_id", "account_type", "opening_date",
                 "status", "average_balance", "current_balance"},
    "transactions": {"transaction_id", "customer_id", "account_id",
                     "transaction_date", "transaction_type", "amount",
                     "channel", "merchant_category"},
    "loans": {"loan_id", "customer_id", "loan_type", "principal",
              "outstanding_balance", "interest_rate", "repayment_status",
              "origination_date"},
    "products": {"customer_id", "product_type", "start_date", "status"},
    "digital_activity": {"customer_id", "month", "mobile_logins", "online_logins",
                         "digital_transactions", "failed_logins"},
    "service_interactions": {"interaction_id", "customer_id", "interaction_date",
                             "channel", "interaction_type", "resolution_status",
                             "satisfaction_score"},
    "account_monthly_balances": {"account_id", "customer_id", "month",
                                 "closing_balance"},
}


def test_all_expected_tables_present(clean_tables):
    for table in RAW_TABLES:
        assert table in clean_tables, f"{table} missing from the generated set"


@pytest.mark.parametrize("table", sorted(EXPECTED_COLUMNS))
def test_required_columns_present(clean_tables, table):
    missing = EXPECTED_COLUMNS[table] - set(clean_tables[table].columns)
    assert not missing, f"{table} is missing columns: {sorted(missing)}"


@pytest.mark.parametrize("table", sorted(EXPECTED_COLUMNS))
def test_no_empty_tables(clean_tables, table):
    assert len(clean_tables[table]) > 0


def test_primary_keys_unique_before_injection(clean_tables):
    for table, key in [("customers", "customer_id"), ("accounts", "account_id"),
                       ("transactions", "transaction_id"), ("loans", "loan_id"),
                       ("service_interactions", "interaction_id")]:
        col = clean_tables[table][key]
        assert col.is_unique, f"{table}.{key} is not unique in the clean data"
        assert col.notna().all(), f"{table}.{key} contains nulls in the clean data"


@pytest.mark.parametrize("table", ["accounts", "transactions", "loans", "products",
                                   "digital_activity", "service_interactions",
                                   "account_monthly_balances"])
def test_referential_integrity_before_injection(clean_tables, table):
    """Every child row points at a real customer before defects are injected."""
    known = set(clean_tables["customers"]["customer_id"])
    orphans = set(clean_tables[table]["customer_id"]) - known
    assert not orphans, f"{table} has {len(orphans)} orphaned customer_ids"


def test_transactions_belong_to_the_customers_own_accounts(clean_tables):
    """A transaction on someone else's account would be a silent, serious bug."""
    accounts = clean_tables["accounts"][["account_id", "customer_id"]]
    tx = clean_tables["transactions"][["account_id", "customer_id"]].drop_duplicates()
    merged = tx.merge(accounts, on="account_id", how="left", suffixes=("_tx", "_acct"))
    mismatched = merged[merged["customer_id_tx"] != merged["customer_id_acct"]]
    assert mismatched.empty, f"{len(mismatched)} transactions posted to another customer's account"


def test_no_personal_identifiers_generated(clean_tables):
    """Data minimisation is a property of the generator, not a promise in a doc."""
    forbidden = {"name", "first_name", "last_name", "email", "phone", "address",
                 "national_id", "passport", "date_of_birth", "dob"}
    for table, df in clean_tables.items():
        overlap = forbidden & {c.lower() for c in df.columns}
        assert not overlap, f"{table} carries identifying columns: {sorted(overlap)}"


def test_ages_and_tenure_are_internally_consistent(clean_tables):
    """Nobody can have banked for longer than they have been an adult."""
    cu = clean_tables["customers"]
    assert (cu["tenure_months"] <= (cu["age"] - 17) * 12).all()
    assert cu["age"].between(18, 90).all()
