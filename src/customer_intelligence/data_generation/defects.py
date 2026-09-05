"""Deliberate corruption of the raw layer.

Real landed data is not clean, and a quality framework that has never been shown
a defect proves nothing. So the generator produces a clean world, and this
module damages it in known, counted ways.

Every injection records what it did in a manifest. The data quality report is
then checked against that manifest -- the checks are graded against a known
answer rather than judged by eye. See ``tests/test_data_quality.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..config import DEFECTS


@dataclass
class Defect:
    """One injected problem, and the check that is expected to catch it."""

    table: str
    defect: str
    dimension: str          # completeness | uniqueness | validity | consistency | integrity | timeliness | accuracy
    description: str
    injected_rows: int
    detected_by: str        # the key of the check in the quality report

    def as_dict(self) -> dict:
        return asdict(self)


DEFECT_MANIFEST: list[Defect] = []


def _sample(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    """Indices of a random `rate` share of `n` rows."""
    k = int(round(n * rate))
    if k <= 0:
        return np.array([], dtype=int)
    return rng.choice(n, size=k, replace=False)


def inject_defects(
    tables: dict[str, pd.DataFrame], seed: int = 7717
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Return corrupted copies of the raw tables plus the defect manifest."""
    rng = np.random.default_rng(seed)
    t = {k: v.copy() for k, v in tables.items()}
    manifest: list[Defect] = []

    def record(**kw) -> None:
        manifest.append(Defect(**kw))

    # ---------------------------------------------------------- customers --
    cu = t["customers"]
    n = len(cu)

    idx = _sample(rng, n, DEFECTS.customer_missing_region)
    cu.loc[cu.index[idx], "region"] = None
    record(table="customers", defect="Missing region", dimension="completeness",
           description="Region not captured at onboarding.",
           injected_rows=len(idx), detected_by="customers.region.missing")

    idx = _sample(rng, n, DEFECTS.customer_missing_income_band)
    cu.loc[cu.index[idx], "income_band"] = None
    record(table="customers", defect="Missing income band", dimension="completeness",
           description="Income band left blank; common where KYC refresh is overdue.",
           injected_rows=len(idx), detected_by="customers.income_band.missing")

    idx = _sample(rng, n, DEFECTS.customer_impossible_age)
    cu.loc[cu.index[idx], "age"] = rng.choice([0, 1, 7, 131, 152, -4], len(idx))
    record(table="customers", defect="Impossible age", dimension="validity",
           description="Age outside a plausible adult range (18-100).",
           injected_rows=len(idx), detected_by="customers.age.invalid")

    idx = _sample(rng, n, DEFECTS.customer_missing_id)
    cu.loc[cu.index[idx], "customer_id"] = None
    record(table="customers", defect="Missing customer ID", dimension="completeness",
           description="Primary key absent; row cannot be joined to anything.",
           injected_rows=len(idx), detected_by="customers.customer_id.missing")

    # Duplicates: whole rows re-landed, as a repeated batch load would produce.
    # Drawn only from rows that still have an identifier: a duplicated row with
    # a null key is not a uniqueness defect, and the rule correctly ignores it.
    # Sampling without this restriction produces an off-by-one against the
    # manifest that only appears at scale.
    keyed = cu.index[cu["customer_id"].notna()]
    k = int(round(n * DEFECTS.customer_duplicates))
    dup_idx = rng.choice(keyed, size=min(k, len(keyed)), replace=False)
    dups = cu.loc[dup_idx].copy()
    t["customers"] = pd.concat([cu, dups], ignore_index=True)
    record(table="customers", defect="Duplicate customer records", dimension="uniqueness",
           description="Same customer_id landed more than once (re-run of a batch load).",
           injected_rows=len(dups), detected_by="customers.customer_id.duplicate")

    # ----------------------------------------------------------- accounts --
    ac = t["accounts"]
    n = len(ac)

    idx = _sample(rng, n, DEFECTS.account_negative_balance)
    savings = ac.index[(ac["account_type"].isin(["Savings", "Fixed deposit"]))]
    idx = rng.choice(savings, size=min(len(idx), len(savings)), replace=False)
    ac.loc[idx, "current_balance"] = -np.abs(ac.loc[idx, "current_balance"])
    record(table="accounts", defect="Negative balance on a savings product", dimension="validity",
           description="Savings and fixed deposit balances cannot be negative.",
           injected_rows=len(idx), detected_by="accounts.balance.invalid")

    idx = _sample(rng, n, DEFECTS.account_invalid_status)
    ac.loc[ac.index[idx], "status"] = rng.choice(["ACTIVE", "active", "A", "Frozen?", ""], len(idx))
    record(table="accounts", defect="Invalid account status", dimension="consistency",
           description="Status outside the controlled vocabulary (Active/Dormant/Closed).",
           injected_rows=len(idx), detected_by="accounts.status.invalid")

    idx = _sample(rng, n, DEFECTS.account_invalid_open_date)
    ac.loc[ac.index[idx], "opening_date"] = pd.NaT
    record(table="accounts", defect="Missing opening date", dimension="completeness",
           description="Account opening date absent.",
           injected_rows=len(idx), detected_by="accounts.opening_date.missing")

    # Orphans: accounts pointing at customers that do not exist.
    k = int(round(n * DEFECTS.account_orphaned))
    orphan_idx = _sample(rng, n, DEFECTS.account_orphaned)
    ac.loc[ac.index[orphan_idx], "customer_id"] = [
        f"C9{i:05d}" for i in range(90_000, 90_000 + len(orphan_idx))
    ]
    record(table="accounts", defect="Orphaned account records", dimension="integrity",
           description="customer_id has no matching row in customers.",
           injected_rows=len(orphan_idx), detected_by="accounts.customer_id.orphan")
    t["accounts"] = ac

    # ------------------------------------------------------- transactions --
    tx = t["transactions"]
    n = len(tx)

    idx = _sample(rng, n, DEFECTS.transaction_missing_amount)
    tx.loc[tx.index[idx], "amount"] = np.nan
    record(table="transactions", defect="Missing transaction amount", dimension="completeness",
           description="Amount absent; the row cannot contribute to any value measure.",
           injected_rows=len(idx), detected_by="transactions.amount.missing")

    idx = _sample(rng, n, DEFECTS.transaction_extreme_amount)
    tx.loc[tx.index[idx], "amount"] = rng.uniform(4e7, 9e8, len(idx)).round(2)
    record(table="transactions", defect="Anomalous transaction amount", dimension="accuracy",
           description="Amounts orders of magnitude beyond the population distribution.",
           injected_rows=len(idx), detected_by="transactions.amount.anomaly")

    idx = _sample(rng, n, DEFECTS.transaction_future_date)
    tx.loc[tx.index[idx], "transaction_date"] = (
        tx["transaction_date"].max() + pd.to_timedelta(rng.integers(40, 900, len(idx)), unit="D")
    )
    record(table="transactions", defect="Transaction dated in the future", dimension="timeliness",
           description="transaction_date after the close of the reporting window.",
           injected_rows=len(idx), detected_by="transactions.date.future")

    dup_idx = _sample(rng, n, DEFECTS.transaction_duplicates)
    dups = tx.iloc[dup_idx].copy()
    t["transactions"] = pd.concat([tx, dups], ignore_index=True)
    record(table="transactions", defect="Duplicate transactions", dimension="uniqueness",
           description="Identical transaction_id posted twice (retry without idempotency).",
           injected_rows=len(dups), detected_by="transactions.transaction_id.duplicate")

    # ------------------------------------------------------------ products --
    pr = t["products"]
    n = len(pr)
    idx = _sample(rng, n, DEFECTS.product_inconsistent_code)
    variants = {
        "Current account": "CURRENT_ACCT", "Savings": "savings",
        "Investment": "INVESTMENT ", "Personal loan": "PERS-LOAN",
        "Asset finance": "asset_finance", "Card": "CARD",
        "Insurance": "INS", "Digital wallet": "wallet",
    }
    pr.loc[pr.index[idx], "product_type"] = pr.loc[pr.index[idx], "product_type"].map(variants)
    record(table="products", defect="Inconsistent product codes", dimension="consistency",
           description="Same product expressed in several source-system spellings.",
           injected_rows=len(idx), detected_by="products.product_type.inconsistent")
    t["products"] = pr

    # --------------------------------------------------------------- loans --
    ln = t["loans"]
    # The rule only requires a rate on facilities that are still running, so the
    # defect is injected on the same population the rule governs. Otherwise the
    # reconciliation reports a miss for a check that is behaving correctly.
    active_loans = ln.index[ln["repayment_status"] != "Closed"]
    k = int(round(len(ln) * DEFECTS.loan_missing_rate))
    idx = rng.choice(active_loans, size=min(k, len(active_loans)), replace=False)
    ln.loc[idx, "interest_rate"] = np.nan
    record(table="loans", defect="Missing interest rate", dimension="completeness",
           description="Interest rate absent on an active facility.",
           injected_rows=len(idx), detected_by="loans.interest_rate.missing")
    t["loans"] = ln

    # ------------------------------------------------------- interactions --
    si = t["service_interactions"]
    idx = _sample(rng, len(si), DEFECTS.interaction_missing_satisfaction)
    si.loc[si.index[idx], "satisfaction_score"] = np.nan
    record(table="service_interactions", defect="Missing satisfaction score",
           dimension="completeness",
           description="Post-contact survey not returned; missing not at random.",
           injected_rows=len(idx), detected_by="service_interactions.satisfaction.missing")
    t["service_interactions"] = si

    DEFECT_MANIFEST.clear()
    DEFECT_MANIFEST.extend(manifest)
    return t, [d.as_dict() for d in manifest]
