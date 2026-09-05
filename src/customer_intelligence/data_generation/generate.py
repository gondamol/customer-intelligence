"""Synthetic population generator.

The design principle is that nothing is drawn independently. Each customer gets
a small vector of latent traits -- affluence, digital affinity, credit appetite,
relationship stability -- and every observable quantity is a noisy function of
those traits. That is what makes segmentation find real structure, makes churn
predictable but not trivially so, and makes propensity models learn something
other than noise.

The latent traits are never written to disk. Models see only what an analyst
would see: balances, transactions, logins, complaints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    ACCOUNT_TYPES, CHANNELS, EMPLOYMENT_TYPES, INCOME_BANDS,
    INTERACTION_CHANNELS, INTERACTION_TYPES, LOAN_TYPES, MERCHANT_CATEGORIES,
    N_CUSTOMERS, N_MONTHS, PANEL_START, PRODUCT_TYPES, RANDOM_SEED, REGIONS,
    RESOLUTION_STATUSES,
)

MONTH_STARTS = pd.date_range(PANEL_START, periods=N_MONTHS, freq="MS")


# ------------------------------------------------------------- helpers ----


def _softmax_choice(rng: np.random.Generator, scores: np.ndarray) -> np.ndarray:
    """Pick one column per row, with probability proportional to exp(score)."""
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = weights / weights.sum(axis=1, keepdims=True)
    cumulative = probs.cumsum(axis=1)
    draws = rng.random((scores.shape[0], 1))
    return (draws > cumulative).sum(axis=1)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------- latents ----


def _latent_traits(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Four correlated traits that drive everything downstream."""
    # A shared "financial capacity" factor plus trait-specific variation.
    capacity = rng.normal(0, 1, n)

    age = np.clip(rng.gamma(shape=6.5, scale=5.2, size=n) + 18, 18, 84)

    # Affluence rises with age (to a point) and with the shared factor.
    age_effect = -0.00095 * (age - 46) ** 2 + 0.35
    affluence = 0.75 * capacity + 0.5 * age_effect + rng.normal(0, 0.55, n)

    # Digital affinity falls with age and rises mildly with affluence.
    digital = -0.045 * (age - 38) + 0.30 * affluence + rng.normal(0, 1.0, n)
    digital = digital / 2.2

    # Credit appetite is highest in mid-life and largely independent of wealth.
    credit_appetite = (
        0.9 * np.exp(-0.5 * ((age - 37) / 12.0) ** 2)
        - 0.25 * affluence
        + rng.normal(0, 0.6, n)
    )

    # Relationship stability: the latent driver of attrition.
    stability = 0.30 * affluence + 0.20 * digital + rng.normal(0, 0.95, n)

    return {
        "age": age,
        "affluence": affluence,
        "digital": digital,
        "credit_appetite": credit_appetite,
        "stability": stability,
    }


def _assign_segment(lat: dict[str, np.ndarray]) -> np.ndarray:
    """Rule-based ground-truth segment.

    Deliberately overlapping: an unsupervised method should recover most of
    this structure but not all of it, which is the honest situation.
    """
    aff, dig, cred = lat["affluence"], lat["digital"], lat["credit_appetite"]
    n = len(aff)
    seg = np.full(n, "Mass Market", dtype=object)

    # Wealth decides the top and bottom of the book; behaviour decides the middle.
    mid = (aff >= -0.75) & (aff < 0.85)
    seg[aff < -0.75] = "Emerging"
    seg[mid & (cred > 0.90)] = "Credit Dependent"
    seg[mid & (cred <= 0.90) & (dig > 0.25)] = "Digital First"
    seg[mid & (cred < -0.45) & (dig <= 0.25)] = "Savings Focused"
    seg[(aff >= 0.85) & (aff < 1.80)] = "Affluent"
    seg[aff >= 1.80] = "High Value"
    return seg


# ----------------------------------------------------------- customers ----


def _customers(rng: np.random.Generator, lat: dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(lat["age"])
    age = lat["age"]
    aff = lat["affluence"]

    income_rank = aff + rng.normal(0, 0.30, n)
    income_band_idx = np.searchsorted(np.quantile(income_rank, [0.30, 0.58, 0.80, 0.94]), income_rank)
    income_band = np.array(INCOME_BANDS, dtype=object)[income_band_idx]

    monthly_income = np.round(
        np.exp(9.55 + 0.60 * aff + rng.normal(0, 0.28, n)), -2
    )

    # Employment mix shifts with age and affluence rather than being uniform.
    emp_scores = np.column_stack([
        0.9 * aff + 0.5,                                  # Salaried
        0.3 * aff + 0.2 * (age > 30),                     # Self-employed
        1.1 * aff - 0.6,                                  # Business owner
        -1.0 * aff + 0.4,                                 # Informal
        -3.0 + 0.16 * (age - 55),                         # Retired
    ]) + rng.normal(0, 0.7, (n, 5))
    employment = np.array(EMPLOYMENT_TYPES, dtype=object)[_softmax_choice(rng, emp_scores)]

    # Tenure is bounded by age: nobody banks before 18.
    max_tenure = np.clip((age - 18) * 12, 6, 300)
    tenure = np.minimum(rng.gamma(2.1, 34, n) + 4, max_tenure).astype(int)

    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "age": np.round(age).astype(int),
        "gender": rng.choice(["Female", "Male"], n, p=[0.49, 0.51]),
        "region": rng.choice(REGIONS, n, p=[0.19, 0.13, 0.14, 0.11, 0.17, 0.09, 0.17]),
        "employment_type": employment,
        "income_band": income_band,
        "monthly_income": monthly_income,
        "tenure_months": tenure,
        "customer_segment": _assign_segment(lat),
        "join_date": (MONTH_STARTS[-1] - pd.to_timedelta(tenure * 30, unit="D")).normalize(),
    })


# --------------------------------------------------------- monthly panel ----


def _monthly_panel(
    rng: np.random.Generator, customers: pd.DataFrame, lat: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, np.ndarray]:
    """The engine of the whole dataset: a customer x month activity panel.

    Disengagement is modelled explicitly. A share of customers begin an
    irreversible decline at some month, and their balances, transactions and
    logins all decay together while their complaint rate rises. That co-movement
    is the signal the churn model is meant to find.
    """
    n, m = len(customers), N_MONTHS
    aff, dig, stab = lat["affluence"], lat["digital"], lat["stability"]
    cred = lat["credit_appetite"]

    # Who declines, and from when.
    p_decline = _logistic(-0.60 - 0.95 * stab)
    declining = rng.random(n) < p_decline
    # Declines begin late in the panel, or after it. A relationship that has
    # already finished unwinding by the end of the feature window is trivially
    # predictable and tells you nothing about a model; the cases that matter are
    # the ones only part-way through when the scoring snapshot is taken.
    decline_start = rng.integers(5, m + 2, n)
    decline_rate = rng.uniform(0.30, 0.75, n)

    months = np.arange(1, m + 1)
    months_into = np.maximum(0, months[None, :] - decline_start[:, None])
    decay = np.where(
        declining[:, None],
        np.exp(-decline_rate[:, None] * months_into),
        1.0,
    )

    # Not every decline ends in attrition. Around a quarter of declining
    # customers recover -- a seasonal dip, a period abroad, a temporary change
    # in circumstances. Without this the decline signal is deterministic and any
    # model trained on it scores implausibly well.
    recovering = declining & (rng.random(n) < 0.32)
    recovery_start = decline_start + rng.integers(2, 6, n)
    ramp = np.clip(0.25 * np.maximum(0, months[None, :] - recovery_start[:, None]), 0, 1)
    decay = np.where(recovering[:, None], decay + (1.0 - decay) * ramp, decay)

    # And some relationships end without warning -- a move, a bereavement, a
    # competitor offer taken on the spot. These begin inside the outcome window,
    # so nothing in the feature window predicts them. This is the irreducible
    # error that keeps the achievable AUC honest.
    abrupt = (~declining) & (rng.random(n) < 0.105)
    abrupt_start = rng.integers(13, m + 1, n)
    months_abrupt = np.maximum(0, months[None, :] - abrupt_start[:, None] + 1)
    decay = np.where(abrupt[:, None], decay * np.exp(-1.4 * months_abrupt), decay)

    # Everyone also drifts a little, up or down, for ordinary reasons.
    drift = np.exp(np.cumsum(rng.normal(0.002, 0.045, (n, m)), axis=1))
    activity = decay * drift

    # --- balances -----------------------------------------------------------
    # Credit-hungry customers hold less against the same income: the balance
    # they run is the visible trace of an appetite that is otherwise latent.
    base_balance = np.exp(10.15 + 1.02 * aff - 0.40 * cred + rng.normal(0, 0.42, n))
    balance = base_balance[:, None] * activity * np.exp(rng.normal(0, 0.11, (n, m)))
    balance = np.maximum(balance, 25.0)

    # --- transactions -------------------------------------------------------
    base_txn = np.exp(1.55 + 0.30 * aff + 0.22 * dig + rng.normal(0, 0.32, n))
    txn_lambda = np.maximum(base_txn[:, None] * activity, 0.05)
    txn_count = rng.poisson(txn_lambda)

    txn_value = txn_count * np.exp(7.35 + 0.80 * aff[:, None] + rng.normal(0, 0.30, (n, m)))
    txn_value = np.where(txn_count > 0, txn_value, 0.0)

    # --- digital ------------------------------------------------------------
    mobile_lambda = np.maximum(np.exp(1.75 + 0.85 * dig)[:, None] * activity, 0.02)
    mobile_logins = rng.poisson(mobile_lambda)
    online_lambda = np.maximum(np.exp(0.55 + 0.45 * dig + 0.25 * aff)[:, None] * activity, 0.01)
    online_logins = rng.poisson(online_lambda)
    digital_txn = rng.binomial(txn_count, _clip01(_logistic(1.35 * dig)[:, None] * np.ones((1, m))))
    failed_logins = rng.poisson(np.maximum(0.16 * (mobile_logins + online_logins) ** 0.55, 0.02))

    long = pd.DataFrame({
        "customer_id": np.repeat(customers["customer_id"].to_numpy(), m),
        "month_index": np.tile(months, n),
        "month": np.tile(MONTH_STARTS.to_numpy(), n),
        "closing_balance": np.round(balance.ravel(), 2),
        "transaction_count": txn_count.ravel(),
        "transaction_value": np.round(txn_value.ravel(), 2),
        "mobile_logins": mobile_logins.ravel(),
        "online_logins": online_logins.ravel(),
        "digital_transactions": digital_txn.ravel(),
        "failed_logins": failed_logins.ravel(),
    })
    # `activity` is returned alongside the panel rather than attached to it:
    # service interactions need the customer x month health matrix, and dataframe
    # attrs do not survive a round trip through parquet.
    return long, activity


# ------------------------------------------------------------- accounts ----


def _accounts(
    rng: np.random.Generator, customers: pd.DataFrame, lat: dict[str, np.ndarray],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    n = len(customers)
    aff, cred = lat["affluence"], lat["credit_appetite"]

    # Number of accounts rises with affluence.
    n_accounts = np.clip(rng.poisson(np.exp(0.10 + 0.42 * aff)) + 1, 1, 5)
    owner_idx = np.repeat(np.arange(n), n_accounts)
    total = len(owner_idx)

    # First account is always Current; later ones skew Savings/Fixed deposit.
    position = np.concatenate([np.arange(k) for k in n_accounts])
    type_scores = np.column_stack([
        np.where(position == 0, 6.0, -1.5),                       # Current
        np.where(position == 0, -6.0, 1.4 - 0.5 * cred[owner_idx]),  # Savings
        np.where(position == 0, -6.0, -0.6 + 0.9 * aff[owner_idx]),  # Fixed deposit
        np.where(position == 0, -6.0, 0.2 + 0.7 * lat["digital"][owner_idx]),  # Wallet
    ]) + rng.normal(0, 0.5, (total, 4))
    acc_type = np.array(ACCOUNT_TYPES, dtype=object)[_softmax_choice(rng, type_scores)]

    # Balances are the customer's monthly balance split across their accounts.
    last = panel[panel["month_index"] == N_MONTHS].set_index("customer_id")["closing_balance"]
    mean_bal = panel.groupby("customer_id", observed=True)["closing_balance"].mean()
    share = rng.dirichlet(np.full(5, 2.0), n)
    share_flat = np.concatenate([share[i, :k] / share[i, :k].sum() for i, k in enumerate(n_accounts)])

    cust_ids = customers["customer_id"].to_numpy()[owner_idx]
    current_balance = np.round(last.loc[cust_ids].to_numpy() * share_flat, 2)
    average_balance = np.round(mean_bal.loc[cust_ids].to_numpy() * share_flat, 2)

    tenure = customers["tenure_months"].to_numpy()[owner_idx]
    age_days = rng.integers(1, np.maximum(tenure * 30, 2)) 
    opening_date = MONTH_STARTS[-1].normalize() - pd.to_timedelta(age_days, unit="D")

    status = np.where(rng.random(total) < 0.035, "Dormant", "Active")
    status = np.where(rng.random(total) < 0.012, "Closed", status)

    return pd.DataFrame({
        "account_id": [f"A{i:07d}" for i in range(1, total + 1)],
        "customer_id": cust_ids,
        "account_type": acc_type,
        "opening_date": opening_date,
        "status": status,
        "average_balance": average_balance,
        "current_balance": current_balance,
    })


# ----------------------------------------------- account monthly balances ----


def _account_monthly_balances(accounts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Month-end balance per account: the standard source extract.

    The customer's monthly balance path comes from the panel; it is split across
    that customer's accounts in proportion to their average balances, so an
    account's share of the relationship stays stable while the total moves.
    """
    share = accounts[["account_id", "customer_id", "average_balance"]].copy()
    totals = share.groupby("customer_id", observed=True)["average_balance"].transform("sum")
    share["share"] = np.where(totals > 0, share["average_balance"] / totals, 0.0)

    merged = panel[["customer_id", "month_index", "month", "closing_balance"]].merge(
        share[["account_id", "customer_id", "share"]], on="customer_id", how="inner"
    )
    merged["closing_balance"] = (merged["closing_balance"] * merged["share"]).round(2)
    return merged[["account_id", "customer_id", "month", "month_index", "closing_balance"]]


# --------------------------------------------------------- transactions ----


def _transactions(
    rng: np.random.Generator, panel: pd.DataFrame, accounts: pd.DataFrame,
    credit_appetite: pd.Series,
) -> pd.DataFrame:
    active = panel[panel["transaction_count"] > 0]
    counts = active["transaction_count"].to_numpy()
    total = int(counts.sum())

    cust = np.repeat(active["customer_id"].to_numpy(), counts)
    month = np.repeat(active["month"].to_numpy(), counts)
    mean_amt = np.repeat(
        (active["transaction_value"] / active["transaction_count"]).to_numpy(), counts
    )

    # One account per transaction, drawn from that customer's own accounts.
    # Done by flat-index arithmetic rather than a per-row lookup, because this
    # runs over several million rows.
    order = accounts.sort_values(["customer_id", "account_id"], ignore_index=True)
    flat_ids = order["account_id"].to_numpy()
    offsets = order.groupby("customer_id", observed=True).cumcount().to_numpy()
    start_index = pd.Series(
        np.flatnonzero(offsets == 0), index=order.loc[offsets == 0, "customer_id"].to_numpy()
    )
    n_acc = order.groupby("customer_id", observed=True).size()

    pick = (rng.random(total) * n_acc.reindex(cust).to_numpy()).astype(int)
    account_id = flat_ids[start_index.reindex(cust).to_numpy() + pick]

    day = rng.integers(0, 28, total)
    txn_date = pd.to_datetime(month) + pd.to_timedelta(day, unit="D")

    amount = np.round(mean_amt * rng.lognormal(-0.11, 0.47, total), 2)
    amount = np.maximum(amount, 1.0)

    # Share of postings that are credits (money in). Customers with a high
    # credit appetite run more out than in, so their inflow/outflow ratio is a
    # visible proxy for an appetite that is never directly observed.
    credit_p = _logistic(-0.94 - 0.45 * credit_appetite.reindex(cust).to_numpy())
    txn_type = np.where(rng.random(total) < credit_p, "Credit", "Debit")

    channel = rng.choice(CHANNELS, total, p=[0.34, 0.09, 0.07, 0.15, 0.20, 0.15])
    merchant = rng.choice(
        MERCHANT_CATEGORIES, total,
        p=[0.17, 0.09, 0.09, 0.08, 0.11, 0.07, 0.03, 0.04, 0.03, 0.15, 0.10, 0.04],
    )

    return pd.DataFrame({
        "transaction_id": [f"T{i:09d}" for i in range(1, total + 1)],
        "customer_id": cust,
        "account_id": account_id,
        "transaction_date": txn_date,
        "transaction_type": txn_type,
        "amount": amount,
        "channel": channel,
        "merchant_category": merchant,
    })


# ---------------------------------------------------------------- loans ----


def _loans(
    rng: np.random.Generator, customers: pd.DataFrame, lat: dict[str, np.ndarray],
) -> pd.DataFrame:
    n = len(customers)
    aff, cred, stab = lat["affluence"], lat["credit_appetite"], lat["stability"]

    has_loan = rng.random(n) < _logistic(-1.35 + 1.25 * cred + 0.20 * aff)
    idx = np.flatnonzero(has_loan)
    k = len(idx)

    loan_type = np.array(LOAN_TYPES, dtype=object)[_softmax_choice(
        rng,
        np.column_stack([
            1.0 + 0.3 * cred[idx],
            0.2 + 0.9 * aff[idx],
            -0.4 + 0.5 * aff[idx],
            0.8 - 1.1 * aff[idx],
        ]) + rng.normal(0, 0.6, (k, 4)),
    )]

    principal = np.round(np.exp(11.0 + 0.85 * aff[idx] + rng.normal(0, 0.45, k)), -2)
    term = rng.choice([12, 24, 36, 48, 60], k, p=[0.18, 0.30, 0.28, 0.14, 0.10])
    # Age of the facility as a share of its term, so most loans are still running.
    months_elapsed = np.clip((term * rng.beta(1.6, 2.4, k)).astype(int) + 1, 1, term)
    outstanding = np.round(principal * np.clip(1 - months_elapsed / term, 0, 1) * rng.uniform(0.85, 1.10, k), 2)
    rate = np.round(np.clip(0.235 - 0.022 * aff[idx] + rng.normal(0, 0.018, k), 0.07, 0.36), 4)

    # Repayment quality tracks relationship stability, not wealth. Thresholds are
    # set to give a book that is mostly performing, with a delinquency tail:
    # roughly 78% current, 10% early arrears, 6% late, 6% default.
    repay_score = 1.30 * stab[idx] + 0.25 * aff[idx] + rng.normal(0, 0.85, k)
    status = np.where(repay_score > -1.15, "Current",
             np.where(repay_score > -1.95, "Late 1-30",
             np.where(repay_score > -2.45, "Late 31-90", "Default")))

    # A minority of facilities have run to term and been settled.
    settled = (rng.random(k) < 0.08) & (repay_score > -1.15)
    status = np.where(settled, "Closed", status)
    outstanding = np.where(settled, 0.0, outstanding)

    origination = MONTH_STARTS[-1].normalize() - pd.to_timedelta(months_elapsed * 30, unit="D")

    return pd.DataFrame({
        "loan_id": [f"L{i:07d}" for i in range(1, k + 1)],
        "customer_id": customers["customer_id"].to_numpy()[idx],
        "loan_type": loan_type,
        "principal": principal,
        "outstanding_balance": np.maximum(outstanding, 0.0),
        "interest_rate": rate,
        "term_months": term,
        "repayment_status": status,
        "origination_date": origination,
    })


# ------------------------------------------------------------- products ----


def _products(
    rng: np.random.Generator, customers: pd.DataFrame, lat: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Product holdings, with an explicit acquisition window.

    A product taken in months 13-15 is a *future* event relative to the training
    feature window -- that is exactly what the propensity models predict.
    """
    n = len(customers)
    aff, dig, cred, stab = lat["affluence"], lat["digital"], lat["credit_appetite"], lat["stability"]
    cust_ids = customers["customer_id"].to_numpy()

    # Utility of each product type per customer.
    utility = {
        "Current account": np.full(n, 6.0),
        "Savings": 0.9 + 0.55 * aff - 0.40 * cred,
        "Investment": -2.05 + 1.35 * aff + 0.35 * stab,
        "Personal loan": -1.60 + 1.20 * cred,
        "Asset finance": -2.60 + 0.95 * aff + 0.60 * cred,
        "Card": -0.85 + 0.75 * aff + 0.45 * dig,
        "Insurance": -1.70 + 0.80 * aff + 0.25 * (customers["age"].to_numpy() > 35),
        "Digital wallet": -0.60 + 1.15 * dig,
    }

    rows = []
    for ptype in PRODUCT_TYPES:
        u = utility[ptype] + rng.normal(0, 0.8, n)
        holds = rng.random(n) < _logistic(u)
        idx = np.flatnonzero(holds)
        if len(idx) == 0:
            continue
        # Most holdings are long-standing; a minority are acquired in the
        # outcome window (months 13-15). The chance of being a recent acquirer
        # rises with the same utility that drives holding at all -- which is
        # what makes propensity learnable from pre-window behaviour.
        recent = rng.random(len(idx)) < _clip01(0.08 + 0.35 * _logistic(u[idx]))
        start_month = np.where(
            recent,
            rng.integers(13, N_MONTHS + 1, len(idx)),
            rng.integers(-60, 13, len(idx)),
        )
        start_date = MONTH_STARTS[0] + pd.to_timedelta((start_month - 1) * 30, unit="D")
        rows.append(pd.DataFrame({
            "customer_id": cust_ids[idx],
            "product_type": ptype,
            "start_date": start_date.normalize(),
            "start_month_index": start_month,
            "status": np.where(rng.random(len(idx)) < 0.045, "Closed", "Active"),
        }))

    products = pd.concat(rows, ignore_index=True)
    return products.sort_values(["customer_id", "start_date"], ignore_index=True)


# -------------------------------------------------- service interactions ----


def _service_interactions(
    rng: np.random.Generator, customers: pd.DataFrame, activity: np.ndarray,
    lat: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Complaint volume rises as a relationship deteriorates.

    Declining customers generate more contacts, more of them complaints, and
    with lower satisfaction -- a leading indicator the churn model can use.
    """
    n = len(customers)
    # `activity` is the customer x month health matrix: 1.0 healthy, 0.0 lapsed.
    friction = np.clip(1.0 - activity, 0.0, 1.0)  # 0 healthy -> 1 fully lapsed

    base_rate = np.exp(-1.55 - 0.30 * lat["stability"])[:, None]
    lam = base_rate * (0.55 + 2.6 * friction)
    counts = rng.poisson(lam)

    total = int(counts.sum())
    row_cust = np.repeat(np.repeat(customers["customer_id"].to_numpy(), N_MONTHS), counts.ravel())
    row_month = np.repeat(np.tile(MONTH_STARTS.to_numpy(), n), counts.ravel())
    row_friction = np.repeat(friction.ravel(), counts.ravel())

    day = rng.integers(0, 28, total)
    date = pd.to_datetime(row_month) + pd.to_timedelta(day, unit="D")

    complaint_p = _clip01(0.16 + 0.55 * row_friction)
    itype = np.where(
        rng.random(total) < complaint_p, "Complaint",
        rng.choice(["Query", "Service request", "Product enquiry", "Dispute"],
                   total, p=[0.46, 0.28, 0.18, 0.08]),
    )

    resolved_p = _clip01(0.82 - 0.42 * row_friction)
    resolution = np.where(
        rng.random(total) < resolved_p, "Resolved",
        rng.choice(["Pending", "Escalated", "Unresolved"], total, p=[0.50, 0.32, 0.18]),
    )

    satisfaction = np.clip(
        np.round(4.35 - 2.1 * row_friction - 0.9 * (itype == "Complaint") + rng.normal(0, 0.65, total)),
        1, 5,
    ).astype(int)

    return pd.DataFrame({
        "interaction_id": [f"S{i:08d}" for i in range(1, total + 1)],
        "customer_id": row_cust,
        "interaction_date": date,
        "channel": rng.choice(INTERACTION_CHANNELS, total, p=[0.34, 0.19, 0.27, 0.13, 0.07]),
        "interaction_type": itype,
        "resolution_status": resolution,
        "satisfaction_score": satisfaction,
    })


# ------------------------------------------------------------ entrypoint ----


def generate_all(
    n_customers: int = N_CUSTOMERS, seed: int = RANDOM_SEED
) -> dict[str, pd.DataFrame]:
    """Generate the complete clean synthetic world.

    Returns the seven source tables. Defects are injected separately so the
    clean truth and the corrupted 'landed' data can be compared.
    """
    rng = np.random.default_rng(seed)

    lat = _latent_traits(rng, n_customers)
    customers = _customers(rng, lat)
    panel, activity = _monthly_panel(rng, customers, lat)
    accounts = _accounts(rng, customers, lat, panel)
    credit_appetite = pd.Series(lat["credit_appetite"], index=customers["customer_id"])
    transactions = _transactions(rng, panel, accounts, credit_appetite)
    loans = _loans(rng, customers, lat)
    products = _products(rng, customers, lat)
    interactions = _service_interactions(rng, customers, activity, lat)

    balances = _account_monthly_balances(accounts, panel)

    digital_activity = panel[[
        "customer_id", "month", "month_index", "mobile_logins",
        "online_logins", "digital_transactions", "failed_logins",
    ]].copy()

    return {
        "customers": customers,
        "accounts": accounts,
        "account_monthly_balances": balances,
        "transactions": transactions,
        "loans": loans,
        "products": products,
        "digital_activity": digital_activity,
        "service_interactions": interactions,
    }
