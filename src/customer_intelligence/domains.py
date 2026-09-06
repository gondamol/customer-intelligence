"""Per-domain configuration.

The same pipeline runs over more than one real dataset. What changes between
them is the panel length, the analysis windows and the outcome definitions --
so those live here, per domain, rather than as constants that quietly mean
something different depending on which build last ran.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    key: str
    name: str
    dataset_key: str
    panel_months: int
    train_feature: tuple[int, ...]
    train_outcome: tuple[int, ...]
    score_feature: tuple[int, ...]
    currency: str
    entity: str                      # what one row of the 360 represents
    window_rationale: str = ""

    @property
    def horizon(self) -> int:
        return len(self.train_outcome)


# ---------------------------------------------------------------------------
# Online Retail II — the primary domain.
#
# The panel is 24 whole months (Dec 2009 – Nov 2011). December 2011 is dropped:
# the extract stops on the 9th, and a third of a month reads as a collapse in
# demand rather than the end of a file.
#
# The windows are chosen for SEASON, not convenience. This is a giftware
# wholesaler: November 2011 carries 84,711 lines against 27,707 in February.
# A model trained to predict a December–February outcome must be applied to
# predict a December–February outcome, or the base rate it learned is simply
# the wrong time of year. So:
#
#   train   features months  1–12  (Dec 09 – Nov 10) -> outcome 13–15 (Dec 10 – Feb 11)
#   score   features months 13–24  (Dec 10 – Nov 11) -> outcome 25–27 (Dec 11 – Feb 12, unobserved)
#
# Both feature windows are a full twelve months, so seasonality is neutralised
# inside them; both outcome windows are the same three calendar months.
# ---------------------------------------------------------------------------
RETAIL = Domain(
    key="retail",
    name="Online Retail II — UK giftware wholesaler",
    dataset_key="online_retail_ii",
    panel_months=24,
    train_feature=tuple(range(1, 13)),
    train_outcome=tuple(range(13, 16)),
    score_feature=tuple(range(13, 25)),
    currency="£",
    entity="customer account",
    window_rationale=(
        "Feature windows are twelve whole months so seasonality is averaged out "
        "inside them. Both outcome windows are December–February, so the model "
        "is applied to the same season it was trained on."
    ),
)

# ---------------------------------------------------------------------------
# Telco Customer Churn — the second domain.
#
# No panel: one row per customer with a contractual churn flag. It is here to
# do something the retail data cannot, which is to supply a *commercially*
# defined churn label -- an account that was actually closed, rather than a
# customer who stopped appearing. The retail model's headline number is not
# comparable to it, and the point of running both is to show why.
# ---------------------------------------------------------------------------
TELCO = Domain(
    key="telco",
    name="Telco Customer Churn — contractual attrition",
    dataset_key="telco_churn",
    panel_months=0,
    train_feature=(),
    train_outcome=(),
    score_feature=(),
    currency="$",
    entity="subscriber",
    window_rationale=(
        "A cross-section, not a panel. Churn is the contractual flag supplied "
        "with the data, so no window design is required or possible."
    ),
)

DOMAINS = {d.key: d for d in (RETAIL, TELCO)}
DEFAULT = RETAIL
