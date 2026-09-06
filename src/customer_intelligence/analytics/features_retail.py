"""The retail modelling feature contract.

One list, used by training and by scoring, so "what went into this model?" is
answered by reading a single file.
"""

from __future__ import annotations

NUMERIC_FEATURES = [
    # recency, frequency, monetary — the spine of any transactional model
    "months_since_last_order", "active_months", "invoices", "revenue",
    "avg_monthly_revenue", "peak_monthly_revenue", "avg_order_value",
    "revenue_last_quarter",
    # basket shape
    "units", "lines", "lines_per_order", "avg_unit_price",
    # breadth of the relationship
    "distinct_products", "distinct_categories", "months_on_book",
    # friction
    "return_lines", "return_value", "return_rate", "discount_rate", "postage",
    # ordering rhythm — how far past their own usual gap they have drifted
    "avg_months_between_orders", "max_months_between_orders", "gap_variability",
    "cadence_overdue",
    # direction of travel
    "revenue_trend", "order_trend",
    # what they buy, as shares of their own spend
    "share_christmas", "share_home", "share_kitchen", "share_bags",
    "share_lighting", "share_party", "share_stationery", "share_storage",
    "share_garden", "share_toys", "share_jewellery",
]

CATEGORICAL_FEATURES = ["country"]

# Deliberately excluded, and why:
#
#   customer_id      an identifier, not a signal
#   segment          an RFM rule applied to features the model already sees.
#                    Including it would let the model shortcut through a label
#                    and make every explanation circular.
#   multi_country    a data quality flag, not customer behaviour
#   window_from/to   bookkeeping
EXCLUDED_FEATURES = ["customer_id", "segment", "multi_country", "window_from", "window_to"]

FEATURE_LABELS = {
    "months_since_last_order": "Months since last order",
    "active_months": "Months with an order",
    "invoices": "Orders placed",
    "revenue": "Revenue in window",
    "avg_monthly_revenue": "Average monthly revenue",
    "peak_monthly_revenue": "Best month",
    "avg_order_value": "Average order value",
    "revenue_last_quarter": "Revenue in the last quarter",
    "units": "Units bought", "lines": "Order lines",
    "lines_per_order": "Lines per order", "avg_unit_price": "Average unit price",
    "distinct_products": "Distinct products bought",
    "distinct_categories": "Categories bought from",
    "months_on_book": "Months on book",
    "return_lines": "Return lines", "return_value": "Value returned",
    "return_rate": "Return rate", "discount_rate": "Discount rate",
    "postage": "Postage paid",
    "avg_months_between_orders": "Usual gap between orders",
    "max_months_between_orders": "Longest gap between orders",
    "gap_variability": "Consistency of ordering",
    "cadence_overdue": "How far past their usual gap",
    "revenue_trend": "Revenue trend (recent vs early)",
    "order_trend": "Order trend (recent vs early)",
    "share_christmas": "Share spent on Christmas & seasonal",
    "share_home": "Share spent on home décor",
    "share_kitchen": "Share spent on kitchen & dining",
    "share_bags": "Share spent on bags & luggage",
    "share_lighting": "Share spent on lighting & candles",
    "share_party": "Share spent on party & celebration",
    "share_stationery": "Share spent on stationery & wrap",
    "share_storage": "Share spent on storage & household",
    "share_garden": "Share spent on garden & outdoor",
    "share_toys": "Share spent on toys & games",
    "share_jewellery": "Share spent on jewellery & accessories",
    "country": "Country",
}


def label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("country_"):
        return f"Country: {feature[len('country_'):]}"
    return feature.replace("_", " ").capitalize()


def model_features(target: str) -> tuple[list[str], list[str]]:
    """Numeric and categorical features for one target.

    The expansion model must not be told how many categories the account already
    buys from, because that is most of the answer to whether it will add another.
    """
    numeric = list(NUMERIC_FEATURES)
    if target == "took_new_category":
        numeric = [f for f in numeric if f != "distinct_categories"]
    return numeric, list(CATEGORICAL_FEATURES)
