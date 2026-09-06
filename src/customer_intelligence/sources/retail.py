"""Online Retail II -> the canonical source tables.

This is the data engineering that earns the rest of the project. The file as
published is one flat sheet of invoice lines, and almost everything interesting
about it is a problem:

  * `Invoice` mixes integers with `C`-prefixed strings, which are credit notes
    (returns), not sales;
  * 0.57% of stock codes are not products at all -- postage, carriage, manual
    adjustments, discounts, samples, bank charges, an Amazon commission of
    -£260,764, bad-debt write-offs, gift vouchers, and a row literally
    described as "This is a test product";
  * the same manual-adjustment code appears as both `M` and `m`;
  * 243,007 lines carry no customer identifier;
  * 4,382 carry no description and a zero price;
  * 34,335 rows are exact duplicates.

None of that was injected. It is what a real transactional extract looks like,
and each item below is a decision about what to do with it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from .download import extract
from .registry import ONLINE_RETAIL_II

RAW_CACHE = DATA_DIR / "external" / "online_retail_ii_raw.parquet"

# The panel runs from the first full month to the last. December 2011 is
# excluded: the extract stops on the 9th, so that month is a third of a month
# and would read as a collapse in demand rather than the end of the file.
PANEL_START = "2009-12-01"
PANEL_END = "2011-11-30"

# ---------------------------------------------------------------------------
# Stock codes that are not products. Keeping them in inflates basket counts and
# corrupts every per-product measure; dropping them silently loses real revenue.
# They are classified, kept in the transaction table, and flagged.
# ---------------------------------------------------------------------------
NON_PRODUCT_CODES = {
    "POST": "Postage", "DOT": "Postage", "C2": "Carriage", "POSTAGE": "Postage",
    "M": "Manual adjustment", "ADJUST": "Manual adjustment", "ADJUST2": "Manual adjustment",
    "D": "Discount", "S": "Samples", "B": "Bad debt adjustment",
    "BANK CHARGES": "Bank charges", "AMAZONFEE": "Marketplace commission",
    "CRUK": "Charity commission", "PADS": "Manual adjustment", "TEST001": "Test data",
    "TEST002": "Test data", "gift_0001_10": "Gift voucher", "gift_0001_20": "Gift voucher",
    "gift_0001_30": "Gift voucher", "gift_0001_40": "Gift voucher",
    "gift_0001_50": "Gift voucher", "gift_0001_60": "Gift voucher",
    "gift_0001_70": "Gift voucher", "gift_0001_80": "Gift voucher",
    "gift_0001_90": "Gift voucher",
}

# ---------------------------------------------------------------------------
# Product taxonomy.
#
# The dataset ships 5,305 stock codes and no categories, which is useless for
# propensity modelling -- a model over 5,305 targets learns nothing. The
# taxonomy below is DERIVED, by keyword, from the description text. It is a
# judgement, not a fact about the data, so the rules are published here and the
# coverage is reported rather than assumed.
#
# Order matters: the first rule that matches wins, so seasonal and occasion
# categories are tested before material or form categories. A Christmas candle
# is Christmas stock, not candle stock, because that is how it is bought.
# ---------------------------------------------------------------------------
CATEGORY_RULES: list[tuple[str, str]] = [
    ("Christmas & seasonal", r"CHRISTMAS|XMAS|ADVENT|SANTA|REINDEER|SNOWMAN|TINSEL|"
                             r"NATIVITY|EASTER|HALLOWEEN|VALENTINE|MISTLETOE"),
    ("Party & celebration",  r"PARTY|BUNTING|BALLOON|CONFETTI|WEDDING|BIRTHDAY|CAKE STAND|"
                             r"CAKESTAND|GARLAND"),
    ("Bags & luggage",       r"\bBAG\b|BAGS|LUGGAGE|TOTE|SHOPPER|RUCKSACK|SATCHEL|HOLDALL"),
    ("Lighting & candles",   r"T-LIGHT|TLIGHT|LIGHT|CANDLE|LANTERN|FAIRY LIGHT|LAMP|NIGHTLIGHT"),
    ("Kitchen & dining",     r"MUG|BOWL|PLATE|CUTLERY|TEAPOT|\bTEA\b|COFFEE|JUG|BAKING|"
                             r"APRON|EGG CUP|TRAY|COASTER|NAPKIN|TIN\b|JAR\b|BOTTLE|"
                             r"CUP\b|SPOON|KNIFE|CAKE|CHOPPING|COOK"),
    ("Jewellery & accessories", r"NECKLACE|BRACELET|EARRING|PENDANT|BROOCH|\bRING\b|"
                                r"SCARF|PURSE|HAIR |UMBRELLA|GLOVE|SLIPPER"),
    ("Toys & games",         r"\bTOY\b|GAME|PUZZLE|DOLL|TEDDY|SKITTLE|SPACEBOY|PLAYHOUSE|"
                             r"BINGO|JIGSAW|SOLDIER|RATTLE|YO-YO"),
    ("Stationery & wrap",    r"CARD\b|CARDS|WRAP|GIFT WRAP|NOTEBOOK|PENCIL|\bPEN\b|CHALK|"
                             r"STICKER|ENVELOPE|PAPER|JOURNAL|TAPE|POSTCARD|NOTE ?BOOK"),
    ("Garden & outdoor",     r"GARDEN|PLANT|WATERING|PARASOL|DECKCHAIR|BIRD |BIRDHOUSE|"
                             r"WINDMILL|TROWEL|SEED"),
    ("Home décor",           r"HEART|HANGING|WALL|SIGN\b|FRAME|MIRROR|CUSHION|CLOCK|VASE|"
                             r"ORNAMENT|DECORATION|DOORMAT|HOOK|DRAWER|CABINET|CANDLESTICK|"
                             r"PHOTO|PICTURE|BUNNY|FLOWER|ROSE|CHANDELIER|DOORSTOP"),
    ("Storage & household",  r"BOX\b|BOXES|BASKET|CRATE|TIN SET|BIN\b|HOLDER|RACK|STAND|"
                             r"CADDY|ORGANISER|LAUNDRY|TOWEL|BLANKET|THROW"),
]

_COMPILED = [(name, re.compile(pattern)) for name, pattern in CATEGORY_RULES]
UNCATEGORISED = "Other giftware"


def categorise(description: str | float) -> str:
    """Assign one derived category to a product description."""
    if not isinstance(description, str) or not description.strip():
        return UNCATEGORISED
    text = description.upper()
    for name, pattern in _COMPILED:
        if pattern.search(text):
            return name
    return UNCATEGORISED


# ---------------------------------------------------------------------------


def load_raw(force: bool = False) -> pd.DataFrame:
    """Read the published workbook once, then cache it as parquet.

    Parsing 1.07 million rows out of a 45 MB xlsx takes about ninety seconds and
    the result never changes, so it is done once per machine.
    """
    if RAW_CACHE.exists() and not force:
        return pd.read_parquet(RAW_CACHE)

    path = extract(ONLINE_RETAIL_II, force=force)
    workbook = pd.ExcelFile(path)
    frame = pd.concat(
        [workbook.parse(sheet) for sheet in workbook.sheet_names], ignore_index=True
    )
    # `Invoice` and `StockCode` are genuinely mixed-type: numeric for ordinary
    # rows, alphanumeric for credit notes and administrative codes. Read as
    # strings, or half the file silently fails to load.
    for column in ("Invoice", "StockCode", "Description", "Country"):
        frame[column] = frame[column].astype("string")
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RAW_CACHE, index=False)
    return frame


def build_source_tables(force: bool = False) -> dict[str, pd.DataFrame]:
    """Turn the published sheet into the project's canonical source tables.

    The output is deliberately *not* cleaned. Duplicates, missing identifiers,
    zero prices and returns all survive into the raw layer, because the quality
    framework has to be given something to find.
    """
    raw = load_raw(force=force)

    df = raw.rename(columns={
        "Invoice": "invoice_id", "StockCode": "stock_code", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "unit_price",
        "Customer ID": "customer_id", "Country": "country",
    }).copy()

    # Restrict to whole months. See PANEL_END above.
    df = df[(df["invoice_date"] >= PANEL_START) & (df["invoice_date"] <= PANEL_END + " 23:59:59")]

    origin = pd.Period(PANEL_START, freq="M")
    df["month_index"] = (
        df["invoice_date"].dt.to_period("M").astype("period[M]").apply(lambda p: (p - origin).n) + 1
    )
    df["month"] = df["invoice_date"].dt.to_period("M").dt.to_timestamp()

    # Customer identifiers arrive as floats because of the missing values.
    df["customer_id"] = np.where(
        df["customer_id"].notna(),
        "C" + df["customer_id"].fillna(0).astype("int64").astype(str),
        None,
    )

    df["stock_code"] = df["stock_code"].str.strip()
    # `M` and `m` are the same manual-adjustment code entered two ways. Upper-
    # casing the administrative codes only -- product codes are case-significant
    # (84031A and 84031B are different items).
    admin_mask = df["stock_code"].str.upper().isin({k.upper() for k in NON_PRODUCT_CODES})
    df.loc[admin_mask, "stock_code"] = df.loc[admin_mask, "stock_code"].str.upper()

    df["is_return"] = df["invoice_id"].str.startswith("C", na=False)
    df["line_value"] = (df["quantity"] * df["unit_price"]).round(2)

    code_upper = df["stock_code"].str.upper()
    non_product = {k.upper(): v for k, v in NON_PRODUCT_CODES.items()}
    df["line_type"] = code_upper.map(non_product).fillna("Product")
    df["is_product"] = df["line_type"] == "Product"

    df["line_id"] = np.arange(1, len(df) + 1)

    transactions = df[[
        "line_id", "invoice_id", "customer_id", "stock_code", "description",
        "quantity", "unit_price", "line_value", "invoice_date", "month",
        "month_index", "country", "is_return", "is_product", "line_type",
    ]].reset_index(drop=True)

    # ---- products ---------------------------------------------------------
    products = (
        df[df["is_product"]]
        .groupby("stock_code", observed=True)
        .agg(
            description=("description", lambda s: s.dropna().mode().iat[0]
                         if s.notna().any() else None),
            median_price=("unit_price", "median"),
            lines=("line_id", "size"),
            units=("quantity", "sum"),
            revenue=("line_value", "sum"),
            first_month=("month_index", "min"),
            last_month=("month_index", "max"),
        )
        .reset_index()
    )
    products["category"] = products["description"].map(categorise)

    # ---- customers --------------------------------------------------------
    identified = df[df["customer_id"].notna()]
    customers = (
        identified.groupby("customer_id", observed=True)
        .agg(
            country=("country", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None),
            countries_seen=("country", "nunique"),
            first_invoice_date=("invoice_date", "min"),
            last_invoice_date=("invoice_date", "max"),
            first_month_index=("month_index", "min"),
            last_month_index=("month_index", "max"),
            invoices=("invoice_id", "nunique"),
        )
        .reset_index()
    )

    # ---- invoices ---------------------------------------------------------
    invoices = (
        df.groupby("invoice_id", observed=True)
        .agg(
            customer_id=("customer_id", "first"),
            invoice_date=("invoice_date", "min"),
            month_index=("month_index", "min"),
            lines=("line_id", "size"),
            units=("quantity", "sum"),
            invoice_value=("line_value", "sum"),
            country=("country", "first"),
            is_return=("is_return", "first"),
        )
        .reset_index()
    )

    return {
        "transactions": transactions,
        "products": products,
        "customers": customers,
        "invoices": invoices,
    }
