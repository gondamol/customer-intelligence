# Data dictionary

Every field, what it means in business terms, how sensitive it would be in a real
deployment, and the rule that validates it.

**Source:** Online Retail II — Chen, D. (2012), UCI Machine Learning Repository,
CC BY 4.0. Monetary amounts are pounds sterling as published.

**Sensitivity classifications** describe how each field *would* be treated in a
live system. This dataset is pseudonymous and openly published; the
classification is part of the demonstration, because deciding sensitivity before
building is what stops a system becoming one that should not exist.

| Class | Meaning |
|---|---|
| **Identifier** | Keys a record. Pseudonymous here by construction. |
| **Internal** | Ordinary operational data, restricted to the analytical team. |
| **Commercial** | Reveals a customer's trading position. Access justified per use. |
| **Derived** | Produced by this system. Carries its inputs' sensitivity plus the risk of being mistaken for fact. |

---

## As published

The source ships a single sheet with eight columns.

| Field | Type | Description | Sensitivity | What is wrong with it |
|---|---|---|---|---|
| `Invoice` | mixed | Invoice number. `C`-prefixed values are **credit notes**, not sales. | Identifier | Mixes integers and strings; read as numeric, 19,494 credit notes vanish |
| `StockCode` | mixed | Product code. 0.57% are administrative, not products. | Identifier | Mixed case for the same admin concept (`M` / `m`) |
| `Description` | string | Product description. | Internal | 4,382 missing; 1,215 codes carry conflicting descriptions |
| `Quantity` | integer | Units. Negative on returns. | Commercial | 22,950 negative; 3,427 on invoices that are not credit notes |
| `InvoiceDate` | datetime | Timestamp of the invoice. | Internal | December 2011 is a partial month |
| `Price` | float | Unit price. | Commercial | 6,207 at zero or below |
| `Customer ID` | float | Customer identifier. | Identifier | **243,007 rows are null** |
| `Country` | string | Billing country. | Internal | 13 accounts invoice to more than one |

---

## `transactions` — one row per invoice line

| Field | Type | Business meaning | Allowed values | Sensitivity | Validation |
|---|---|---|---|---|---|
| `line_id` | integer | Surrogate key assigned on load | > 0 | Identifier | Unique |
| `invoice_id` | string | Invoice this line belongs to | `C`-prefix = credit note | Identifier | Non-null |
| `customer_id` | string | Owning account, `C` + digits | null where unattributed | Identifier | Must resolve where present |
| `stock_code` | string | Product or administrative code | — | Identifier | Case-normalised for admin codes only |
| `description` | string | What was sold | — | Internal | Non-null in the conformed layer |
| `quantity` | integer | Units; negative on a return | ≠ 0 | Commercial | Positive on a sale line |
| `unit_price` | float | Price per unit, £ | > 0 on a sale | Commercial | Positive on a sale line |
| `line_value` | float | `quantity × unit_price`, £ | — | Commercial | \|value\| ≤ £50,000 |
| `invoice_date` | datetime | When it was invoiced | Within the panel | Internal | 2009-12-01 to 2011-11-30 |
| `month_index` | integer | Panel month, 1 = Dec 2009 | 1–24 | Derived | Within the panel |
| `country` | string | Billing country | 43 values | Internal | — |
| `is_return` | boolean | Line sits on a credit note | — | Derived | — |
| `is_product` | boolean | A product, not a charge | — | Derived | — |
| `line_type` | string | Classification of the line | 12 values | Derived | In the vocabulary |

### `line_type` — the classification that keeps charges out of product analysis

| Value | What it is | Example scale |
|---|---|---|
| `Product` | An actual product | 99.4% of lines |
| `Postage`, `Carriage` | Shipping charged to the customer | £434,988 |
| `Manual adjustment` | Booked by hand (`M`, `m`, `ADJUST`, `PADS`) | −£82,796 |
| `Discount` | Discount line | −£13,485 |
| `Samples` | Samples sent | −£6,066 |
| `Bank charges` | Bank fees booked to the ledger | −£35,563 |
| `Marketplace commission` | Amazon commission | **−£260,764** |
| `Charity commission` | CRUK commission | −£7,933 |
| `Bad debt adjustment` | Written off | **−£147,614** |
| `Gift voucher` | Vouchers sold | — |
| `Test data` | **"This is a test product"** | 17 lines |

## `products` — one row per stock code

| Field | Type | Business meaning | Sensitivity | Validation |
|---|---|---|---|---|
| `stock_code` | string | Product code | Identifier | Unique |
| `description` | string | **Modal** description across all lines for that code | Internal | Modal, because 1,215 codes carry more than one |
| `category` | string | **Derived** taxonomy, 11 categories + `Other giftware` | Derived | In the vocabulary |
| `median_price` | float | Median unit price, £ | Commercial | > 0 |
| `lines`, `units`, `revenue` | numeric | Lifetime totals | Commercial | ≥ 0 |
| `first_month`, `last_month` | integer | Panel months first and last sold | Derived | 1–24 |

**The taxonomy is a judgement.** Ordered keyword rules over the description text,
published in `sources/retail.py`, covering 87.7% of revenue. Order matters:
seasonal and occasion rules are tested before material and form rules, so a
Christmas candle is Christmas stock rather than candle stock, because that is how
it is bought.

## `customers` — one row per identified account

| Field | Type | Business meaning | Sensitivity | Validation |
|---|---|---|---|---|
| `customer_id` | string | Pseudonymous account identifier | Identifier | Unique, non-null |
| `country` | string | Modal billing country | Internal | — |
| `countries_seen` | integer | Distinct countries invoiced | Derived | Flagged where > 1 |
| `first_invoice_date` | datetime | First order | Internal | Within the panel |
| `first_month_index` | integer | Panel month of first order | Derived | Bounds the monthly spine |
| `invoices` | integer | Lifetime order count | Commercial | > 0 |

## `invoices` — one row per invoice

Grain: `invoice_id`. Carries customer, date, month index, line count, units,
value, country, and whether it is a credit note.

---

## Derived — the Customer 360

Produced by `retail_03_customer_360.sql` over a bounded month window. All
**Derived** sensitivity. These are what the models actually consume.

### Recency, frequency, monetary

| Field | How it is computed | Why it exists |
|---|---|---|
| `months_since_last_order` | Window close minus last active month | The classic recency measure |
| `active_months` | Months in the window with an order | Distinguishes a light regular buyer from one who stopped |
| `invoices`, `revenue`, `units`, `lines` | Sums over the window | Frequency and value |
| `avg_order_value` | `revenue / invoices` | Order size, independent of frequency |
| `avg_monthly_revenue`, `peak_monthly_revenue` | Mean and max monthly | Level and best case |
| `revenue_last_quarter` | Last three months of the window | Where the account is *now* |

### Ordering rhythm — the features that carry the lapse model

| Field | How it is computed | Why it exists |
|---|---|---|
| `avg_months_between_orders` | Mean gap between active months | Every account has a cadence |
| `max_months_between_orders` | Longest observed gap | How quiet this account gets normally |
| `gap_variability` | Standard deviation of gaps | Steady or erratic |
| **`cadence_overdue`** | Months silent ÷ usual gap | **The strongest single input to the lapse model.** A quarterly buyer and a weekly buyer both six weeks silent are in entirely different states, and only this feature can tell them apart |

### Breadth and mix

| Field | How it is computed | Why it exists |
|---|---|---|
| `distinct_products`, `distinct_categories` | Counted within the window | Depth of the relationship |
| `lines_per_order` | Lines ÷ invoices | Basket shape |
| `avg_unit_price` | Mean unit price paid | Premium or value buyer |
| `share_christmas` … `share_jewellery` | Category spend ÷ total spend | **Shares, not amounts**: a large and a small account buying the same mix should look alike on mix and differ on value |

### Friction

| Field | How it is computed | Why it exists |
|---|---|---|
| `return_lines`, `return_value` | From the separated returns table | Never netted off sales |
| `return_rate` | `return_value / revenue` | Drives the service-recovery rule at ≥ 20% |
| `discount_rate` | `discounts / revenue` | Margin pressure |
| `postage` | Postage and carriage paid | Cost to serve |

### Direction of travel

| Field | How it is computed | Why it exists |
|---|---|---|
| `revenue_trend`, `order_trend` | Last 3 months ÷ first 3, floored at 1.0 when the base is zero | Level says what an account is; trend says where it is going |

### Scores

| Field | Meaning | Not |
|---|---|---|
| `lapse_risk` | P(no order next quarter). **Null** where not scoreable | A statement they have left |
| `growth_propensity` | P(beats the same quarter a year earlier). **Null** where not scoreable | A revenue forecast |
| `opportunity_score` | Weighted sum of five percentile-ranked components, 0–100 | A revenue figure |
| `next_best_product` | Top item from the recommender | A prediction that they will buy it |
| `segment` | RFM rule set, quintiles within this book | A fitted clustering |
| `quadrant` | Risk × opportunity placement, or `Not scored` | A placement invented from a missing score |
| `scoreable` | ≥3 active months in the window | A judgement about account value |
