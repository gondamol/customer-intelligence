# Data dictionary

Every field in the source layer, what it means in business terms, how sensitive
it would be in a real deployment, and the rule that validates it.

**Sensitivity classifications** describe how each field *would* be treated if the
data were real. All data here is synthetic and none of it is confidential; the
classification is part of the demonstration, because deciding sensitivity before
building is what stops a system from quietly becoming one that should not exist.

| Class | Meaning |
|---|---|
| **Identifier** | Keys a record. Pseudonymous here by construction. |
| **Internal** | Ordinary operational data. Restricted to the analytical team. |
| **Sensitive** | Would be personal or financial data under most regimes. Access logged, minimised, and justified per use. |
| **Derived** | Produced by this system. Carries the sensitivity of its inputs plus the risk of being mistaken for fact. |

Monetary amounts are in **monetary units (MU)** — deliberately unnamed, so no
figure can be read as a currency amount for any real market.

---

## `customers` — one row per customer

| Field | Type | Description & business meaning | Allowed values | Sensitivity | Validation rule |
|---|---|---|---|---|---|
| `customer_id` | string | Unique synthetic customer identifier. The join key for the entire model. | `C` + 6 digits | Identifier | Unique, non-null |
| `age` | integer | Age in years. Drives life-stage product relevance. | 18–100 | Sensitive | Between 18 and 100; impossible values nulled, never imputed |
| `gender` | string | Recorded gender. Held for outcome monitoring only; **excluded from every model** — see note below. | Female, Male | Sensitive | In the permitted set |
| `region` | string | Geographic region of the primary relationship. | 7 named regions | Internal | Non-null after conformance (`Unknown` where absent) |
| `employment_type` | string | Employment category. Context for income stability. | Salaried, Self-employed, Business owner, Informal, Retired | Sensitive | In the permitted set |
| `income_band` | string | Banded declared income. Banded rather than exact to reduce precision to what the analysis needs. | Band 1–5, Unknown | Sensitive | In the permitted set |
| `monthly_income` | numeric | Declared monthly income in MU. | > 0 | Sensitive | Positive where present |
| `tenure_months` | integer | Months since the relationship opened. | ≥ 0 | Internal | Cannot exceed `(age − 17) × 12` |
| `customer_segment` | string | Segment declared at source. **Excluded from every model** — see `model_governance.md`. | 7 named segments | Derived | In the permitted set |
| `join_date` | date | Date the relationship opened. | ≤ reporting close | Internal | Consistent with `tenure_months` |

> **On `gender`:** it is carried in the source data and shown in the Customer
> 360 profile, because a real extract would contain it and pretending otherwise
> would misrepresent the problem. It is **not** an input to any model. "It was
> in the data" is not a reason to let a commercial targeting model condition on
> a protected attribute — if the model found it predictive, that would be the
> problem rather than the justification. It is retained for outcome
> *monitoring*: checking whether flag rates and recommended actions differ
> across groups, which is the use that justifies holding the attribute at all.
> `tests/test_leakage.py::test_gender_is_not_a_model_feature` enforces this.

## `accounts` — one row per account

| Field | Type | Description & business meaning | Allowed values | Sensitivity | Validation rule |
|---|---|---|---|---|---|
| `account_id` | string | Unique account identifier. | `A` + 7 digits | Identifier | Unique, non-null |
| `customer_id` | string | Owning customer. | FK → `customers` | Identifier | Must resolve; orphans dropped |
| `account_type` | string | Product class of the account. | Current, Savings, Fixed deposit, Wallet | Internal | In the permitted set |
| `opening_date` | date | When the account was opened. | ≤ reporting close | Internal | Not in the future |
| `status` | string | Operational state. | Active, Dormant, Closed, Unknown | Internal | Mapped to the controlled vocabulary |
| `average_balance` | numeric | Mean balance over the observation period, MU. | ≥ 0 for deposit products | Sensitive | Non-negative on Savings / Fixed deposit |
| `current_balance` | numeric | Balance at the close of the window, MU. | ≥ 0 for deposit products | Sensitive | Negative deposit balances nulled as sign errors |

## `account_monthly_balances` — one row per account per month

| Field | Type | Description & business meaning | Allowed values | Sensitivity | Validation rule |
|---|---|---|---|---|---|
| `account_id` | string | Account the balance belongs to. | FK → `accounts` | Identifier | Must resolve |
| `customer_id` | string | Owning customer. | FK → `customers` | Identifier | Must resolve |
| `month` | date | First day of the calendar month. | Panel months | Internal | Within the panel |
| `month_index` | integer | Month number, 1 = first panel month. | 1–15 | Internal | Within the panel |
| `closing_balance` | numeric | Month-end balance, MU. The basis of every balance trend. | ≥ 0 | Sensitive | Non-negative |

## `transactions` — one row per posting

| Field | Type | Description & business meaning | Allowed values | Sensitivity | Validation rule |
|---|---|---|---|---|---|
| `transaction_id` | string | Unique posting identifier. | `T` + 9 digits | Identifier | Unique; duplicates are retries and are dropped |
| `customer_id` | string | Transacting customer. | FK → `customers` | Identifier | Must resolve |
| `account_id` | string | Account debited or credited. Must belong to the same customer. | FK → `accounts` | Identifier | Must resolve to the customer's own account |
| `transaction_date` | date | Date of the posting. | < reporting close | Internal | Not after the reporting close |
| `transaction_type` | string | Direction of the money. | Credit, Debit | Internal | In the permitted set |
| `amount` | numeric | Value in MU. | > 0, ≤ 5,000,000 | Sensitive | Positive; beyond the ceiling it is quarantined, not capped |
| `channel` | string | Where the transaction happened. Basis of digital engagement. | Mobile app, Internet banking, Branch, ATM, Agent, Card POS | Internal | In the permitted set |
| `merchant_category` | string | Spend category. | 12 categories | Sensitive | In the permitted set |

## `loans` — one row per facility

| Field | Type | Description & business meaning | Allowed values | Sensitivity | Validation rule |
|---|---|---|---|---|---|
| `loan_id` | string | Unique facility identifier. | `L` + 7 digits | Identifier | Unique, non-null |
| `customer_id` | string | Borrowing customer. | FK → `customers` | Identifier | Must resolve |
| `loan_type` | string | Facility class. | Personal loan, Asset finance, Overdraft, Microloan | Internal | In the permitted set |
| `principal` | numeric | Amount originally advanced, MU. | > 0 | Sensitive | Positive |
| `outstanding_balance` | numeric | Amount still owed, MU. | 0 ≤ x ≤ 1.5 × principal | Sensitive | Cannot exceed the principal advanced |
| `interest_rate` | numeric | Annual rate, as a decimal. | 0.07–0.36 | Sensitive | Required on any facility not Closed |
| `term_months` | integer | Contractual term. | 12–60 | Internal | In the permitted set |
| `repayment_status` | string | Current standing. Drives the arrears guardrail in the decision engine. | Current, Late 1-30, Late 31-90, Default, Closed | Sensitive | In the permitted set |
| `origination_date` | date | When the facility was advanced. | ≤ reporting close | Internal | Not in the future |

## `products` — one row per holding

| Field | Type | Description & business meaning | Allowed values | Sensitivity | Validation rule |
|---|---|---|---|---|---|
| `customer_id` | string | Holding customer. | FK → `customers` | Identifier | Must resolve |
| `product_type` | string | Canonical product name. Source spellings are mapped on conformance. | 8 canonical names | Internal | In the canonical vocabulary |
| `start_date` | date | When the holding began. | — | Internal | — |
| `start_month_index` | integer | Panel month the holding began. **Governs leakage**: a holding starting after the window close is never counted. | integer | Internal | ≤ window close for any feature |
| `status` | string | Whether the holding is live. | Active, Closed | Internal | In the permitted set |

## `digital_activity` — one row per customer per month

| Field | Type | Description & business meaning | Allowed values | Sensitivity | Validation rule |
|---|---|---|---|---|---|
| `customer_id` | string | The customer. | FK → `customers` | Identifier | Must resolve |
| `month` / `month_index` | date / int | Panel month. | 1–15 | Internal | Within the panel |
| `mobile_logins` | integer | App sessions. Leading indicator of engagement. | ≥ 0 | Internal | Non-negative |
| `online_logins` | integer | Web sessions. | ≥ 0 | Internal | Non-negative |
| `digital_transactions` | integer | Transactions initiated in a digital channel. | ≥ 0, ≤ transaction count | Internal | Non-negative |
| `failed_logins` | integer | Failed authentication attempts. Friction signal. | ≥ 0 | Internal | Non-negative |

## `service_interactions` — one row per contact

| Field | Type | Description & business meaning | Allowed values | Sensitivity | Validation rule |
|---|---|---|---|---|---|
| `interaction_id` | string | Unique contact identifier. | `S` + 8 digits | Identifier | Unique, non-null |
| `customer_id` | string | The customer. | FK → `customers` | Identifier | Must resolve |
| `interaction_date` | date | Date of contact. | Within the panel | Internal | Within the panel |
| `channel` | string | How they got in touch. | Call centre, Branch, In-app chat, Email, Social | Internal | In the permitted set |
| `interaction_type` | string | What it was about. Complaints drive the service-recovery rule. | Query, Complaint, Service request, Product enquiry, Dispute | Sensitive | In the permitted set |
| `resolution_status` | string | Whether it was closed out. Unresolved contacts outrank commercial actions. | Resolved, Pending, Escalated, Unresolved | Internal | In the permitted set |
| `satisfaction_score` | integer | Post-contact rating. **Missing not at random** — dissatisfied customers respond less. | 1–5, or null | Sensitive | In range where present |

---

## Derived fields worth explaining

These are produced by `sql/03_customer_360.sql` and are what the models actually
consume. All are **Derived** sensitivity.

| Field | How it is computed | Why it exists |
|---|---|---|
| `balance_trend` | Mean balance over the last 3 months of the window ÷ mean over the first 3. Defaults to 1.0 when the base is zero. | Direction of travel. A high-balance customer in steep decline and a low-balance customer growing steadily look identical on level alone. |
| `transaction_trend` | Same construction on transaction counts. | The strongest single attrition signal in the model. |
| `digital_trend` | Same construction on total logins. | Disengagement usually shows in the channel before it shows in the balance. |
| `months_since_last_activity` | Window close minus the last month with any transaction. | Silence, measured. Requires the full monthly spine — a panel built by aggregating transactions alone would drop exactly these customers. |
| `active_months` | Count of months in the window with at least one transaction. | Distinguishes a consistently light user from one who stopped. |
| `digital_share` | Digital transactions ÷ all transactions. | Cost to serve, and which channel a recommendation should use. |
| `balance_to_income` | Average monthly balance ÷ declared monthly income. | Financial posture. The observable trace of credit appetite, which is otherwise latent. |
| `outflow_to_inflow` | Total debits ÷ total credits. | How much of what arrives leaves again. |
| `opportunity_score` | Weighted sum of five percentile-ranked components. Weights published in `config.py`. | Headroom in the relationship. **Not a revenue figure.** |
| `churn_probability` | Logistic regression on the 12-month feature window. | Modelled attrition risk. A probability, not a verdict. |
| `investment_propensity` / `lending_propensity` | Logistic regression on the eligible population only. Null where the customer already holds the product. | Likelihood of uptake next period. **Not a suitability or credit assessment.** |
