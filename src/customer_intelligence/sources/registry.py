"""The datasets this project uses, and where they come from.

Every source is real, openly licensed, and downloaded from its original
publisher at build time. Nothing here is redistributed in this repository: the
raw files are fetched into ``data/external/`` on first run and cached, so the
licence terms travel with the data rather than being assumed.

Attribution is a field on the record, not a line in a README that can drift out
of date -- the application reads it from here and prints it on the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    publisher: str
    url: str                    # the file actually downloaded
    landing_page: str           # where a reader can verify all of this
    licence: str
    licence_url: str
    attribution: str
    citation: str
    description: str
    filename: str
    member: str = ""            # file to extract, when the download is an archive
    sha256: str = ""            # filled in on first download; see verify()

    @property
    def is_archive(self) -> bool:
        return self.url.lower().endswith(".zip")


ONLINE_RETAIL_II = Dataset(
    key="online_retail_ii",
    name="Online Retail II",
    publisher="UCI Machine Learning Repository (dataset 502)",
    url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    landing_page="https://archive.ics.uci.edu/dataset/502/online+retail+ii",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/legalcode",
    attribution="Chen, D. (2012). Online Retail II. UCI Machine Learning Repository.",
    citation=(
        "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning "
        "Repository. https://doi.org/10.24432/C5CG6D"
    ),
    description=(
        "Two years of real invoice-level transactions from a UK-based online "
        "retailer selling giftware, largely to wholesale customers. "
        "1,067,371 rows, 5,942 identified customers, 43 countries, "
        "December 2009 to December 2011."
    ),
    filename="online_retail_ii.zip",
    member="online_retail_II.xlsx",
)

TELCO_CHURN = Dataset(
    key="telco_churn",
    name="Telco Customer Churn",
    publisher="IBM (sample dataset, via the IBM/telco-customer-churn-on-icp4d repository)",
    url=("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
         "master/data/Telco-Customer-Churn.csv"),
    landing_page="https://github.com/IBM/telco-customer-churn-on-icp4d",
    licence="Apache-2.0 (repository licence)",
    licence_url="https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE",
    attribution="IBM sample dataset: Telco Customer Churn.",
    citation=(
        "IBM. Telco Customer Churn [Dataset]. IBM Cognos Analytics sample data, "
        "distributed via github.com/IBM/telco-customer-churn-on-icp4d"
    ),
    description=(
        "7,043 telecommunications customers with a contractually defined churn "
        "flag, service holdings, contract type, tenure and charges. Used here "
        "as a second domain, and as the commercially-defined churn label that "
        "transactional data cannot supply."
    ),
    filename="Telco-Customer-Churn.csv",
)

BANK_MARKETING = Dataset(
    key="bank_marketing",
    name="Bank Marketing",
    publisher="UCI Machine Learning Repository (dataset 222)",
    url="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    landing_page="https://archive.ics.uci.edu/dataset/222/bank+marketing",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/legalcode",
    attribution="Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository.",
    citation=(
        "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C5K306"
    ),
    description=(
        "41,188 direct marketing contacts from a Portuguese banking "
        "institution, with a real product-uptake outcome (term deposit "
        "subscription). Used as the third domain."
    ),
    filename="bank_marketing.zip",
    member="bank-additional/bank-additional-full.csv",
)

DATASETS: dict[str, Dataset] = {
    d.key: d for d in (ONLINE_RETAIL_II, TELCO_CHURN, BANK_MARKETING)
}

PRIMARY = ONLINE_RETAIL_II


def attribution_table() -> list[dict]:
    """Everything the application needs to credit its sources."""
    return [
        {
            "Dataset": d.name,
            "Publisher": d.publisher,
            "Licence": d.licence,
            "Attribution": d.attribution,
            "Source": d.landing_page,
        }
        for d in DATASETS.values()
    ]
