"""Next-best-category, by co-occurrence.

A classifier answering "will this account buy a category it has never bought?"
scored ROC-AUC 0.550 on this book -- barely above chance. That result is
published in the model card rather than hidden, and it is also the reason this
module exists.

The failure is not really a modelling failure, it is a framing one. Whether an
account adds *some* new category in a quarter is dominated by whether it orders
at all, and by which assortment happened to be in the catalogue that month. The
question an account manager actually asks is different and narrower:

    "This account buys kitchen and home. What should I show them next?"

That is a recommendation problem, not a binary classification, and the right
tool is item-to-item collaborative filtering over what similar accounts buy.
It is also directly usable: the decision engine can name the category, which a
propensity score never could.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ItemRecommender:
    """Item-to-item collaborative filtering over co-purchase.

    Similarity is cosine between category columns of the binary
    account x category matrix, which is the standard formulation and has the
    property that matters here: it is not dominated by category size. Christmas
    stock is bought by half the book, and a raw co-occurrence count would
    recommend it to everyone regardless of fit.
    """

    def __init__(self) -> None:
        self.similarity: pd.DataFrame | None = None
        self.popularity: pd.Series | None = None
        self.categories: list[str] = []

    def fit(self, holdings: pd.DataFrame) -> "ItemRecommender":
        """`holdings` is a boolean account x category matrix."""
        matrix = holdings.astype(float)
        self.categories = list(matrix.columns)

        norms = np.sqrt((matrix ** 2).sum(axis=0))
        norms = norms.replace(0, np.nan)
        normalised = matrix.div(norms, axis=1)

        similarity = (normalised.T @ normalised).copy()
        # Never recommend a category the account already buys.
        for category in similarity.columns:
            similarity.loc[category, category] = 0.0
        self.similarity = similarity.fillna(0.0)
        self.popularity = matrix.mean(axis=0)
        return self

    def score_all(self, holdings: pd.DataFrame) -> pd.DataFrame:
        """Affinity of every account to every category it does not yet buy."""
        if self.similarity is None:
            raise RuntimeError("fit() the recommender before scoring")
        matrix = holdings.astype(float)
        raw = matrix @ self.similarity
        # Normalise by how many categories the account already buys, so a broad
        # account does not out-score a narrow one on every candidate.
        breadth = matrix.sum(axis=1).replace(0, np.nan)
        scores = raw.div(breadth, axis=0)
        return scores.where(~holdings.astype(bool), np.nan)

    def recommend(self, holdings: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """The top-n categories per account, with the affinity behind each."""
        scores = self.score_all(holdings)
        rows = []
        ranked = scores.rank(axis=1, ascending=False, method="first")
        for position in range(1, n + 1):
            mask = ranked == position
            picked = scores.where(mask).stack(future_stack=True).dropna()
            for (customer, category), value in picked.items():
                rows.append({"customer_id": customer, "rank": position,
                             "category": category, "affinity": float(value)})
        if not rows:
            return pd.DataFrame(columns=["customer_id", "rank", "category", "affinity"])
        return pd.DataFrame(rows).sort_values(["customer_id", "rank"], ignore_index=True)

    def evaluate(self, holdings: pd.DataFrame, actual_new: pd.DataFrame, k: int = 3) -> dict:
        """Hit-rate at k against what accounts actually bought next period.

        The honest test of a recommender: of the accounts that did add a
        category, how often was it in the top k we would have suggested? Compared
        against recommending the most popular categories they lack, which is the
        benchmark any recommender has to beat to have earned its complexity.
        """
        scores = self.score_all(holdings)
        added = actual_new.reindex(index=holdings.index, columns=holdings.columns).fillna(False)
        movers = added.any(axis=1)
        if movers.sum() == 0:
            return {"accounts_evaluated": 0}

        def hit_rate(candidate_scores: pd.DataFrame) -> float:
            top = candidate_scores.loc[movers].rank(axis=1, ascending=False, method="first") <= k
            return float((top & added.loc[movers]).any(axis=1).mean())

        popular = pd.DataFrame(
            np.tile(self.popularity.values, (len(holdings), 1)),
            index=holdings.index, columns=holdings.columns,
        ).where(~holdings.astype(bool), np.nan)

        return {
            "accounts_evaluated": int(movers.sum()),
            "hit_rate_at_k": hit_rate(scores),
            "popularity_baseline": hit_rate(popular),
            "k": k,
        }



# Backwards-compatible alias: the class is item-agnostic, and this project fits
# it at two different granularities to show why one of them works.
CategoryRecommender = ItemRecommender
