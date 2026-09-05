"""Training, evaluation and explanation of the predictive models.

Three models share this code: attrition risk, investment propensity and lending
propensity. They differ only in target and eligible population.

Two decisions are worth stating plainly, because they are the ones an
interviewer should press on:

1. **Logistic regression is the production model.** Gradient boosting is trained
   alongside it and its score reported honestly, but a small AUC gain does not
   pay for a model whose reasons cannot be read off the coefficients. Where the
   gap is large the trade-off would be worth revisiting; the comparison table
   exists so that judgement is made on evidence rather than taste.

2. **The operating threshold is not 0.5.** These outcomes are rare, so 0.5 flags
   almost nobody. The threshold is set to flag a fixed share of the book --
   because outreach capacity is finite, and the real constraint is how many
   customers a team can actually contact, not where a probability crosses a half.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import label, model_features, prepare

# Share of the eligible book the model is allowed to flag. This is a capacity
# assumption, stated here so it can be argued with.
CAPACITY_SHARE = {"churned": 0.10, "took_investment": 0.15, "took_lending": 0.15}

RANDOM_STATE = 42


@dataclass
class ModelResult:
    target: str
    display_name: str
    estimator: Pipeline
    metrics: dict
    comparison: pd.DataFrame
    importances: pd.DataFrame
    coefficients: pd.DataFrame
    calibration: pd.DataFrame
    confusion: np.ndarray
    threshold: float
    base_rate: float
    n_train: int
    n_test: int
    feature_names: list[str] = field(default_factory=list)


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                     sparse_output=False)),
        ]), categorical),
    ], remainder="drop")


def _candidates() -> dict[str, object]:
    return {
        # No class weighting. It is the obvious reflex on an imbalanced target,
        # and it is wrong here: balancing inflates predicted probabilities toward
        # 0.5, so a model with a 11% base rate predicts a mean of 0.36 and every
        # probability band, risk cut-off and calibration curve downstream becomes
        # meaningless. Nothing is gained in exchange, because the operating
        # threshold is a capacity quantile of the scores, not 0.5 -- ranking is
        # what the threshold consumes, and ranking is unaffected by weighting.
        "Logistic regression": LogisticRegression(
            max_iter=2000, C=0.5, random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(
            n_estimators=220, max_depth=12, min_samples_leaf=25,
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.08,
            subsample=0.85, random_state=RANDOM_STATE),
    }


def _threshold_for_capacity(scores: np.ndarray, share: float) -> float:
    """The probability above which exactly `share` of the book sits."""
    return float(np.quantile(scores, 1.0 - share))


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate by decile of predicted probability.

    The question this answers is not "does it rank well" but "when it says 20%,
    does 20% happen". A model used to decide who gets contacted must be right
    about the level, not only the order.
    """
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    out = df.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    ).reset_index(drop=True)
    return out


def train_model(
    features: pd.DataFrame, y: pd.Series, target: str, display_name: str,
) -> ModelResult:
    """Fit, compare, evaluate and explain one model."""
    numeric, categorical = model_features(target)
    numeric = [c for c in numeric if c in features.columns]
    categorical = [c for c in categorical if c in features.columns]
    X = prepare(features, numeric, categorical)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # ---- compare candidates on held-out data -------------------------------
    rows, fitted = [], {}
    for name, clf in _candidates().items():
        pipe = Pipeline([("prep", _preprocessor(numeric, categorical)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = (pipe, p)
        rows.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, p),
            "PR-AUC": average_precision_score(y_test, p),
            "Brier": brier_score_loss(y_test, p),
        })
    comparison = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False, ignore_index=True)

    # ---- logistic regression is the model that ships ------------------------
    chosen_name = "Logistic regression"
    model, p_test = fitted[chosen_name]

    threshold = _threshold_for_capacity(p_test, CAPACITY_SHARE.get(target, 0.10))
    y_hat = (p_test >= threshold).astype(int)

    metrics = {
        # Calibration in the large: mean predicted against observed. If these
        # diverge, every probability band and cut-off downstream is wrong,
        # however good the ranking metrics look.
        "mean_predicted": float(p_test.mean()),
        "observed_rate": float(y_test.mean()),
        "roc_auc": roc_auc_score(y_test, p_test),
        "pr_auc": average_precision_score(y_test, p_test),
        "brier": brier_score_loss(y_test, p_test),
        "precision": precision_score(y_test, y_hat, zero_division=0),
        "recall": recall_score(y_test, y_hat, zero_division=0),
        "f1": f1_score(y_test, y_hat, zero_division=0),
        "flagged_share": float(y_hat.mean()),
        "lift": (precision_score(y_test, y_hat, zero_division=0) / y_test.mean()
                 if y_test.mean() > 0 else np.nan),
    }

    # ---- explanation --------------------------------------------------------
    feature_names = list(model.named_steps["prep"].get_feature_names_out())
    feature_names = [n.split("__", 1)[-1] for n in feature_names]

    coefs = model.named_steps["clf"].coef_[0]
    coefficients = pd.DataFrame({
        "feature": feature_names,
        "label": [label(n) for n in feature_names],
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    })
    coefficients["abs"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("abs", ascending=False, ignore_index=True)

    # Permutation importance is O(features x repeats x rows); on a book this
    # size a capped subsample gives the same ordering for a fraction of the time.
    perm_n = min(len(X_test), 6000)
    X_perm = X_test.sample(perm_n, random_state=RANDOM_STATE)
    perm = permutation_importance(
        model, X_perm, y_test.loc[X_perm.index], n_repeats=5,
        random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    importances = pd.DataFrame({
        "feature": X.columns,
        "label": [label(c) for c in X.columns],
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False, ignore_index=True)

    return ModelResult(
        target=target,
        display_name=display_name,
        estimator=model,
        metrics=metrics,
        comparison=comparison,
        importances=importances,
        coefficients=coefficients,
        calibration=_calibration_table(y_test.to_numpy(), p_test),
        confusion=confusion_matrix(y_test, y_hat),
        threshold=threshold,
        base_rate=float(y.mean()),
        n_train=len(X_train),
        n_test=len(X_test),
        feature_names=feature_names,
    )


def score(result: ModelResult, features: pd.DataFrame) -> np.ndarray:
    """Apply a fitted model to a scoring snapshot."""
    numeric, categorical = model_features(result.target)
    numeric = [c for c in numeric if c in features.columns]
    categorical = [c for c in categorical if c in features.columns]
    return result.estimator.predict_proba(prepare(features, numeric, categorical))[:, 1]


def top_reasons(
    result: ModelResult, row: pd.Series, population: pd.DataFrame, n: int = 4
) -> list[dict]:
    """Why this customer scores as they do.

    The contribution of a feature is its standardised coefficient times how far
    this customer sits from the population average on it. That is the actual
    arithmetic of the model's decision for this row -- readable, additive, and
    checkable by hand, which is the whole reason for keeping the model linear.
    """
    numeric, _ = model_features(result.target)
    numeric = [c for c in numeric if c in population.columns]

    coefs = result.coefficients.set_index("feature")["coefficient"]
    means = population[numeric].mean()
    stds = population[numeric].std().replace(0, np.nan)

    contributions = []
    for feat in numeric:
        if feat not in coefs.index or pd.isna(row.get(feat)):
            continue
        z = (row[feat] - means[feat]) / stds[feat]
        if pd.isna(z):
            continue
        contributions.append({
            "feature": feat,
            "label": label(feat),
            "value": row[feat],
            "population_mean": means[feat],
            "z": float(z),
            "contribution": float(coefs[feat] * z),
        })

    contributions.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return contributions[:n]
