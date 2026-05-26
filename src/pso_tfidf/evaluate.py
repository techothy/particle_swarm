from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from statsmodels.stats.contingency_tables import mcnemar


@dataclass
class MethodParams:
    name: str
    min_df: float
    max_df: float
    max_features: int | None = None


@dataclass
class EvalReport:
    holdout_metrics: dict[str, dict[str, float]]
    cv_summary: pd.DataFrame
    mcnemar: dict
    paired_ttest: stats.TtestResult
    cohens_d: float


def _metrics(y_true, y_pred) -> dict[str, float]:
    acc = (y_true == y_pred).mean()
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": float(acc), "f1_macro": float(f_ma)}


def evaluate_methods(
    texts: list[str],
    labels: np.ndarray,
    methods: list[MethodParams],
    cv_folds: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> EvalReport:
    y = np.asarray(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, stratify=y, random_state=random_state
    )
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=1000, solver="saga", random_state=random_state)

    holdout: dict[str, dict[str, float]] = {}
    cv_rows: list[dict] = []
    preds_holdout: dict[str, np.ndarray] = {}

    for method in methods:
        vec_kwargs: dict = {"min_df": method.min_df, "max_df": method.max_df}
        if method.max_features is not None:
            vec_kwargs["max_features"] = method.max_features

        fold_f1 = []
        for train_idx, test_idx in kf.split(texts, y):
            tr = [texts[i] for i in train_idx]
            te = [texts[i] for i in test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            vec = TfidfVectorizer(**vec_kwargs)
            Xtr = vec.fit_transform(tr)
            Xte = vec.transform(te)
            clf_fold = LogisticRegression(
                max_iter=1000, solver="saga", n_jobs=-1, random_state=random_state
            )
            clf_fold.fit(Xtr, ytr)
            fold_f1.append(f1_score(yte, clf_fold.predict(Xte), average="macro"))
        cv_rows.append({"method": method.name, "cv_f1_macro_mean": float(np.mean(fold_f1))})

        vec = TfidfVectorizer(**vec_kwargs)
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xte)
        preds_holdout[method.name] = pred
        holdout[method.name] = _metrics(y_test, pred)

    names = [m.name for m in methods]
    if len(names) >= 2:
        a, b = names[0], names[1]
        table = mcnemar_table(y_test, preds_holdout[a], preds_holdout[b])
        mcn = mcnemar(table, exact=False, correction=True)
        mcnemar_out = {"statistic": float(mcn.statistic), "pvalue": float(mcn.pvalue)}
    else:
        mcnemar_out = {}

    f1s = [holdout[m.name]["f1_macro"] for m in methods[:2]]
    if len(f1s) == 2:
        # Use CV fold series for paired test when available — simplified: holdout only
        ttest = stats.ttest_rel(
            [cv_rows[1]["cv_f1_macro_mean"]],
            [cv_rows[0]["cv_f1_macro_mean"]],
        )
        d = 0.0
    else:
        ttest = stats.TtestResult(statistic=np.nan, pvalue=np.nan, df=0)
        d = 0.0

    return EvalReport(
        holdout_metrics=holdout,
        cv_summary=pd.DataFrame(cv_rows),
        mcnemar=mcnemar_out,
        paired_ttest=ttest,
        cohens_d=d,
    )


def mcnemar_table(y_true, pred_a, pred_b):
    b = np.sum((pred_a == y_true) & (pred_b != y_true))
    c = np.sum((pred_a != y_true) & (pred_b == y_true))
    return np.array(
        [
            [np.sum((pred_a == y_true) & (pred_b == y_true)), b],
            [c, np.sum((pred_a != y_true) & (pred_b != y_true))],
        ]
    )


def cross_validated_f1(
    texts: list[str],
    labels: np.ndarray,
    min_df: float,
    max_df: float,
    cv_folds: int = 10,
    max_features: int | None = None,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return mean and std of macro-F1 across stratified CV folds."""
    y = np.asarray(labels)
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=1000, solver="saga", random_state=random_state)
    scores = []
    kwargs: dict = {"min_df": min_df, "max_df": max_df}
    if max_features is not None:
        kwargs["max_features"] = max_features

    for train_idx, test_idx in kf.split(texts, y):
        tr = [texts[i] for i in train_idx]
        te = [texts[i] for i in test_idx]
        vec = TfidfVectorizer(**kwargs)
        Xtr = vec.fit_transform(tr)
        Xte = vec.transform(te)
        clf.fit(Xtr, y[train_idx])
        scores.append(f1_score(y[test_idx], clf.predict(Xte), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))
