"""Treinamento, tuning e avaliação dos modelos do Tech Challenge Fase 2.

Uso (a partir da raiz do repositório):
    python src/modeling.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing import (add_binary_target, load_data,
                           split_features_target,
                           train_test_split_stratified)

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_models(scale_pos_weight: float) -> dict:
    """Modelos e grades de hiperparâmetros (otimização por F1)."""
    return {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced",
                                                 max_iter=5000,
                                                 random_state=SEED))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced",
                                   random_state=SEED, n_jobs=-1),
            {"n_estimators": [300, 500], "max_depth": [None, 8, 12],
             "min_samples_leaf": [1, 3]}),
        "XGBoost": (
            XGBClassifier(random_state=SEED, eval_metric="logloss",
                          scale_pos_weight=scale_pos_weight, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [3, 5],
             "learning_rate": [0.05, 0.1]}),
    }


def evaluate(model, x_test, y_test) -> dict:
    """Métricas completas no conjunto de teste."""
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    return {
        "acuracia": round(accuracy_score(y_test, y_pred), 4),
        "precisao": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
    }


def main() -> None:
    df = add_binary_target(load_data())
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split_stratified(x, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    results = {}
    for name, (estimator, grid) in build_models(spw).items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring="f1", n_jobs=-1)
        search.fit(x_train, y_train)
        results[name] = evaluate(search.best_estimator_, x_test, y_test)
        results[name]["f1_cv"] = round(search.best_score_, 4)

    table = pd.DataFrame(results).T
    print(table)
    table.to_csv(RESULTS_DIR / "metricas_modelos.csv", index_label="modelo")


if __name__ == "__main__":
    main()
