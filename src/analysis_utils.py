"""Shared classification/evaluation utilities for checkpoints 4-8."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src import config


def models() -> dict:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=config.RANDOM_STATE)),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=8,
                                                  max_features="sqrt", n_jobs=2,
                                                  random_state=config.RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.055,
                                 subsample=.85, colsample_bytree=.9, min_child_weight=4,
                                 objective="binary:logistic", eval_metric="logloss",
                                 n_jobs=2, random_state=config.RANDOM_STATE),
    }


def evaluate(y, prob, threshold: float = .5) -> dict:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float).ravel()
    pred = (prob >= threshold).astype(int)
    out = {"n": len(y), "positives": int(y.sum()), "accuracy": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "brier": float(brier_score_loss(y, prob)),
           "log_loss": float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]))}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def save_plot(filename: str) -> None:
    p = config.RESULTS_DIR / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    print("  figure ->", p.relative_to(config.ROOT))


def save_table(data, filename: str) -> pd.DataFrame:
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / filename, index=False)
    print("  table  ->", filename)
    return df


def dump_json(data: dict, filename: str) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.RESULTS_DIR / filename).open("w", encoding="utf8") as f:
        json.dump(data, f, indent=2, default=str)


def plot_confusion(y, prob, filename: str, title: str, threshold: float=.5) -> None:
    cm = confusion_matrix(y, np.asarray(prob) >= threshold, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual", title=title)
    save_plot(filename)
