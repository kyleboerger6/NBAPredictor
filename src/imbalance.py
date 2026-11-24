"""Checkpoint 5: explicit *underdog upset* task, with imbalance remedies.
Underdog/favorite is established only from pre-game Elo probabilities (threshold .65).
This target is NOT synonymous with an away-team win.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src import config
from src.analysis_utils import evaluate, save_plot, save_table, plot_confusion
from src.split import Splits

FAVORITE_CUTOFF = .65


def upset_subset(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Keep clear Elo favorites; label=1 iff the statistically weaker team won."""
    p = X["elo_prob_home"]
    mask = (p >= FAVORITE_CUTOFF) | (p <= (1 - FAVORITE_CUTOFF))
    fav_home = (p.loc[mask] >= FAVORITE_CUTOFF).astype(int)
    upset = (y.loc[mask].astype(int) != fav_home).astype(int)
    return X.loc[mask], upset


def candidate_models():
    kw = dict(max_iter=2000, random_state=config.RANDOM_STATE)
    return {
        "unweighted": make_pipeline(StandardScaler(), LogisticRegression(**kw)),
        "class_weight_balanced": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", **kw)),
        "SMOTE_train_only": ImbPipeline([("scale", StandardScaler()),
                                          ("smote", SMOTE(random_state=config.RANDOM_STATE, k_neighbors=5)),
                                          ("classifier", LogisticRegression(**kw))]),
    }


def threshold_on_validation(y_val, probs) -> float:
    """Choose an operating point without looking at the test set."""
    candidates = np.arange(.10, .751, .025)
    ranked = [(evaluate(y_val, probs, threshold=t)["f1"], -abs(t-.5), t) for t in candidates]
    return float(max(ranked)[2])


def run(s: Splits) -> dict:
    print("[5] Upset detection: checking prevalence / class weighting / SMOTE...")
    Xtr, ytr = upset_subset(s.X_train, s.y_train)
    Xv, yv = upset_subset(s.X_val, s.y_val)
    Xt, yt = upset_subset(s.X_test, s.y_test)
    prev = pd.DataFrame([{"split": name, "strong_favorite_games": len(y),
                          "upset_count": int(y.sum()), "upset_rate": float(y.mean())}
                         for name, y in [("train", ytr), ("validation", yv), ("test", yt)]])
    save_table(prev, "05_upset_prevalence.csv")
    if min(ytr.value_counts().reindex([0, 1], fill_value=0)) < 6 or len(yv) == 0 or len(yt) == 0:
        raise ValueError("Not enough upset/favorite examples for SMOTE; use a lower FAVORITE_CUTOFF and disclose change.")
    choices = {}
    for name, mod in candidate_models().items():
        mod.fit(Xtr, ytr)
        val_p = mod.predict_proba(Xv)[:, 1]
        choices[name] = threshold_on_validation(yv, val_p)
    # Fit a new model using train+val AFTER choosing operating thresholds on validation.
    Xa, ya = upset_subset(*s.train_val())
    rows = []
    rows.append({"model": "never_predict_upset", "threshold": .5, **evaluate(yt, np.zeros(len(yt)))})
    rows.append({"model": "always_predict_upset", "threshold": .5, **evaluate(yt, np.ones(len(yt)))})
    output = s.meta_test.loc[Xt.index].copy()
    output["actual_upset"] = yt
    for name, mod in candidate_models().items():
        mod.fit(Xa, ya)
        p = mod.predict_proba(Xt)[:, 1]
        threshold = choices[name]
        rows.append({"model": name, "threshold": threshold, **evaluate(yt, p, threshold)})
        output[f"prob_{name}"] = p
        output[f"predict_{name}"] = (p >= threshold).astype(int)
    metrics = save_table(pd.DataFrame(rows), "05_upset_metrics.csv")
    save_table(output.reset_index(drop=True), "05_upset_predictions.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(metrics["model"], metrics["f1"])
    ax.set(title="Underdog upset detection (test): minority-class F1", ylabel="F1 score", ylim=(0, 1))
    plt.xticks(rotation=20, ha="right")
    save_plot("figures/05_upsets.png")
    # Important: selecting via validation is preferable, but we display all test approaches honestly.
    return {"metrics": metrics, "prevalence": prev, "thresholds": choices}
