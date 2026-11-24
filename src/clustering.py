"""Checkpoint 7: fit team archetypes on training-era pregame stats only.
Evaluate whether team-cluster identity helps *beyond* existing numeric features.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src import config
from src.analysis_utils import evaluate, save_table, save_plot
from src.split import Splits

TEAM_STATS = ["roll_PTS", "roll_PTS_ALLOWED", "roll_FG_PCT", "roll_REB", "roll_AST", "roll_WIN"]
K = 3


def side_values(X: pd.DataFrame, side: str) -> pd.DataFrame:
    a = X[[f"{side}_{c}" for c in TEAM_STATS]].copy()
    a.columns = TEAM_STATS
    return a


def add_cluster_features(X: pd.DataFrame, scaler: StandardScaler, km: KMeans) -> pd.DataFrame:
    out = X.copy()
    for side in ("home", "away"):
        labels = km.predict(scaler.transform(side_values(X, side)))
        for k in range(K): out[f"{side}_archetype_{k}"] = (labels == k).astype(int)
    return out


def forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=8, max_features="sqrt",
                                  n_jobs=2, random_state=config.RANDOM_STATE)


def run(s: Splits, classifier: dict) -> dict:
    print("[7] Team archetypes: K-Means on pregame team performance...")
    train_team = pd.concat([side_values(s.X_train, "home"), side_values(s.X_train, "away")])
    scaler = StandardScaler().fit(train_team)
    z = scaler.transform(train_team)
    km = KMeans(n_clusters=K, n_init=10, random_state=config.RANDOM_STATE).fit(z)
    prof = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=TEAM_STATS)
    prof.insert(0, "archetype", np.arange(K))
    counts = pd.Series(km.labels_).value_counts().reindex(range(K), fill_value=0)
    prof["training_team_game_instances"] = counts.values
    save_table(prof, "07_archetype_profiles.csv")
    Xtr, Xv, Xte = (add_cluster_features(X, scaler, km)
                    for X in (s.X_train, s.X_val, s.X_test))
    Xall = pd.concat([Xtr, Xv]); yall = pd.concat([s.y_train, s.y_val])
    # Comparable Random Forest baseline, same hyperparameters but without extra cluster columns.
    base = classifier["models"]["random_forest"]
    extended = forest().fit(Xall, yall)
    p_base = base.predict_proba(s.X_test)[:, 1]
    p_extended = extended.predict_proba(Xte)[:, 1]
    m = save_table(pd.DataFrame([{"model": "RF_without_archetypes", **evaluate(s.y_test, p_base)},
                                 {"model": "RF_with_archetypes", **evaluate(s.y_test, p_extended)}]),
                   "07_clustering_comparison.csv")
    save_table(pd.DataFrame({"GAME_ID": s.meta_test["GAME_ID"].to_numpy(),
                             "actual": s.y_test.to_numpy(),
                             "baseline_probability": p_base, "archetype_probability": p_extended}),
               "07_clustering_predictions.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(m.model, m.accuracy)
    ax.set(title="Team archetypes: test accuracy change", ylim=(0, 1), ylabel="Accuracy")
    plt.xticks(rotation=10, ha="right")
    save_plot("figures/07_archetypes.png")
    return {"comparison": m, "profiles": prof}
