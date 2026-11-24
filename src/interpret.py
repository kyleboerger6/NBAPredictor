"""Checkpoint 6: three global importance techniques and local LIME examples.
Permutation is on held-out test for descriptive analysis ONLY, not for choosing the model.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from src import config
from src.analysis_utils import save_plot, save_table
from src.split import Splits


def run(s: Splits, trained_models: dict, *, run_optional: bool = True) -> dict:
    print("[6] Feature importance: Random Forest / permutation / SHAP / LIME...")
    rf = trained_models["random_forest"]
    impurity = save_table(pd.DataFrame({"feature": s.features, "importance": rf.feature_importances_})
                          .sort_values("importance", ascending=False), "06_forest_importance.csv")
    # Keep compute bounded: max 500 games; same fixed set for reproducibility.
    X = s.X_test.sample(min(180, len(s.X_test)), random_state=config.RANDOM_STATE)
    y = s.y_test.loc[X.index]
    perm = permutation_importance(rf, X, y, n_repeats=2, random_state=config.RANDOM_STATE,
                                  scoring="roc_auc", n_jobs=1)
    perm_tbl = save_table(pd.DataFrame({"feature": s.features, "importance": perm.importances_mean,
                                        "std": perm.importances_std}).sort_values("importance", ascending=False),
                          "06_permutation_importance.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    top = perm_tbl.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance"])
    ax.axvline(0, linewidth=.8, color="black")
    ax.set(title="Permutation importance (held-out test, ROC AUC drop)", xlabel="Mean change in ROC AUC")
    save_plot("figures/06_permutation.png")
    status = {"forest_importance": True, "permutation_importance": True,
              "SHAP": False, "LIME": False}
    if run_optional:
        # SHAP can return (games,features,2) for two-class forests; select winning class.
        import shap
        sample = s.X_test.sample(min(80, len(s.X_test)), random_state=config.RANDOM_STATE)
        e = shap.TreeExplainer(rf)
        vals = np.asarray(e.shap_values(sample))
        if vals.ndim == 3: vals = vals[:, :, 1]
        if vals.shape != sample.shape:
            raise RuntimeError(f"Unexpected SHAP dimensions {vals.shape}, expected {sample.shape}")
        sh_tbl = save_table(pd.DataFrame({"feature": s.features,
                                          "mean_abs_shap": np.abs(vals).mean(axis=0)})
                            .sort_values("mean_abs_shap", ascending=False), "06_shap_importance.csv")
        fig, ax = plt.subplots(figsize=(9, 5))
        q = sh_tbl.head(12).iloc[::-1]
        ax.barh(q["feature"], q["mean_abs_shap"])
        ax.set(title="SHAP importance: Random Forest (80 held-out examples)",
               xlabel="Mean absolute SHAP attribution")
        save_plot("figures/06_shap.png")
        status["SHAP"] = True

        from lime.lime_tabular import LimeTabularExplainer
        background = s.X_train.sample(min(850, len(s.X_train)), random_state=config.RANDOM_STATE)
        lime = LimeTabularExplainer(background.to_numpy(), feature_names=s.features,
                                   class_names=["Away wins", "Home wins"],
                                   mode="classification", random_state=config.RANDOM_STATE,
                                   discretize_continuous=True)
        examples=[]
        # Explanation is for local prediction behaviour, NOT a causal effect.
        for ix in np.linspace(0, len(s.X_test)-1, num=min(3, len(s.X_test)), dtype=int):
            row = s.X_test.iloc[ix]
            exp = lime.explain_instance(row.to_numpy(), rf.predict_proba, num_features=8, num_samples=800)
            for statement, local_weight in exp.as_list(label=1):
                examples.append({"GAME_ID": s.meta_test.iloc[ix]["GAME_ID"],
                                 "home_win_probability": float(rf.predict_proba(row.to_frame().T)[0, 1]),
                                 "condition": statement, "local_weight": local_weight})
        save_table(pd.DataFrame(examples), "06_lime_examples.csv")
        status["LIME"] = True
    return {"importance": perm_tbl, "forest": impurity, "status": status}
