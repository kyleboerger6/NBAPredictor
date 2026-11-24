"""Checkpoint 4: home-team win classification and test-set predictions.
Choose the best algorithm by validation ROC AUC; refit on train + validation; test once.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src import config
from src.analysis_utils import models, evaluate, save_table, save_plot, plot_confusion, dump_json
from src.split import Splits


def run(s: Splits) -> dict:
    rows = []
    print("[4] Comparing classifiers on validation data...")
    for name, estimator in models().items():
        estimator.fit(s.X_train, s.y_train)
        m = evaluate(s.y_val, estimator.predict_proba(s.X_val)[:, 1])
        rows.append({"model": name, "split": "validation", **m})
        print(f"    {name:14s} val ROC-AUC={m['roc_auc']:.3f}; accuracy={m['accuracy']:.3f}")
    v = pd.DataFrame(rows)
    best = str(v.sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]["model"])
    print(f"    Selection on validation: {best}")
    # Validation is never looked at again to choose a model after this stage.
    all_X, all_y = s.train_val()
    final_models = {}
    test_rows = []
    predictions = s.meta_test.reset_index(drop=True).copy()
    predictions["actual_home_win"] = s.y_test.to_numpy()
    predictions["elo_prob_home"] = s.X_test["elo_prob_home"].to_numpy()
    base = np.ones(len(s.y_test), dtype=float)
    test_rows.append({"model": "always_home", "split": "test", **evaluate(s.y_test, base)})
    predictions["prob_always_home"] = base
    for name, model in models().items():
        model.fit(all_X, all_y)
        final_models[name] = model
        prob = model.predict_proba(s.X_test)[:, 1]
        predictions[f"prob_{name}"] = prob
        test_rows.append({"model": name, "split": "test", **evaluate(s.y_test, prob)})
    save_table(pd.concat([v, pd.DataFrame(test_rows)], ignore_index=True), "04_classification_metrics.csv")
    save_table(predictions, "04_test_predictions.csv")
    plot_confusion(s.y_test, predictions[f"prob_{best}"], "figures/04_confusion.png", f"{best} test predictions")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    a = pd.DataFrame(test_rows)
    ax.bar(a["model"], a["accuracy"])
    ax.set(title="Historical NBA outcomes: held-out test accuracy", ylabel="Accuracy", ylim=(0, 1))
    plt.xticks(rotation=20, ha="right")
    save_plot("figures/04_accuracy.png")
    dump_json({"chosen_by_validation": best, "train_games": len(s.X_train),
               "validation_games": len(s.X_val), "test_games": len(s.X_test)}, "04_selection.json")
    return {"best": best, "models": final_models, "predictions": predictions, "metrics": pd.DataFrame(test_rows)}
