"""Checkpoint 8a: probability calibration and per-season drift (diagnostics only)."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from src.analysis_utils import evaluate, save_plot, save_table
from src.split import Splits


def run(s: Splits, classifier: dict) -> dict:
    print("[8] Calibration & performance by season...")
    p = classifier["predictions"][f"prob_{classifier['best']}"]
    y = s.y_test
    fraction, pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    calibration = save_table(pd.DataFrame({"mean_predicted_probability": pred,
                                           "observed_win_rate": fraction}), "08_calibration.csv")
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.plot(pred, fraction, "o-", label=classifier["best"])
    ax.set(xlabel="Mean predicted home-win probability", ylabel="Observed home-win rate",
           title="Reliability diagram", xlim=(0,1), ylim=(0,1))
    ax.legend()
    save_plot("figures/08_calibration.png")
    season_rows = []
    for season in sorted(s.meta_test["SEASON"].unique()):
        ix = s.meta_test["SEASON"] == season
        m = evaluate(y.loc[ix], p[ix.to_numpy()])
        season_rows.append({"season": season, **m})
    seas = save_table(pd.DataFrame(season_rows), "08_season_metrics.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(seas["season"].astype(str), seas["accuracy"], "o-")
    ax.set(title="Selected model: held-out accuracy by season", xlabel="Season starting year",
           ylabel="Accuracy", ylim=(0, 1))
    save_plot("figures/08_seasons.png")
    return {"calibration": calibration, "seasons": seas}
