"""ONE-COMMAND project runner: python run_all.py

Does not replace teammate's checkpoint modules. Reads games.csv (gitignored), builds pregame
features, runs the 5 project experiments, writes reproducible tables/charts to results/.
"""
from __future__ import annotations
import sys, time, traceback, argparse
import pandas as pd
from src import config
from src.load import clean_games
from src.features import build_game_features, leakage_report
from src.split import get_splits, describe
from src.analysis_utils import dump_json
from src import classify, imbalance, interpret, clustering, reliability


def main(*, skip_explainers: bool = False):
    begin=time.perf_counter()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("NBA PREDICTOR — REPRODUCIBLE END-TO-END RUN")
    print("Python:", sys.version.split()[0])
    if not config.GAMES_CSV.exists():
        raise FileNotFoundError(f"Dataset missing: {config.GAMES_CSV}. See README for Kaggle download instructions.")
    print("[1] Clean historical games...")
    games=clean_games(pd.read_csv(config.GAMES_CSV))
    print("    clean games:",len(games))
    print("[2] Engineer pre-game features (history-only)...")
    features=build_game_features(games)
    leakage_report(features)
    # Also write the teammate's intended parquet outputs when pyarrow is installed.
    try:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        games.to_parquet(config.GAMES_CLEAN, index=False)
        features.to_parquet(config.FEATURES_FILE, index=False)
    except ImportError:
        print("    NOTE: pyarrow not installed; skipping optional parquet cache. Install requirements.txt.")
    print("[3] Chronological train / validation / test splits...")
    splits=get_splits(features)
    describe(splits)
    classification=classify.run(splits)
    upset=imbalance.run(splits)
    importance=interpret.run(splits, classification["models"], run_optional=not skip_explainers)
    archetypes=clustering.run(splits, classification)
    reliability_result=reliability.run(splits, classification)
    seconds=round(time.perf_counter()-begin, 1)
    summary={
        "status": "completed" if not skip_explainers else "partial_explainers_not_run", "seconds": seconds, "dataset_games_after_cleaning":len(games),
        "games_after_feature_generation":len(features),
        "selected_model":classification["best"],
        "classification_test_metrics":classification["metrics"].to_dict(orient="records"),
        "upset_metrics":upset["metrics"].to_dict(orient="records"),
        "upset_prevalence":upset["prevalence"].to_dict(orient="records"),
        "importance_status":importance["status"],
        "top_permutation_features":importance["importance"].head(10).to_dict(orient="records"),
        "clustering":archetypes["comparison"].to_dict(orient="records"),
        "seasonal_metrics":reliability_result["seasons"].to_dict(orient="records"),
    }
    dump_json(summary,"summary.json")
    print(f"\nPIPELINE FINISHED ({summary['status']}) in {seconds:.1f}s. Results -> {config.RESULTS_DIR}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-explainers", action="store_true",
                        help="Development smoke test only. Skips SHAP/LIME; NOT a complete final run.")
    args = parser.parse_args()
    try: main(skip_explainers=args.skip_explainers)
    except Exception as exc:
        print(f"ERROR — pipeline did not finish: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
