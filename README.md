# NBAPredictor

**Mitigating Imbalance in NBA Predictions via Interpretable Ensemble Models**
CIS 4930 Data Mining project: pre-game NBA outcome prediction with class-imbalance handling (upsets), cross-method feature importance (permutation / SHAP / LIME), and team-archetype clustering.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
Download the Kaggle dataset **"NBA games data"** (`nathanlauga/nba-games`) and place `games.csv` and `teams.csv` in `data/raw/`. Data files are git-ignored. Do not commit them.

## Pipeline
```bash
.venv/bin/python -m src.load       # 1. clean games      -> data/processed/games_clean.parquet
.venv/bin/python -m src.features   # 2. pre-game features -> data/processed/features.parquet
.venv/bin/python -m src.split      # 3. chronological train/val/test summary
.venv/bin/python -m pytest -q      # leakage + pipeline tests (synthetic data, no download needed)
```
All settings (rolling window, Elo parameters, split seasons) live in `src/config.py`.

## Leakage safeguards
- Rolling and season stats use `shift(1)`, so a game never sees its own result.
- Elo is recorded **before** each game updates it.
- Splits are by season (train 2003–16, val 2017–18, test 2019–21), and the scaler is fit on train only.
- `tests/test_pipeline.py` brute-force recomputes features and checks that altering a game's box score leaves that game's features unchanged.

## Status
- [x] 0 Environment  - [x] 1 Load/clean  - [x] 2 Features  - [x] 3 Split
- [ ] 4 Classification  - [ ] 5 Class imbalance  - [ ] 6 Feature importance  - [ ] 7 Clustering  - [ ] 8 Write-up

## Completion package — new checkpoints 4–8 (Nov 2026)
The original `src.load`, `src.features`, `src.split` and tests were preserved.
New files: `src/analysis_utils.py`, `src/classify.py`, `src/imbalance.py`,
`src/interpret.py`, `src/clustering.py`, `src/reliability.py`, `run_all.py`.

**Run every stage with one command (recommended):**

Windows PowerShell, with the environment already activated:
```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python run_all.py
```
Linux/macOS:
```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 run_all.py
```

Prerequisites: `data/raw/games.csv` from the cited Kaggle dataset. `teams.csv` is
optional for the current experiments. No live NBA API, web scraping, or GPU is needed.
`run_all.py` reads the CSV itself and runs all stages even without cached Parquet files.

`python run_all.py --skip-explainers` is intended ONLY for a smoke test when
`lime` or SHAP isn't installed; it does not complete the promised interpretability
experiment. For the final run, use `python run_all.py` without that flag.

### What each output means
| Stage | CSV output | Figure |
|---|---|---|
| 4. Classification | `04_classification_metrics.csv`, `04_test_predictions.csv` | `figures/04_accuracy.png`, `figures/04_confusion.png` |
| 5. Upset imbalance | `05_upset_prevalence.csv`, `05_upset_metrics.csv`, `05_upset_predictions.csv` | `figures/05_upsets.png` |
| 6. Interpretability | `06_forest_importance.csv`, `06_permutation_importance.csv`, `06_shap_importance.csv`, `06_lime_examples.csv` | `figures/06_permutation.png`, `figures/06_shap.png` |
| 7. Team clustering | `07_archetype_profiles.csv`, `07_clustering_comparison.csv` | `figures/07_archetypes.png` |
| 8. Reliability | `08_calibration.csv`, `08_season_metrics.csv` | `figures/08_calibration.png`, `figures/08_seasons.png` |

`results/summary.json` records the measurements and whether the run was complete.

### Important method details (include in your presentation)
* **A matchup prediction is made before tipoff.** Current-game box scores are NEVER predictors. Pre-game rolling stats, season-to-date averages and pre-game Elo are used.
* **Train/validation/test do not mix seasons.** Model choice uses the 2017–18/2018–19 validation seasons. The 2019–20 through 2021–22 seasons form a held-out test set. Test data are never used for model selection or training.
* **Upsets are a separate classification task.** An upset is defined as an Elo underdog (pre-game implied win chance below 35%) beating an Elo favorite (above 65%). The threshold is a declared research decision, not an NBA fact. SMOTE is used *only for training*.
* **Importance is descriptive, not causal.** Forest impurity importance, held-out permutation importance, SHAP and three local LIME explanations need not agree because correlated inputs split credit differently.
* **Team archetype clustering uses pregame data and trains the clustering transform on training years only.** Cluster number is fixed at 3 for simplicity; the project does not claim these are uniquely true NBA playing styles.
* **Calibration and season analysis use the existing held-out probabilities.** A confidence of 0.7 is not automatically well calibrated.
* **No score is guaranteed.** Present negative/no improvements honestly; report the actual measured outcomes and their limitations.

The original checkpoint tests should still pass; the `tests/test_completion.py`
file provides additional tests of the upset label and evaluation functions.

## Git collaboration
Create a branch, add the new Python files/README/tests (not data or `results/*.csv`),
commit and open a pull request. Don't overwrite a teammate's changes to existing modules.
See `FINISH_AND_PRESENT.md` for detailed Windows/Git workflow and a presentation outline.

## Presentation/report drafts
After a successful **full** run, create an editable presentation and an auto-populated PDF draft:
```bash
python scripts/build_deliverables.py
```
Outputs: `deliverables/NBA_Project_Results_Presentation.pptx` and
`deliverables/NBA_Project_Results_Report_DRAFT.pdf`.
These are aids to group review, **not** independently verified final submissions.
