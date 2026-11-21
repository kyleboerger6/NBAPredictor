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
