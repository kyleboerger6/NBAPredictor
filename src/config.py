"""Project-wide settings. Change values here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

GAMES_CSV = RAW_DIR / "games.csv"          # Kaggle "NBA games data" (Nathan Lauga)
TEAMS_CSV = RAW_DIR / "teams.csv"
GAMES_CLEAN = PROCESSED_DIR / "games_clean.parquet"
FEATURES_FILE = PROCESSED_DIR / "features.parquet"

RANDOM_STATE = 42

# --- Game filtering ---------------------------------------------------------
# GAME_ID's leading digit encodes game type: 1=preseason, 2=regular, 4=playoffs, 5=play-in
GAME_TYPES = (2,)            # regular season only; add 4 to include playoffs
MIN_SEASON = 2003            # SEASON = starting year (2003 -> 2003-04 season)
MAX_SEASON = 2021            # 2022-23 is incomplete in the Kaggle dump

# --- Rolling features -------------------------------------------------------
ROLL_WINDOW = 10             # games in the "recent form" window
ROLL_MIN_PERIODS = 3         # fewer prior games than this -> feature is NaN (row dropped)
REST_CAP = 7                 # cap rest days (season openers get this value)

# --- Elo ---------------------------------------------------------------------
ELO_START = 1500
ELO_K = 20
ELO_HOME_ADV = 100
ELO_SEASON_REGRESS = 0.75    # keep 75% of rating, regress 25% toward ELO_START each season

# --- Chronological split (by SEASON) -----------------------------------------
TRAIN_SEASONS = range(2003, 2017)   # 2003-04 .. 2016-17
VAL_SEASONS = range(2017, 2019)     # 2017-18 .. 2018-19
TEST_SEASONS = range(2019, 2022)    # 2019-20 (bubble) .. 2021-22
