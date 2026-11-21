"""Checkpoint 1: load and clean the raw Kaggle games file.

Usage:  python -m src.load
"""
import pandas as pd

from src import config

STAT_COLS = ["PTS", "FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]
REQUIRED_COLS = (
    ["GAME_DATE_EST", "GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "SEASON", "HOME_TEAM_WINS"]
    + [f"{c}_home" for c in STAT_COLS]
    + [f"{c}_away" for c in STAT_COLS]
)


def game_type(game_id: pd.Series) -> pd.Series:
    """Leading digit of the 8-digit GAME_ID (leading zeros are dropped in the CSV)."""
    return (game_id.astype("int64") // 10_000_000).astype(int)


def clean_games(raw: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLS) - set(raw.columns)
    if missing:
        raise ValueError(f"games.csv is missing columns: {sorted(missing)}")

    df = raw.copy()
    if "GAME_STATUS_TEXT" in df.columns:
        df = df[df["GAME_STATUS_TEXT"].astype(str).str.strip() == "Final"]

    df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
    df = df.drop_duplicates(subset="GAME_ID", keep="first")
    df = df.dropna(subset=REQUIRED_COLS)

    df = df[game_type(df["GAME_ID"]).isin(config.GAME_TYPES)]
    df = df[df["SEASON"].between(config.MIN_SEASON, config.MAX_SEASON)]

    df = df[REQUIRED_COLS].copy()
    df["HOME_TEAM_WINS"] = df["HOME_TEAM_WINS"].astype(int)
    # Tie-break on GAME_ID so ordering is deterministic for same-day games.
    return df.sort_values(["GAME_DATE_EST", "GAME_ID"]).reset_index(drop=True)


def load_teams() -> pd.DataFrame | None:
    if not config.TEAMS_CSV.exists():
        return None
    return pd.read_csv(config.TEAMS_CSV)[["TEAM_ID", "ABBREVIATION"]]


def main() -> pd.DataFrame:
    if not config.GAMES_CSV.exists():
        raise FileNotFoundError(
            f"{config.GAMES_CSV} not found. Download the Kaggle dataset "
            "'nathanlauga/nba-games' and put games.csv (and teams.csv) in data/raw/."
        )
    df = clean_games(pd.read_csv(config.GAMES_CSV))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.GAMES_CLEAN, index=False)

    summary = df.groupby("SEASON").agg(games=("GAME_ID", "size"), home_win_rate=("HOME_TEAM_WINS", "mean"))
    print(summary.round(3).to_string())
    print(f"\nTotal games: {len(df):,}   overall home win rate: {df['HOME_TEAM_WINS'].mean():.3f}")
    print(f"Saved -> {config.GAMES_CLEAN}")
    return df


if __name__ == "__main__":
    main()
