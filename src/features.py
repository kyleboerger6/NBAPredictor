"""Checkpoint 2: leakage-safe pre-game features.

Every feature for a game is computed ONLY from games played strictly before it:
  * rolling/season stats use shift(1) before rolling/expanding
  * Elo is recorded before the game's result updates the rating

Usage:  python -m src.features
"""
import numpy as np
import pandas as pd

from src import config

META_COLS = ["GAME_ID", "GAME_DATE_EST", "SEASON", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
TARGET = "HOME_TEAM_WINS"

# Per-game stats (team perspective) that get rolling averages
ROLL_STATS = ["PTS", "PTS_ALLOWED", "MARGIN", "FG_PCT", "FG3_PCT", "FT_PCT", "AST", "REB",
              "OPP_FG_PCT", "OPP_FG3_PCT", "OPP_REB", "WIN"]
SEASON_STATS = ["WIN", "MARGIN"]
STAT_COLS = ["PTS", "FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]


# ---------------------------------------------------------------------------
# 1. Reshape: one row per (team, game)
# ---------------------------------------------------------------------------
def to_team_games(games: pd.DataFrame) -> pd.DataFrame:
    def side(me: str, opp: str, is_home: int) -> pd.DataFrame:
        team_col, opp_col = ("HOME_TEAM_ID", "VISITOR_TEAM_ID") if is_home else ("VISITOR_TEAM_ID", "HOME_TEAM_ID")
        out = pd.DataFrame({
            "GAME_ID": games["GAME_ID"],
            "GAME_DATE_EST": games["GAME_DATE_EST"],
            "SEASON": games["SEASON"],
            "TEAM_ID": games[team_col],
            "OPP_ID": games[opp_col],
            "IS_HOME": is_home,
        })
        for c in STAT_COLS:
            out[c] = games[f"{c}_{me}"]
        out["PTS_ALLOWED"] = games[f"PTS_{opp}"]
        out["OPP_FG_PCT"] = games[f"FG_PCT_{opp}"]
        out["OPP_FG3_PCT"] = games[f"FG3_PCT_{opp}"]
        out["OPP_REB"] = games[f"REB_{opp}"]
        out["MARGIN"] = out["PTS"] - out["PTS_ALLOWED"]
        out["WIN"] = games[TARGET] if is_home else 1 - games[TARGET]
        return out

    tg = pd.concat([side("home", "away", 1), side("away", "home", 0)], ignore_index=True)
    return tg.sort_values(["TEAM_ID", "GAME_DATE_EST", "GAME_ID"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Rolling / season-to-date / rest features (per team)
# ---------------------------------------------------------------------------
def add_team_features(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE_EST", "GAME_ID"]).copy()
    by_team_season = tg.groupby(["TEAM_ID", "SEASON"], sort=False)

    w, mp = config.ROLL_WINDOW, config.ROLL_MIN_PERIODS
    for c in ROLL_STATS:
        # shift(1) => the current game is never in its own window
        tg[f"roll_{c}"] = by_team_season[c].transform(lambda s: s.shift(1).rolling(w, min_periods=mp).mean())
    for c in SEASON_STATS:
        tg[f"szn_{c}"] = by_team_season[c].transform(lambda s: s.shift(1).expanding(min_periods=mp).mean())
    tg["games_played"] = by_team_season.cumcount()

    # Rest days: across seasons, so the season opener gets a large gap -> capped
    rest = tg.groupby("TEAM_ID")["GAME_DATE_EST"].diff().dt.days
    tg["rest_days"] = rest.fillna(config.REST_CAP).clip(upper=config.REST_CAP)
    tg["back_to_back"] = (tg["rest_days"] == 1).astype(int)
    return tg


TEAM_FEATURES = ([f"roll_{c}" for c in ROLL_STATS] + [f"szn_{c}" for c in SEASON_STATS]
                 + ["games_played", "rest_days", "back_to_back"])


# ---------------------------------------------------------------------------
# 3. Elo (pre-game ratings)
# ---------------------------------------------------------------------------
def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Return pre-game Elo for both teams. Ratings update AFTER being recorded."""
    games = games.sort_values(["GAME_DATE_EST", "GAME_ID"])
    ratings: dict[int, float] = {}
    last_season: dict[int, int] = {}
    pre_home, pre_away = [], []

    for gid, season, h, a, hp, ap, hw in zip(
        games["GAME_ID"], games["SEASON"], games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"],
        games["PTS_home"], games["PTS_away"], games[TARGET],
    ):
        for t in (h, a):
            if t not in ratings:
                ratings[t] = config.ELO_START
            elif last_season[t] != season:  # new season: regress toward the mean
                ratings[t] = (config.ELO_SEASON_REGRESS * ratings[t]
                              + (1 - config.ELO_SEASON_REGRESS) * config.ELO_START)
            last_season[t] = season

        rh, ra = ratings[h], ratings[a]
        pre_home.append(rh)
        pre_away.append(ra)

        # Update with this game's result (538-style margin-of-victory multiplier)
        exp_home = 1 / (1 + 10 ** (-(rh + config.ELO_HOME_ADV - ra) / 400))
        mov = abs(hp - ap)
        winner_diff = (rh + config.ELO_HOME_ADV - ra) * (1 if hw else -1)
        mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        delta = config.ELO_K * mult * (hw - exp_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    out = pd.DataFrame({"GAME_ID": games["GAME_ID"].values, "elo_home": pre_home, "elo_away": pre_away})
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["elo_prob_home"] = 1 / (1 + 10 ** (-(out["elo_diff"] + config.ELO_HOME_ADV) / 400))
    return out


# ---------------------------------------------------------------------------
# 4. Back to one row per game: home_*, away_*, diff_*
# ---------------------------------------------------------------------------
def build_game_features(games: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    tg = add_team_features(to_team_games(games))
    keep = ["GAME_ID"] + TEAM_FEATURES
    home = tg[tg["IS_HOME"] == 1][keep].rename(columns={c: f"home_{c}" for c in TEAM_FEATURES})
    away = tg[tg["IS_HOME"] == 0][keep].rename(columns={c: f"away_{c}" for c in TEAM_FEATURES})

    df = games[META_COLS + [TARGET]].merge(home, on="GAME_ID").merge(away, on="GAME_ID")
    for c in TEAM_FEATURES:
        if c != "back_to_back":
            df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]
    df = df.merge(compute_elo(games), on="GAME_ID")

    if dropna:
        before = len(df)
        df = df.dropna(subset=feature_columns(df))
        print(f"Dropped {before - len(df):,} early-season games with < {config.ROLL_MIN_PERIODS} prior games")
    return df.sort_values(["GAME_DATE_EST", "GAME_ID"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and c != TARGET]


# ---------------------------------------------------------------------------
# 5. Sanity check: suspiciously strong features usually mean leakage
# ---------------------------------------------------------------------------
def leakage_report(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    corr = df[feature_columns(df)].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    print("Top |corr| with target:\n" + corr.head(8).round(3).to_string())
    bad = corr[corr > threshold]
    if len(bad):
        print(f"\nWARNING: {len(bad)} feature(s) have |corr| > {threshold} -- check for leakage:\n{bad}")
    else:
        print(f"\nOK: no feature has |corr| > {threshold}")
    return corr


def main() -> pd.DataFrame:
    games = pd.read_parquet(config.GAMES_CLEAN)
    df = build_game_features(games)
    df.to_parquet(config.FEATURES_FILE, index=False)
    print(f"{len(df):,} games x {len(feature_columns(df))} features -> {config.FEATURES_FILE}\n")
    leakage_report(df)
    return df


if __name__ == "__main__":
    main()
