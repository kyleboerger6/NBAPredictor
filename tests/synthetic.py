"""Fake games in the exact Kaggle games.csv schema, for testing the pipeline without the real data."""
import numpy as np
import pandas as pd


def make_raw_games(n_teams: int = 10, seasons=range(2003, 2022), games_per_team: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = np.arange(1610612737, 1610612737 + n_teams)
    rows, gid = [], 0
    for season in seasons:
        strength = dict(zip(teams, rng.normal(0, 5, n_teams)))
        day = pd.Timestamp(f"{season}-10-25")
        for _ in range(games_per_team):
            order = rng.permutation(teams)
            for h, a in zip(order[::2], order[1::2]):
                gid += 1
                hp = int(105 + strength[h] + 3 + rng.normal(0, 11))
                ap = int(105 + strength[a] + rng.normal(0, 11))
                if hp == ap:
                    hp += 1
                rows.append({
                    "GAME_DATE_EST": day.strftime("%Y-%m-%d"),
                    "GAME_ID": int(f"2{season % 100:02d}{gid:05d}"),
                    "GAME_STATUS_TEXT": "Final",
                    "HOME_TEAM_ID": h, "VISITOR_TEAM_ID": a, "SEASON": season,
                    "PTS_home": hp, "FG_PCT_home": rng.uniform(.4, .5), "FT_PCT_home": rng.uniform(.7, .85),
                    "FG3_PCT_home": rng.uniform(.3, .4), "AST_home": rng.integers(18, 30), "REB_home": rng.integers(38, 52),
                    "PTS_away": ap, "FG_PCT_away": rng.uniform(.4, .5), "FT_PCT_away": rng.uniform(.7, .85),
                    "FG3_PCT_away": rng.uniform(.3, .4), "AST_away": rng.integers(18, 30), "REB_away": rng.integers(38, 52),
                    "HOME_TEAM_WINS": int(hp > ap),
                })
            day += pd.Timedelta(days=int(rng.integers(1, 3)))
    df = pd.DataFrame(rows)
    # Messiness the real file has: a preseason game, a duplicate, a null row
    pre = df.iloc[[0]].copy(); pre["GAME_ID"] = 10300001
    dup = df.iloc[[5]].copy()
    null = df.iloc[[7]].copy(); null["GAME_ID"] = 20399999; null["PTS_home"] = np.nan
    return pd.concat([df, pre, dup, null], ignore_index=True)
