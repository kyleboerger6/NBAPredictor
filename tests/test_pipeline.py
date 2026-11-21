"""Run with:  .venv/bin/python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import build_game_features, feature_columns, to_team_games
from src.load import clean_games
from src.split import get_splits, scale
from tests.synthetic import make_raw_games


@pytest.fixture(scope="module")
def games():
    return clean_games(make_raw_games())


@pytest.fixture(scope="module")
def feats(games):
    return build_game_features(games)


# ---------------- Checkpoint 1 ----------------
def test_clean_removes_junk(games):
    assert games["GAME_ID"].is_unique
    assert not games.isna().any().any()
    assert (games["GAME_ID"] // 10_000_000 == 2).all()          # no preseason
    assert games["GAME_DATE_EST"].is_monotonic_increasing


# ---------------- Checkpoint 2 ----------------
def test_rolling_matches_brute_force(games, feats):
    """Recompute features by hand from strictly-earlier games and compare."""
    tg = to_team_games(games)
    rng = np.random.default_rng(1)
    for gid in rng.choice(feats["GAME_ID"], 25, replace=False):
        row = feats[feats["GAME_ID"] == gid].iloc[0]
        team, season, date = row["HOME_TEAM_ID"], row["SEASON"], row["GAME_DATE_EST"]
        prior = tg[(tg["TEAM_ID"] == team) & (tg["SEASON"] == season) & (tg["GAME_DATE_EST"] < date)]
        assert row["home_roll_PTS"] == pytest.approx(prior["PTS"].tail(config.ROLL_WINDOW).mean())
        assert row["home_szn_WIN"] == pytest.approx(prior["WIN"].mean())
        assert row["home_games_played"] == len(prior)


def test_features_unchanged_when_game_result_changes(games, feats):
    """Perturbation test: rewriting a game's box score must not change THAT game's features
    (it should only change later games). Catches leakage in every feature incl. Elo."""
    gid = feats["GAME_ID"].iloc[len(feats) // 2]
    tampered = games.copy()
    idx = tampered["GAME_ID"] == gid
    tampered.loc[idx, ["PTS_home", "PTS_away"]] = [150, 60]
    tampered.loc[idx, "HOME_TEAM_WINS"] = 1
    tampered.loc[idx, ["FG_PCT_home", "REB_home"]] = [0.9, 99]
    new = build_game_features(tampered)

    cols = feature_columns(feats)
    before = feats.set_index("GAME_ID").loc[gid, cols].astype(float)
    after = new.set_index("GAME_ID").loc[gid, cols].astype(float)
    pd.testing.assert_series_equal(before, after)

    # ...but a later game for the same home team must change (proves the test has teeth)
    team, date = feats.set_index("GAME_ID").loc[gid, ["HOME_TEAM_ID", "GAME_DATE_EST"]]
    later = feats[(feats["HOME_TEAM_ID"] == team) & (feats["GAME_DATE_EST"] > date)]["GAME_ID"].iloc[0]
    assert not feats.set_index("GAME_ID").loc[later, cols].astype(float).equals(
        new.set_index("GAME_ID").loc[later, cols].astype(float))


def test_no_nans_and_target_binary(feats):
    assert not feats[feature_columns(feats)].isna().any().any()
    assert set(feats["HOME_TEAM_WINS"].unique()) <= {0, 1}


# ---------------- Checkpoint 3 ----------------
def test_split_is_chronological(feats):
    s = get_splits(feats)
    assert s.meta_train["SEASON"].max() < s.meta_val["SEASON"].min()
    assert s.meta_val["SEASON"].max() < s.meta_test["SEASON"].min()
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == len(feats)


def test_scaler_fit_on_train_only(feats):
    s = get_splits(feats)
    _, Xtr, Xva, _ = scale(s)
    assert np.allclose(Xtr.mean(), 0, atol=1e-8)
    assert not np.allclose(Xva.mean(), 0, atol=1e-8)   # val isn't re-centered on itself
