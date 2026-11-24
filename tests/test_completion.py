"""Added tests for completion-stage logic; run with pytest."""
import numpy as np
import pandas as pd
from src.analysis_utils import evaluate
from src.imbalance import upset_subset
from src.clustering import TEAM_STATS, side_values


def test_evaluate_matches_simple_hand_counts():
    results = evaluate([0, 0, 1, 1], [.2, .8, .6, .4])
    assert results["accuracy"] == .5
    assert results["precision"] == .5
    assert results["recall"] == .5


def test_upset_uses_underdog_not_away_win():
    X = pd.DataFrame({"elo_prob_home": [.8, .2, .53, .75]})
    y = pd.Series([0, 1, 0, 1])
    xs, ys = upset_subset(X, y)
    assert list(xs.index) == [0, 1, 3]
    # home favorite loses; away favorite loses; home favorite wins
    assert ys.tolist() == [1, 1, 0]


def test_side_rows_keep_home_and_away_separate():
    X = pd.DataFrame({f"home_{c}":[i] for i,c in enumerate(TEAM_STATS)} |
                     {f"away_{c}":[100+i] for i,c in enumerate(TEAM_STATS)})
    assert side_values(X, "home").shape == (1, len(TEAM_STATS))
    assert side_values(X, "away").iloc[0, 0] == 100
