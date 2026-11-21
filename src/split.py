"""Checkpoint 3: chronological train / validation / test split (by season).

Usage:  python -m src.split
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import META_COLS, TARGET, feature_columns


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_train: pd.DataFrame   # GAME_ID, date, season, team IDs -- for per-season analysis later
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame

    @property
    def features(self) -> list[str]:
        return list(self.X_train.columns)

    def train_val(self) -> tuple[pd.DataFrame, pd.Series]:
        """Train+val combined, for refitting the final model before the single test evaluation."""
        return pd.concat([self.X_train, self.X_val]), pd.concat([self.y_train, self.y_val])


def get_splits(df: pd.DataFrame | None = None, features: list[str] | None = None) -> Splits:
    if df is None:
        df = pd.read_parquet(config.FEATURES_FILE)
    features = features or feature_columns(df)

    def part(seasons: range):
        sub = df[df["SEASON"].isin(seasons)]
        return sub[features], sub[TARGET], sub[META_COLS]

    Xtr, ytr, mtr = part(config.TRAIN_SEASONS)
    Xva, yva, mva = part(config.VAL_SEASONS)
    Xte, yte, mte = part(config.TEST_SEASONS)

    # Guard: splits must be strictly ordered in time
    assert mtr["GAME_DATE_EST"].max() < mva["GAME_DATE_EST"].min(), "train overlaps val"
    assert mva["GAME_DATE_EST"].max() < mte["GAME_DATE_EST"].min(), "val overlaps test"
    return Splits(Xtr, ytr, Xva, yva, Xte, yte, mtr, mva, mte)


def scale(splits: Splits) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the scaler on TRAIN ONLY, then transform val/test. (Trees don't need this; LogReg does.)"""
    scaler = StandardScaler().fit(splits.X_train)
    tf = lambda X: pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return scaler, tf(splits.X_train), tf(splits.X_val), tf(splits.X_test)


def describe(s: Splits) -> None:
    for name, X, y, m in [("train", s.X_train, s.y_train, s.meta_train),
                          ("val", s.X_val, s.y_val, s.meta_val),
                          ("test", s.X_test, s.y_test, s.meta_test)]:
        print(f"{name:>5}: {len(X):>6,} games  {m['GAME_DATE_EST'].min().date()} -> "
              f"{m['GAME_DATE_EST'].max().date()}  home win rate {y.mean():.3f}")
    print(f"{len(s.features)} features")


if __name__ == "__main__":
    describe(get_splits())
