"""ML models.

SeasonModel: predicts season fantasy points (and 20th/80th pct) from ESPN preseason
projection + ADP + prior production + age/experience. Trained on 2018-2025, validated
leave-one-season-out against the ESPN baseline.
"""
from __future__ import annotations
import pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import MODELS, ESPN_PPR_SCORING
from . import features as F

PARAMS = dict(max_iter=400, learning_rate=0.03, max_depth=4, min_samples_leaf=25, l2_regularization=1.0, random_state=7)

def _train_rows(df: pd.DataFrame) -> pd.DataFrame:
    # relevant pool: players ESPN thought were draftable-ish
    return df[(df.proj_s >= 40) | (df.adp_f < 200)].copy()

class SeasonModel:
    def __init__(self):
        self.mean = None; self.q20 = None; self.q80 = None; self.games = None

    @staticmethod
    def _proj_ppg(d):
        g = d.pj_games.fillna(16).clip(lower=8)
        return d.proj_s / g

    def fit(self, df: pd.DataFrame):
        d = _train_rows(df)
        X = F.featurize(d)
        # (1) skill: per-game residual vs ESPN, on players who actually played >= 4 games
        played = d.y_games >= 4
        y_ppg = (d.y_ppg - self._proj_ppg(d))[played]
        Xp = X[played]
        self.mean = HistGradientBoostingRegressor(**PARAMS).fit(Xp, y_ppg)
        self.q20 = HistGradientBoostingRegressor(loss="quantile", quantile=0.2, **PARAMS).fit(Xp, y_ppg)
        self.q80 = HistGradientBoostingRegressor(loss="quantile", quantile=0.8, **PARAMS).fit(Xp, y_ppg)
        # (2) availability: games played (all rows, incl. zero-game busts)
        self.games = HistGradientBoostingRegressor(**{**PARAMS, "max_iter": 200, "max_depth": 3}).fit(X, d.y_games.clip(0, 17))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = F.featurize(df)
        base = self._proj_ppg(df).values
        out = pd.DataFrame(index=df.index)
        g = np.clip(self.games.predict(X), 4, 17)
        ppg = np.clip(base + self.mean.predict(X), 0, None)
        out["ml_ppg"] = ppg
        out["ml_games"] = g
        out["ml_pts"] = ppg * g
        out["ml_p20"] = np.clip(base + self.q20.predict(X), 0, None) * np.minimum(g, g - 2)   # fewer games in the bad tail
        out["ml_p80"] = np.clip(base + self.q80.predict(X), 0, None) * np.minimum(17, g + 1.5)
        return out

    def save(self, name="season_model.pkl"):
        pickle.dump(self, open(MODELS / name, "wb"))

    @staticmethod
    def load(name="season_model.pkl") -> "SeasonModel":
        return pickle.load(open(MODELS / name, "rb"))


def cross_validate(df: pd.DataFrame, seasons=(2020, 2021, 2022, 2024, 2025)) -> pd.DataFrame:
    rows = []
    for s in seasons:
        tr, te = df[df.season != s], _train_rows(df[df.season == s])
        te = te[te.proj_rank <= 180]  # evaluate on draftable players only
        m = SeasonModel().fit(tr)
        p = m.predict(te)
        for pos in ["ALL", "QB", "RB", "WR", "TE"]:
            mask = slice(None) if pos == "ALL" else (te.pos == pos).values
            y = te.y_pts.values[mask]; e = te.proj_s.values[mask]; mp = p.ml_pts.values[mask]
            rows.append(dict(season=s, pos=pos, n=int(len(y)),
                             mae_espn=np.abs(y - e).mean(), mae_ml=np.abs(y - mp).mean(),
                             corr_espn=np.corrcoef(y, e)[0, 1], corr_ml=np.corrcoef(y, mp)[0, 1],
                             # rank-ordering within position: Spearman
                             sp_espn=pd.Series(y).corr(pd.Series(e), method="spearman"),
                             sp_ml=pd.Series(y).corr(pd.Series(mp), method="spearman"),
                             cover=((y >= p.ml_p20.values[mask]) & (y <= p.ml_p80.values[mask])).mean()))
    return pd.DataFrame(rows)

MARKET_W = 0.7   # weight on the ADP/ECR market curve vs the ML model (validated: best MAE, keeps market rank order)

def train_and_save(df=None):
    from ffhub.models import SeasonModel as SM     # ensure pickle references the module path, not __main__
    from ffhub.market import MarketCurve as MC
    df = df if df is not None else F.season_table(ESPN_PPR_SCORING)
    m = SM().fit(df); m.save()
    pickle.dump(MC().fit(df), open(MODELS / "market.pkl", "wb"))
    return m

def load_market():
    return pickle.load(open(MODELS / "market.pkl", "rb"))

if __name__ == "__main__":
    df = F.season_table(ESPN_PPR_SCORING)
    cv = cross_validate(df)
    pd.set_option("display.width", 200)
    print(cv.round(3).to_string())
    print("\nOVERALL (ALL rows):")
    print(cv[cv.pos == "ALL"][["mae_espn", "mae_ml", "corr_espn", "corr_ml", "sp_espn", "sp_ml", "cover"]].mean().round(3))
    train_and_save(df); print("saved models/season_model.pkl")
