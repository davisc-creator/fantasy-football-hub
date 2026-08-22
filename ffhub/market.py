"""Market projection: position-specific curve mapping ADP (positional rank) -> expected points.
Fit on history; the ML model then learns deviations from this market anchor."""
from __future__ import annotations
import numpy as np, pandas as pd

class MarketCurve:
    def __init__(self):
        self.coef = {}

    @staticmethod
    def _x(rank):
        r = np.log(np.clip(np.asarray(rank, float), 1, 400))
        return np.column_stack([np.ones_like(r), r, r ** 2])

    def fit(self, df: pd.DataFrame):
        d = df[df.adp.notna() & (df.adp < 170)].copy()
        d["pos_adp_rank"] = d.groupby(["season", "pos"]).adp.rank(method="first")
        for pos, g in d.groupby("pos"):
            X = self._x(g.pos_adp_rank); y = g.y_pts.values
            w = 1.0 / np.sqrt(g.pos_adp_rank.values)          # emphasise the top of the board
            self.coef[pos] = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        return self

    def predict(self, pos: pd.Series, pos_adp_rank: pd.Series) -> np.ndarray:
        out = np.full(len(pos), np.nan)
        for p, c in self.coef.items():
            m = (pos == p).values
            if m.any(): out[m] = np.clip(self._x(pos_adp_rank[m]) @ c, 0, None)
        return out

def add_market(df: pd.DataFrame, curve: MarketCurve, adp_col="adp_f") -> pd.DataFrame:
    df = df.copy()
    grp = ["season", "pos"] if "season" in df else ["pos"]
    df["pos_adp_rank"] = df.groupby(grp)[adp_col].rank(method="first")
    df["market"] = curve.predict(df.pos, df.pos_adp_rank)
    return df
