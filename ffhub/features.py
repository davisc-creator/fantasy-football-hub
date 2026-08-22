"""Training-set construction.

Season model rows = one (player, season) with ESPN preseason projection + ADP,
prior-season production, age/experience/draft capital -> target actual points.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from .config import CACHE, SEASON, ESPN_PPR_SCORING
from . import nfl

NUM = ["pass_yd", "pass_td", "pass_int", "rush_att", "rush_yd", "rush_td", "rec", "targets", "rec_yd", "rec_td", "fum_lost"]

def _season_totals(scoring=ESPN_PPR_SCORING) -> pd.DataFrame:
    w = nfl.offense_weekly(scoring)
    ns = nfl.neutral_stats(w)
    w = pd.concat([w[["player_id", "player_display_name", "position", "season", "week", "team", "target_share", "air_yards_share",
                     "wopr", "carries", "targets", "receiving_air_yards", "rushing_epa", "receiving_epa", "passing_epa", "attempts"]],
                  ns, w[["pts"]]], axis=1)
    g = w.groupby(["player_id", "season"])
    agg = g.agg(name=("player_display_name", "last"), pos=("position", "last"), team=("team", "last"),
                games=("week", "nunique"), pts=("pts", "sum"),
                pass_yd=("pass_yd", "sum"), pass_td=("pass_td", "sum"), pass_int=("pass_int", "sum"), pass_att=("attempts", "sum"),
                carries=("carries", "sum"), rush_yd=("rush_yd", "sum"), rush_td=("rush_td", "sum"),
                targets=("targets", "sum"), rec=("rec", "sum"), rec_yd=("rec_yd", "sum"), rec_td=("rec_td", "sum"),
                tgt_share=("target_share", "mean"), ay_share=("air_yards_share", "mean"), wopr=("wopr", "mean"),
                rush_epa=("rushing_epa", "sum"), rec_epa=("receiving_epa", "sum"), pass_epa=("passing_epa", "sum"),
                pts_sd=("pts", "std"), pts_max=("pts", "max")).reset_index()
    agg["ppg"] = agg.pts / agg.games.clip(lower=1)
    # weekly top-12/24 finish rate proxy: share of weeks above position median
    return agg

def season_table(scoring=ESPN_PPR_SCORING) -> pd.DataFrame:
    """Merge ESPN history (proj/adp/actual) with nflverse prior-season features."""
    hist = pd.read_parquet(CACHE / "espn_history_2018_2025.parquet")
    hist = hist[hist.pos.isin(["QB", "RB", "WR", "TE"])].copy()
    ids = nfl.playerids()[["espn_id", "gsis_id", "birthdate", "draft_year", "draft_round", "draft_ovr"]].dropna(subset=["espn_id"])
    ids["espn_id"] = ids.espn_id.astype(int)
    hist = hist.merge(ids, on="espn_id", how="left")
    tot = _season_totals(scoring)
    # actual under *this* scoring (ESPN 'actual' is its own PPR; recompute from nflverse for consistency)
    act = tot[["player_id", "season", "pts", "games", "ppg"]].rename(columns={"player_id": "gsis_id", "pts": "y_pts", "games": "y_games", "ppg": "y_ppg"})
    prev = tot.copy(); prev["season"] = prev.season + 1
    prev = prev.drop(columns=["name", "pos", "team"]).add_prefix("p1_").rename(columns={"p1_player_id": "gsis_id", "p1_season": "season"})
    prev2 = tot.copy(); prev2["season"] = prev2.season + 2
    prev2 = prev2[["player_id", "season", "pts", "games", "ppg"]].add_prefix("p2_").rename(columns={"p2_player_id": "gsis_id", "p2_season": "season"})
    df = hist.merge(act, on=["gsis_id", "season"], how="left").merge(prev, on=["gsis_id", "season"], how="left").merge(prev2, on=["gsis_id", "season"], how="left")
    df["age"] = df.season - pd.to_datetime(df.birthdate, errors="coerce").dt.year
    df["exp"] = df.season - df.draft_year
    df["rookie"] = (df.exp == 0).astype(int)
    df["draft_ovr"] = df.draft_ovr.fillna(300)
    df["adp"] = np.where((df.adp > 0) & (df.adp < 170), df.adp, np.nan)
    # ESPN projection stat components (neutral keys are not applied in history file; parse ids)
    from .espn import STAT_ID
    def comp(s):
        try: d = json.loads(s)
        except Exception: return {}
        return {STAT_ID.get(k, k): v for k, v in d.items()}
    c = df.proj_stats.map(comp)
    for k in ["pass_yd", "pass_td", "rush_att", "rush_yd", "rush_td", "targets", "rec", "rec_yd", "rec_td", "games"]:
        df[f"pj_{k}"] = c.map(lambda d: d.get(k, np.nan))
    # ESPN's projection rescored under our scoring
    df["proj_s"] = c.map(lambda d: nfl.score_espn_stats(d, scoring))
    df["proj_s"] = df.proj_s.where(df.proj_s > 0, df.proj)
    # target: actual points for players who had a projection; missing nflverse actual => 0 (didn't play)
    df = df[df.proj.notna() & (df.season != 2023)].copy()   # 2023 ESPN projections missing from API
    df["proj_rank"] = df.groupby("season").proj_s.rank(ascending=False, method="first")
    # seasons where ADP is unusable (2018/2019/2025) fall back to projection rank
    ok = df.groupby("season").adp.transform(lambda s: s.notna().mean() > 0.5)
    df["adp_f"] = np.where(ok & df.adp.notna(), df.adp, df.proj_rank)
    df["y_pts"] = df.y_pts.fillna(0.0); df["y_games"] = df.y_games.fillna(0)
    df["y_ppg"] = df.y_ppg.fillna(0.0)
    return df

FEATURES = ["proj_s", "adp_f", "proj_rank", "age", "exp", "rookie", "draft_ovr",
            "pj_pass_yd", "pj_pass_td", "pj_rush_att", "pj_rush_yd", "pj_rush_td", "pj_targets", "pj_rec", "pj_rec_yd", "pj_rec_td", "pj_games",
            "p1_games", "p1_pts", "p1_ppg", "p1_pass_yd", "p1_pass_td", "p1_pass_att", "p1_carries", "p1_rush_yd", "p1_rush_td",
            "p1_targets", "p1_rec", "p1_rec_yd", "p1_rec_td", "p1_tgt_share", "p1_ay_share", "p1_wopr", "p1_rush_epa", "p1_rec_epa", "p1_pass_epa",
            "p1_pts_sd", "p1_pts_max", "p2_pts", "p2_games", "p2_ppg", "pos_code"]

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["pos_code"] = X.pos.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})
    for c in FEATURES:
        if c not in X: X[c] = np.nan
    return X[FEATURES].astype(float)

if __name__ == "__main__":
    df = season_table()
    print(df.shape, df.groupby("season").size().to_dict())
    print(df[["name", "pos", "season", "adp_f", "proj", "proj_s", "y_pts", "y_games", "p1_pts", "age", "exp"]].sample(8, random_state=1).to_string())
    print("match rate gsis:", df.gsis_id.notna().mean().round(3), " p1 present:", df.p1_pts.notna().mean().round(3))
