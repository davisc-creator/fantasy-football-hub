"""nflverse data access + league-neutral scoring."""
from __future__ import annotations
import warnings
import pandas as pd, numpy as np, polars as pl
from .config import CACHE, SEASON

warnings.filterwarnings("ignore")
HIST_FIRST = 2015

def _cached(name: str, loader, refresh=False) -> pd.DataFrame:
    p = CACHE / f"{name}.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    df = loader()
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    df.to_parquet(p)
    return df

def weekly_stats(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached("player_stats_week_2015_2025" if not refresh else "player_stats_week_2015_2025",
                   lambda: n.load_player_stats(seasons=list(range(HIST_FIRST, SEASON)), summary_level="week"), refresh)

def schedules(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached("schedules", lambda: n.load_schedules(seasons=list(range(HIST_FIRST, SEASON + 1))), refresh)

def playerids(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached("ff_playerids", lambda: n.load_ff_playerids(), refresh)

def players(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached("players", lambda: n.load_players(), refresh)

def fp_rankings(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached("ff_rankings_draft", lambda: n.load_ff_rankings(type="draft"), refresh)

def rosters_now(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached(f"rosters_{SEASON}", lambda: n.load_rosters(seasons=[SEASON]), refresh)

def injuries(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached("injuries", lambda: n.load_injuries(seasons=list(range(2016, SEASON + 1))), refresh)

def snap_counts(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached("snap_counts", lambda: n.load_snap_counts(seasons=list(range(2016, SEASON + 1))), refresh)

def depth_charts(refresh=False) -> pd.DataFrame:
    import nflreadpy as n
    return _cached(f"depth_charts_{SEASON}", lambda: n.load_depth_charts(seasons=[SEASON]), refresh)

# ------------------------------------------------------------ scoring
# map nflverse weekly columns -> neutral stat keys
NFLV = {"passing_yards": "pass_yd", "passing_tds": "pass_td", "passing_interceptions": "pass_int",
        "passing_2pt_conversions": "pass_2pt", "rushing_yards": "rush_yd", "rushing_tds": "rush_td",
        "rushing_2pt_conversions": "rush_2pt", "receptions": "rec", "receiving_yards": "rec_yd", "receiving_tds": "rec_td",
        "receiving_2pt_conversions": "rec_2pt", "rushing_40": "rush_40plus", "receiving_40": "rec_40plus",
        "fg_made_40_49": "fg_40_49", "fg_missed": "fg_miss", "pat_made": "xp"}

def neutral_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nflverse weekly rows to the neutral stat vocabulary."""
    out = pd.DataFrame(index=df.index)
    for src, dst in NFLV.items():
        if src in df:
            out[dst] = df[src].fillna(0)
    out["fum_lost"] = (df.get("rushing_fumbles_lost", 0).fillna(0) + df.get("receiving_fumbles_lost", 0).fillna(0)
                       + df.get("sack_fumbles_lost", 0).fillna(0))
    out["fg_0_39"] = df.get("fg_made_0_19", 0).fillna(0) + df.get("fg_made_20_29", 0).fillna(0) + df.get("fg_made_30_39", 0).fillna(0)
    out["fg_50_plus"] = df.get("fg_made_50_59", 0).fillna(0) + df.get("fg_made_60_", 0).fillna(0)
    ry = df.get("rushing_yards", 0).fillna(0)
    out["rush_100_bonus"] = (ry >= 100).astype(float)
    out["rush_125_bonus"] = (ry >= 125).astype(float)
    return out

def score(stats: pd.DataFrame, scoring: dict) -> pd.Series:
    pts = pd.Series(0.0, index=stats.index)
    for k, v in scoring.items():
        if k in stats:
            pts = pts + stats[k].fillna(0) * v
    return pts

def score_espn_stats(stat_dict: dict, scoring: dict) -> float:
    """Score an ESPN projection stat dict (already mapped to neutral keys by espn.py)."""
    if not isinstance(stat_dict, dict):
        return np.nan
    t = 0.0
    for k, v in scoring.items():
        t += float(stat_dict.get(k, 0) or 0) * v
    return t

def offense_weekly(scoring: dict) -> pd.DataFrame:
    """Regular-season weekly rows for QB/RB/WR/TE/K scored under `scoring`."""
    w = weekly_stats()
    w = w[(w.season_type == "REG") & (w.position.isin(["QB", "RB", "WR", "TE", "K"]))].copy()
    w["pts"] = score(neutral_stats(w), scoring).values
    return w
