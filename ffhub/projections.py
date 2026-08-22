"""2026 projection table per league: ML season model + ESPN + FantasyPros ECR + byes."""
from __future__ import annotations
import re, unicodedata
import numpy as np, pandas as pd
from .config import SEASON, League, LEAGUES, CACHE, OUT, REG_WEEKS
from . import nfl, features as F
from .models import SeasonModel, load_market, MARKET_W
from .espn import ESPN

# per-position weight on ML vs ESPN (from CV: QB gains nothing from ML)
ML_WEIGHT = {"QB": 0.35, "RB": 1.0, "WR": 1.0, "TE": 1.0, "K": 0.0, "DST": 0.0}
# widen quantile band (CV coverage 50% at nominal 60%)
BAND_SCALE = 1.2

TEAM_FIX = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}

def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s.lower())
    return re.sub(r"[^a-z]", "", s)

def bye_weeks() -> dict:
    s = nfl.schedules(); s = s[(s.season == SEASON) & (s.game_type == "REG")]
    teams = set(s.home_team) | set(s.away_team)
    byes = {}
    for t in teams:
        played = set(s[(s.home_team == t) | (s.away_team == t)].week)
        missing = sorted(set(range(1, REG_WEEKS + 1)) - played)
        byes[t] = missing[0] if missing else None
    return byes

def fantasypros() -> pd.DataFrame:
    r = nfl.fp_rankings()
    r = r[r.page_type == "redraft-overall"][["player", "pos", "team", "ecr", "sd", "best", "worst", "bye", "id"]].copy()
    r["pos"] = r.pos.str.upper().str.replace("DEF", "DST").str.replace("D/ST", "DST")
    r = r.rename(columns={"id": "fantasypros_id", "ecr": "fp_ecr", "sd": "fp_sd", "best": "fp_best", "worst": "fp_worst"})
    r["key"] = r.player.map(norm_name)
    return r

def build(league: League, espn_league_id: str | None = None, limit: int = 700) -> pd.DataFrame:
    """Projection table for `league`. ESPN data pulled from the ESPN league (or espn10 if Yahoo)."""
    lid = espn_league_id or (league.league_id if league.platform == "espn" else LEAGUES["espn10"].league_id)
    e = ESPN(lid)
    p = e.players(limit)
    p = p[p.pos.isin(["QB", "RB", "WR", "TE", "K", "DST"])].copy()
    p["nfl"] = p.nfl.replace(TEAM_FIX)
    p["espn_proj_s"] = p.espn_proj_stats.map(lambda d: nfl.score_espn_stats(d, league.scoring))
    p["espn_proj_s"] = np.where(p.espn_proj_s > 0, p.espn_proj_s, p.espn_proj.fillna(0))
    # --- ML features (offense only)
    ids = nfl.playerids()[["espn_id", "gsis_id", "birthdate", "draft_year", "draft_round", "draft_ovr", "yahoo_id", "fantasypros_id"]].dropna(subset=["espn_id"])
    ids["espn_id"] = ids.espn_id.astype(int)
    p = p.merge(ids, on="espn_id", how="left")
    tot = F._season_totals(league.scoring)
    prev = tot[tot.season == SEASON - 1].drop(columns=["name", "pos", "team", "season"]).add_prefix("p1_").rename(columns={"p1_player_id": "gsis_id"})
    prev2 = tot[tot.season == SEASON - 2][["player_id", "pts", "games", "ppg"]].add_prefix("p2_").rename(columns={"p2_player_id": "gsis_id"})
    p = p.merge(prev, on="gsis_id", how="left").merge(prev2, on="gsis_id", how="left")
    p["season"] = SEASON
    p["age"] = SEASON - pd.to_datetime(p.birthdate, errors="coerce").dt.year
    p["exp"] = SEASON - p.draft_year
    p["rookie"] = (p.exp == 0).astype(int)
    p["draft_ovr"] = p.draft_ovr.fillna(300)
    p["proj_s"] = p.espn_proj_s
    p["proj_rank"] = p.proj_s.rank(ascending=False, method="first")
    p["adp"] = np.where((p.adp > 0) & (p.adp < 500), p.adp, np.nan)
    p["adp_f"] = p.adp.fillna(p.proj_rank)
    for k in ["pass_yd", "pass_td", "rush_att", "rush_yd", "rush_td", "targets", "rec", "rec_yd", "rec_td", "games"]:
        p[f"pj_{k}"] = p.espn_proj_stats.map(lambda d: d.get(k, np.nan) if isinstance(d, dict) else np.nan)
    m = SeasonModel.load()
    off = p[p.pos.isin(["QB", "RB", "WR", "TE"])]
    pred = m.predict(off)
    p = p.join(pred)
    w = p.pos.map(ML_WEIGHT).fillna(0)
    p["ml_raw"] = p.ml_pts
    p["ml_pts"] = w * p.ml_pts.fillna(p.espn_proj_s) + (1 - w) * p.espn_proj_s     # ML anchored on ESPN (QB mostly ESPN)
    # --- market anchor: ADP/ECR consensus -> expected points via historical curve (needs adp_c, built below)
    fp = fantasypros()
    p["key"] = p.name.map(norm_name)
    p = p.merge(fp.drop(columns=["fantasypros_id", "team", "player", "bye"]), on=["key", "pos"], how="left")
    p["adp_c"] = np.where(p.adp.notna() & p.fp_ecr.notna(), 0.6 * p.adp + 0.4 * p.fp_ecr, p.adp.fillna(p.fp_ecr))
    p["adp_c"] = p.adp_c.fillna(p.proj_rank + 30)
    p["adp_sd"] = p.fp_sd.fillna(np.maximum(3, p.adp_c * 0.12)).clip(lower=2)
    p["pos_adp_rank"] = p.groupby("pos").adp_c.rank(method="first")
    mc = load_market()
    p["market"] = mc.predict(p.pos, p.pos_adp_rank)
    has_m = p.market.notna() & p.pos.isin(["QB", "RB", "WR", "TE"])
    p["proj"] = np.where(has_m, MARKET_W * p.market + (1 - MARKET_W) * p.ml_pts, p.ml_pts)
    lo = (p.ml_p20.fillna(p.proj * 0.8) - p.ml_raw.fillna(p.proj)) * BAND_SCALE
    hi = (p.ml_p80.fillna(p.proj * 1.2) - p.ml_raw.fillna(p.proj)) * BAND_SCALE
    p["floor"] = np.clip(p.proj + lo, 0, None); p["ceiling"] = p.proj + hi
    p["games"] = p.ml_games.fillna(16.5)
    p["ml_ppg"] = p.ml_ppg.fillna(p.proj / 16.5)
    byes = bye_weeks()
    p["bye"] = p.nfl.map(byes)
    cols = ["espn_id", "name", "pos", "nfl", "bye", "injury", "adp", "fp_ecr", "fp_sd", "adp_c", "adp_sd", "espn_rank", "espn_proj_s",
            "ml_pts", "ml_ppg", "market", "proj", "floor", "ceiling", "games", "age", "exp", "on_team", "p1_pts", "p1_games", "espn_weekly", "yahoo_id", "outlook"]
    out = p[cols].sort_values("proj", ascending=False).reset_index(drop=True)
    out["pos_rank"] = out.groupby("pos").proj.rank(ascending=False, method="first").astype(int)
    out["espn_weekly"] = out.espn_weekly.map(lambda d: {str(k): v for k, v in d.items()} if isinstance(d, dict) else {})
    out.to_parquet(CACHE / f"proj_{league.key}.parquet")
    return out

if __name__ == "__main__":
    import sys
    lg = LEAGUES[sys.argv[1] if len(sys.argv) > 1 else "espn10"]
    t = build(lg)
    pd.set_option("display.width", 220)
    show = ["name", "pos", "nfl", "bye", "adp", "fp_ecr", "adp_c", "espn_proj_s", "ml_pts", "ml_ppg", "market", "proj", "floor", "ceiling", "games", "injury"]
    print(t[show].head(40).round(1).to_string())
    for pos in ["QB", "RB", "WR", "TE"]:
        x = t[t.pos == pos].head(3)
        print(pos, x.name.tolist())
    print("byes missing:", t.bye.isna().sum(), "| fp matched:", t.fp_ecr.notna().sum(), "/", len(t))
