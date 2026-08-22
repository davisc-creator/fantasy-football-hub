"""Backtest the draft engine on a past season.

Build the preseason pool for `year` exactly as the engine would have seen it (ESPN projection,
ESPN ADP, model trained WITHOUT that year), draft from every slot against ADP bots, then score
every team's roster by ACTUAL weekly points (optimal weekly lineup, replacement fill).
Baselines: ADP bot (what a typical drafter does), ESPN-projection greedy.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
from .config import LEAGUES, CACHE, ESPN_PPR_SCORING, REG_WEEKS, POSITIONS
from . import features as F, nfl
from .models import SeasonModel
from .draft import Engine, DraftState, Valuer, snake_team, counts_of

def pool_for(year: int, scoring=ESPN_PPR_SCORING) -> tuple[pd.DataFrame, SeasonModel]:
    df = F.season_table(scoring)
    m = SeasonModel().fit(df[df.season != year])
    from .market import MarketCurve, add_market
    from .models import MARKET_W
    mc = MarketCurve().fit(df[df.season != year])
    d = df[df.season == year].copy()
    pred = m.predict(d)
    d = d.join(pred)
    w = d.pos.map({"QB": 0.35, "RB": 1, "WR": 1, "TE": 1}).fillna(0)
    ml_raw = d.ml_pts.copy()
    d["ml_pts"] = w * d.ml_pts + (1 - w) * d.proj_s
    d = add_market(d.drop(columns=["season"]).assign(season=year), mc, "adp_f")
    d["proj"] = np.where(d.market.notna(), MARKET_W * d.market + (1 - MARKET_W) * d.ml_pts, d.ml_pts)
    d["floor"] = np.clip(d.proj + (d.ml_p20 - ml_raw) * 1.2, 0, None)
    d["ceiling"] = d.proj + (d.ml_p80 - ml_raw) * 1.2
    d["games"] = d.ml_games
    d["espn_only"] = d.proj_s
    d["adp_c"] = d.adp.fillna(d.proj_rank + 20)
    d["adp_sd"] = np.maximum(3, d.adp_c * 0.12)
    # byes for that season
    s = nfl.schedules(); s = s[(s.season == year) & (s.game_type == "REG")]
    weeks = REG_WEEKS if year >= 2021 else 17
    byes = {}
    for t in set(s.home_team) | set(s.away_team):
        played = set(s[(s.home_team == t) | (s.away_team == t)].week)
        miss = sorted(set(range(1, weeks + 1)) - played); byes[t] = miss[0] if miss else 0
    tot = F._season_totals(scoring)
    team = tot[tot.season == year].set_index("player_id").team
    d["nfl"] = d.gsis_id.map(team)
    d["bye"] = d.nfl.map(byes)
    d = d[d.proj > 5].copy()
    return d[["espn_id", "name", "pos", "nfl", "bye", "proj", "floor", "ceiling", "games", "adp_c", "adp_sd", "espn_only", "gsis_id", "y_pts"]].reset_index(drop=True), m

def actual_weekly(year: int, scoring=ESPN_PPR_SCORING) -> pd.DataFrame:
    w = nfl.offense_weekly(scoring)
    w = w[w.season == year]
    return w.pivot_table(index="player_id", columns="week", values="pts", aggfunc="sum").reindex(columns=range(1, REG_WEEKS + 1)).fillna(0)

def score_roster(roster: pd.DataFrame, act: pd.DataFrame, lg, repl_wk: dict) -> float:
    """Actual season value: optimal weekly lineup from real weekly points, replacement fill."""
    slots = lg.starter_counts
    wmult = np.where(np.isin(np.arange(1, REG_WEEKS + 1), lg.playoff_weeks), 1.5, 1.0)
    pts = np.array([act.loc[g].values if g in act.index else np.zeros(REG_WEEKS) for g in roster.gsis_id])
    pos = roster.pos.values
    total = np.zeros(REG_WEEKS)
    flex = []
    for p in ["QB", "RB", "WR", "TE"]:
        n = slots[p]; m = pos == p
        v = -np.sort(-pts[m], axis=0) if m.sum() else np.zeros((0, REG_WEEKS))
        k = min(n, v.shape[0])
        total += np.maximum(v[:k], repl_wk[p]).sum(axis=0) + (n - k) * repl_wk[p]
        if p in ("RB", "WR", "TE"):
            flex.append(v[k] if v.shape[0] > k else np.zeros(REG_WEEKS))
    total += np.maximum(np.max(np.stack(flex), axis=0), repl_wk["RB"])
    return float((total * wmult).sum())

def run(year=2024, league_key="espn10", slots=None, sims=24, n_cand=8, surv_sims=40, verbose=True, reps=2):
    lg = LEAGUES[league_key]
    pool, _ = pool_for(year)
    pool = pool[pool.pos.isin(["QB", "RB", "WR", "TE"])].reset_index(drop=True)
    lg_noKD = type(lg)(**{**lg.__dict__})
    lg_noKD.roster = type(lg.roster)(**{**lg.roster.__dict__, "K": 0, "DST": 0, "BENCH": lg.roster.BENCH + 2})
    act = actual_weekly(year)
    eng = Engine(lg_noKD, pool)
    # ESPN-greedy pool (same engine machinery, projections = ESPN)
    pool_e = pool.copy(); pool_e["proj"] = pool_e.espn_only; pool_e["floor"] = pool_e.proj * 0.75; pool_e["ceiling"] = pool_e.proj * 1.25
    eng_e = Engine(lg_noKD, pool_e)
    repl_wk = {p: (0.0 if np.isnan(eng.valuer.repl[p]) else eng.valuer.repl[p] / 17 * 0.85) for p in POSITIONS}
    results = []
    slots = slots or list(range(1, lg.teams + 1))
    for rep in range(reps):
      for slot in slots:
        for mode in ["engine", "espn_greedy", "adp_bot"]:
            E = eng_e if mode == "espn_greedy" else eng
            st = DraftState(lg_noKD, slot)
            T = E.sample_times(1)[0]
            t0 = time.time()
            while st.current_pick <= E.total_picks:
                s = st.on_clock
                avail = np.ones(E.n, bool)
                for pid in st.taken: avail[E.id2i[pid]] = False
                counts = counts_of(E.pos[[E.id2i[p] for p in st.roster_of(s)]])
                if s == slot and mode == "engine":
                    rec = E.recommend(st, n_cand=n_cand, sims=sims, surv_sims=surv_sims)
                    j = E.id2i[int(rec.espn_id.iloc[0])]
                elif s == slot and mode == "espn_greedy":
                    j = E.my_policy_pick(avail, [E.id2i[p] for p in st.roster_of(s)], counts, st.current_pick)
                else:
                    j = E.opp_pick(T, avail, counts, st.current_pick)
                st.pick(int(E.ids[j]), s)
            scores = {s: score_roster(pool.set_index("espn_id").loc[st.roster_of(s)].reset_index(), act, lg_noKD, repl_wk) for s in range(1, lg.teams + 1)}
            rank = sorted(scores, key=scores.get, reverse=True).index(slot) + 1
            results.append(dict(year=year, rep=rep, slot=slot, mode=mode, score=scores[slot], rank=rank, league_mean=np.mean(list(scores.values())), secs=time.time() - t0))
            if verbose: print(f"{year} slot {slot:>2} {mode:<12} actual {scores[slot]:6.0f}  rank {rank:>2}/{lg.teams}  (league mean {np.mean(list(scores.values())):.0f})  {time.time()-t0:.0f}s", flush=True)
    r = pd.DataFrame(results)
    r.to_csv(CACHE / f"backtest_{year}_{league_key}.csv", index=False)
    print("\nSUMMARY"); print(r.groupby("mode").agg(mean_rank=("rank", "mean"), mean_score=("score", "mean"), top3=("rank", lambda x: (x <= 3).mean())).round(2))
    return r

if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    run(yr)
