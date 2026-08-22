"""Mock-draft harness: play a full draft with the engine as me, ADP bots as opponents, report."""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
from .config import LEAGUES, CACHE
from .draft import Engine, DraftState, snake_team

def run(league_key="espn10", slot=None, sims=20, verbose=True):
    lg = LEAGUES[league_key]
    pool = pd.read_parquet(CACHE / f"proj_{league_key}.parquet")
    eng = Engine(lg, pool)
    slot = slot or int(np.random.default_rng().integers(1, lg.teams + 1))
    st = DraftState(lg, slot)
    T = eng.sample_times(1)[0]
    t0 = time.time()
    while st.current_pick <= eng.total_picks:
        if st.on_clock == slot:
            rec = eng.recommend(st, n_cand=10, sims=sims)
            top = rec.iloc[0]
            st.pick(int(top.espn_id), slot)
            if verbose:
                r = (st.current_pick - 2) // lg.teams + 1
                alts = ", ".join(f"{x['name']}({x.ev_gain:+.0f})" for _, x in rec.iloc[1:4].iterrows())
                print(f"R{r:>2} P{st.current_pick-1:>3}  {top.pos:<3} {top["name"]:<24} proj {top.proj:5.0f}  adp {top.adp:5.1f}  surv {top.surv_next:.0%}  | alts: {alts}")
        else:
            avail = np.ones(eng.n, bool)
            for pid in st.taken: avail[eng.id2i[pid]] = False
            s = st.on_clock
            counts = {p: 0 for p in ["QB", "RB", "WR", "TE", "K", "DST"]}
            for pid in st.roster_of(s): counts[eng.pos[eng.id2i[pid]]] += 1
            j = eng.opp_pick(T, avail, counts, st.current_pick)
            st.pick(int(eng.ids[j]), s)
    mine = pool.set_index("espn_id").loc[st.roster_of(slot)]
    vals = {s: eng.valuer.value(pool.set_index("espn_id").loc[st.roster_of(s)], samples=20) for s in range(1, lg.teams + 1)}
    rank = sorted(vals, key=vals.get, reverse=True).index(slot) + 1
    print(f"\nslot {slot}: roster value {vals[slot]:.0f}  -> rank {rank}/{lg.teams}   ({time.time()-t0:.0f}s)")
    print(mine[["name", "pos", "nfl", "bye", "proj", "adp_c"]].round(1).to_string())
    return rank, vals

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "espn10", int(sys.argv[2]) if len(sys.argv) > 2 else None)
