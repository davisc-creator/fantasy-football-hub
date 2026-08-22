"""Draft engine.

- Opponent model: each player has a draft time ~ Normal(adp_c, adp_sd); a sim samples one time
  per player and each opponent takes the earliest-time available player their roster allows.
- Survival: P(available at my next pick) across sims.
- Roster value: expected season points of the optimal weekly lineup with byes, injury
  availability, season-outcome variance (upside has option value), playoff weighting and
  replacement-level fill for empty slots. Fully vectorised over outcome draws.
- Recommendation: for each candidate at the current pick, simulate the rest of the draft
  (me = marginal-roster-value greedy), value the final roster under common random numbers.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from .config import League, REG_WEEKS, FLEX_POS, POSITIONS

rng = np.random.default_rng(11)
PLAYOFF_W = 1.5
MAX_POS_OPP = {"QB": 2, "RB": 7, "WR": 7, "TE": 2, "K": 1, "DST": 1}
MAX_POS_ME = {"QB": 2, "RB": 6, "WR": 6, "TE": 2, "K": 1, "DST": 1}

# --------------------------------------------------------------- roster value
class Valuer:
    def __init__(self, league: League, pool: pd.DataFrame):
        self.lg = league
        self.slots = league.starter_counts
        self.wmult = np.where(np.isin(np.arange(1, REG_WEEKS + 1), league.playoff_weeks), PLAYOFF_W, 1.0)
        drafted = league.teams * league.draft_rounds
        self.repl = {}
        for pos in POSITIONS:
            und = pool[(pool.pos == pos) & (pool.adp_c > drafted)].sort_values("proj", ascending=False)
            sub = pool[pool.pos == pos]
            self.repl[pos] = float(und.proj.iloc[:3].median()) if len(und) >= 3 else (float(sub.proj.quantile(0.3)) if len(sub) else 0.0)
        self.repl_wk = {p: self.repl[p] / 17.0 * 0.85 for p in POSITIONS}
        self.pool = pool
        n = len(pool)
        self.pos = pool.pos.values
        self.pos_mask = {p: (self.pos == p) for p in POSITIONS}
        self.p_play = np.clip(pool.games.fillna(15).values / 17.0, 0.3, 1.0)
        base = (pool.proj.values / 17.0)[:, None] * np.ones((1, REG_WEEKS))
        bye = pool.bye.fillna(0).values.astype(int)
        for i, b in enumerate(bye):
            if 1 <= b <= REG_WEEKS: base[i, b - 1] = 0
        self.wk_base = base                                  # [n x weeks]
        proj = np.maximum(pool.proj.values, 1)
        lo = np.clip(pool.floor.fillna(pool.proj * 0.75).values / proj, 0.2, 1.0)
        hi = np.clip(pool.ceiling.fillna(pool.proj * 1.25).values / proj, 1.0, 3.0)
        self.sigma = (np.log(hi) - np.log(lo)) / 1.68
        self.draws = 0

    def prepare(self, draws: int, seed: int = 3):
        r = np.random.default_rng(seed)
        n = len(self.pool)
        self.mult = np.exp(r.normal(-self.sigma ** 2 / 2, self.sigma, size=(draws, n))).astype(np.float32)
        self.avail = (r.random((draws, n, REG_WEEKS)) < self.p_play[None, :, None]).astype(np.float32)
        self.draws = draws

    def _lineup_batch(self, wk: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """wk: [D x players x weeks] -> [D] season values (optimal weekly lineup, 1 FLEX)."""
        D = wk.shape[0]
        total = np.zeros((D, REG_WEEKS), np.float32)
        flex_cands = []
        for p in POSITIONS:
            n = self.slots[p]
            m = pos == p
            if m.sum() == 0:
                total += n * self.repl_wk[p]
                if p in FLEX_POS: flex_cands.append(np.full((D, REG_WEEKS), self.repl_wk["RB"], np.float32))
                continue
            v = -np.sort(-wk[:, m, :], axis=1)                 # desc along players
            k = min(n, v.shape[1])
            starters = v[:, :k, :]
            # any empty starter slot -> replacement; any starter below replacement -> stream
            total += np.maximum(starters, self.repl_wk[p]).sum(axis=1) + (n - k) * self.repl_wk[p]
            if p in FLEX_POS:
                nxt = v[:, k, :] if v.shape[1] > k else np.full((D, REG_WEEKS), 0, np.float32)
                flex_cands.append(nxt)
        if self.slots["FLEX"] and flex_cands:
            fx = np.max(np.stack(flex_cands, 0), axis=0)
            total += np.maximum(fx, self.repl_wk["RB"])
        return (total * self.wmult[None, :]).sum(axis=1)

    def value_idx(self, idx, draw_ids=None) -> float:
        """Mean value of roster (pool row indices) over the given CRN draws."""
        idx = np.asarray(idx, int)
        if len(idx) == 0: return 0.0
        if draw_ids is None: draw_ids = np.arange(self.draws)
        wk = self.wk_base[idx][None] * self.avail[draw_ids][:, idx, :] * self.mult[draw_ids][:, idx][:, :, None]
        return float(self._lineup_batch(wk, self.pos[idx]).mean())

    def expected(self, idx) -> float:
        """Deterministic value (no outcome variance) — fast inner-loop policy signal."""
        idx = np.asarray(idx, int)
        if len(idx) == 0: return 0.0
        wk = (self.wk_base[idx] * self.p_play[idx][:, None])[None]
        return float(self._lineup_batch(wk, self.pos[idx])[0])

# --------------------------------------------------------------- draft state
def snake_team(pick: int, teams: int) -> int:
    r, i = (pick - 1) // teams, (pick - 1) % teams
    return i + 1 if r % 2 == 0 else teams - i

def my_picks(slot: int, teams: int, rounds: int) -> list[int]:
    return [p for p in range(1, teams * rounds + 1) if snake_team(p, teams) == slot]

@dataclass
class DraftState:
    league: League
    slot: int
    taken: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    @property
    def teams(self): return self.league.teams
    @property
    def rounds(self): return self.league.draft_rounds
    @property
    def current_pick(self): return len(self.order) + 1
    @property
    def on_clock(self): return snake_team(self.current_pick, self.teams)
    def roster_of(self, slot): return [pid for pid, s in self.taken.items() if s == slot]
    def next_my_picks(self, n=3):
        return [p for p in my_picks(self.slot, self.teams, self.rounds) if p > self.current_pick][:n]
    def pick(self, espn_id: int, slot=None):
        slot = slot or self.on_clock
        self.taken[espn_id] = slot; self.order.append((self.current_pick, slot, espn_id))
    def undo(self):
        if self.order:
            _, _, pid = self.order.pop(); self.taken.pop(pid, None)

def counts_of(pos_list) -> dict:
    c = {p: 0 for p in POSITIONS}
    for p in pos_list: c[p] += 1
    return c

# --------------------------------------------------------------- engine
class Engine:
    def __init__(self, league: League, pool: pd.DataFrame):
        self.lg = league
        pool = pool[pool.pos.isin(POSITIONS)].sort_values("proj", ascending=False).reset_index(drop=True)
        self.pool = pool
        self.valuer = Valuer(league, pool)
        self.ids = pool.espn_id.values
        self.id2i = {int(pid): i for i, pid in enumerate(self.ids)}
        self.pos = pool.pos.values
        self.proj = pool.proj.values
        self.adp = pool.adp_c.values.astype(float)
        self.sd = pool.adp_sd.values.astype(float)
        self.n = len(pool)
        self.total_picks = league.teams * league.draft_rounds
        st = league.starter_counts
        share = {"RB": 0.5, "WR": 0.4, "TE": 0.1}
        self.base = {}
        for p in POSITIONS:
            k = int(round(league.teams * (st[p] + st["FLEX"] * share.get(p, 0)) * (1.2 if p in ("RB", "WR") else 1.0)))
            arr = np.sort(self.proj[self.pos == p])[::-1]
            self.base[p] = float(arr[min(k, len(arr) - 1)]) if len(arr) else 0.0
        self.vorp = self.proj - np.array([self.base[p] for p in self.pos])

    def sample_times(self, sims: int) -> np.ndarray:
        return rng.normal(self.adp[None, :], self.sd[None, :], size=(sims, self.n))

    def _allowed(self, counts, pick, pos, me=False) -> bool:
        rnd = (pick - 1) // self.lg.teams + 1
        last = self.lg.draft_rounds
        cap = MAX_POS_ME if me else MAX_POS_OPP
        if counts[pos] >= cap[pos]: return False
        if pos in ("K", "DST") and rnd < last - 3: return False
        if not me:
            if pos == "QB" and counts["QB"] >= 1 and rnd < 9: return False
            if pos == "TE" and counts["TE"] >= 1 and rnd < 10: return False
        left = last - rnd + 1
        empty = [p for p in ("K", "DST") if counts[p] == 0 and self.lg.starter_counts[p] > 0]
        if left <= len(empty) and pos not in empty: return False
        # must be able to fill all starting slots by the end
        need = sum(max(0, self.lg.starter_counts[p] - counts[p]) for p in POSITIONS)
        if need >= left and self.lg.starter_counts[pos] - counts[pos] <= 0: return False
        return True

    def opp_pick(self, times, avail, counts, pick) -> int:
        cand = np.where(avail)[0]
        order = cand[np.argsort(times[cand])]
        for j in order[:40]:
            if self._allowed(counts, pick, self.pos[j]): return j
        return order[0]

    def my_policy_pick(self, avail, roster_idx: list, counts, pick) -> int:
        """Inner-sim policy: best marginal expected roster value among top VORP candidates."""
        cand = np.where(avail)[0]
        cand = cand[np.argsort(-self.vorp[cand])]
        picked, best, bj = 0, -1e9, -1
        base = self.valuer.expected(roster_idx)
        for j in cand:
            if not self._allowed(counts, pick, self.pos[j], me=True): continue
            v = self.valuer.expected(roster_idx + [j]) - base
            # small scarcity nudge: prefer higher VORP on near-ties
            v += 0.02 * self.vorp[j]
            if v > best: best, bj = v, j
            picked += 1
            if picked >= 8: break
        return bj if bj >= 0 else cand[0]

    def simulate_rest(self, state: DraftState, first_pick, times, stop_at=None) -> dict:
        avail = np.ones(self.n, bool)
        rosters = {s: [] for s in range(1, self.lg.teams + 1)}
        for pid, s in state.taken.items():
            if pid in self.id2i:
                avail[self.id2i[pid]] = False; rosters[s].append(self.id2i[pid])
        counts = {s: counts_of(self.pos[r]) for s, r in rosters.items()}
        pick, end, forced = state.current_pick, stop_at or self.total_picks, first_pick
        while pick <= end:
            s = snake_team(pick, self.lg.teams)
            if s == state.slot:
                j = forced if forced is not None else self.my_policy_pick(avail, rosters[s], counts[s], pick)
                forced = None
            else:
                j = self.opp_pick(times, avail, counts[s], pick)
            avail[j] = False; rosters[s].append(j); counts[s][self.pos[j]] += 1
            pick += 1
        return rosters

    def survival(self, state: DraftState, sims: int = 200) -> pd.DataFrame:
        nxt = state.next_my_picks(2)
        df = self.pool[["espn_id", "name", "pos", "nfl", "bye", "proj", "adp_c"]].copy()
        df["surv_next"] = np.nan; df["surv_next2"] = np.nan
        if not nxt: return df
        T = self.sample_times(sims)
        s1 = np.zeros(self.n); s2 = np.zeros(self.n)
        for s in range(sims):
            ros = self.simulate_rest(state, None, T[s], stop_at=nxt[0] - 1)
            gone = np.zeros(self.n, bool); gone[[i for r in ros.values() for i in r]] = True
            s1 += ~gone
            if len(nxt) > 1:
                ros = self.simulate_rest(state, None, T[s], stop_at=nxt[1] - 1)
                gone = np.zeros(self.n, bool); gone[[i for r in ros.values() for i in r]] = True
                s2 += ~gone
        df["surv_next"] = s1 / sims
        df["surv_next2"] = s2 / sims if len(nxt) > 1 else np.nan
        return df

    def candidates(self, state: DraftState, n_cand=12) -> list[int]:
        avail = np.ones(self.n, bool)
        for pid in state.taken:
            if pid in self.id2i: avail[self.id2i[pid]] = False
        counts = counts_of(self.pos[[self.id2i[p] for p in state.roster_of(state.slot) if p in self.id2i]])
        pick = state.current_pick
        idx = np.where(avail)[0]
        cand = []
        for j in idx[np.argsort(-self.vorp[idx])]:
            if self._allowed(counts, pick, self.pos[j], me=True): cand.append(int(j))
            if len(cand) >= n_cand: break
        for p in POSITIONS:
            pj = idx[self.pos[idx] == p]
            if len(pj):
                j = int(pj[np.argmax(self.proj[pj])])
                if j not in cand and self._allowed(counts, pick, p, me=True): cand.append(j)
        return cand

    # ---- parallel helpers -------------------------------------------------------
    def _eval_candidate(self, state: DraftState, j: int, T: np.ndarray, draws_per_sim: int) -> list:
        vals = []
        for s in range(len(T)):
            ros = self.simulate_rest(state, j, T[s])
            vals.append(self.valuer.value_idx(ros[state.slot], np.arange(s * draws_per_sim, (s + 1) * draws_per_sim)))
        return vals

    def recommend(self, state: DraftState, n_cand=12, sims=40, draws_per_sim=4, surv_sims=150, workers: int | None = None) -> pd.DataFrame:
        cand = self.candidates(state, n_cand)
        if not cand:
            avail = np.ones(self.n, bool)
            for pid in state.taken:
                if pid in self.id2i: avail[self.id2i[pid]] = False
            idx = np.where(avail)[0]; cand = [int(idx[np.argmax(self.vorp[idx])])]
        T = self.sample_times(sims)
        self.valuer.prepare(sims * draws_per_sim)
        surv = self.survival(state, sims=surv_sims).set_index("espn_id")
        rows = []
        import os
        workers = workers if workers is not None else max(1, min(len(cand), (os.cpu_count() or 4) - 2))
        if workers > 1 and len(cand) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(self, state, T, draws_per_sim)) as ex:
                all_vals = list(ex.map(_worker_eval, cand))
        else:
            all_vals = [self._eval_candidate(state, j, T, draws_per_sim) for j in cand]
        for j, vals in zip(cand, all_vals):
            pid = int(self.ids[j])
            rows.append(dict(espn_id=pid, name=self.pool["name"].iloc[j], pos=self.pos[j], nfl=self.pool.nfl.iloc[j],
                             bye=self.pool.bye.iloc[j], proj=self.proj[j], vorp=self.vorp[j], adp=self.adp[j],
                             ev=float(np.mean(vals)), ev_se=float(np.std(vals) / np.sqrt(len(vals))),
                             surv_next=surv.loc[pid, "surv_next"], surv_next2=surv.loc[pid, "surv_next2"]))
        df = pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)
        df["ev_gain"] = df.ev - df.ev.max()
        return df

    def tiers(self, gap_frac=0.045) -> pd.DataFrame:   # noqa: E301
        out = self.pool[["espn_id", "name", "pos", "proj", "adp_c"]].copy(); out["tier"] = 0
        for p in POSITIONS:
            m = (out.pos == p).values
            pr = out.loc[m, "proj"].values; t, tiers = 1, []
            for i, v in enumerate(pr):
                if i > 0 and (pr[i - 1] - v) > gap_frac * max(pr[0], 1): t += 1
                tiers.append(t)
            out.loc[m, "tier"] = tiers
        return out


# ---- process-pool plumbing (module level so it pickles) ----
_W = {}
def _init_worker(engine, state, T, draws):
    _W["e"], _W["s"], _W["T"], _W["d"] = engine, state, T, draws
def _worker_eval(j):
    return _W["e"]._eval_candidate(_W["s"], j, _W["T"], _W["d"])
