"""In-season management: lineups, waivers, trades.

All recommendations are "marginal rest-of-season roster value" using the same valuer as the
draft engine (byes, injuries, outcome variance, playoff weighting), restricted to weeks >= now.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .config import League, LEAGUES, CACHE, REG_WEEKS, POSITIONS, FLEX_POS, DATA
from .draft import Valuer
from .espn import ESPN

def current_week() -> int:
    import datetime as dt
    kickoff = dt.date(2026, 9, 10)          # 2026 NFL week 1 Thursday
    today = dt.date.today()
    if today < kickoff: return 1
    return min(REG_WEEKS, (today - kickoff).days // 7 + 1)

class Season:
    def __init__(self, league: League, week: int | None = None):
        self.lg = league
        self.week = week or current_week()
        self.pool = pd.read_parquet(CACHE / f"proj_{league.key}.parquet")
        self.pool = self.pool[self.pool.pos.isin(POSITIONS)].reset_index(drop=True)
        self.valuer = Valuer(league, self.pool)
        # rest-of-season: zero out weeks already played
        self.valuer.wk_base[:, : self.week - 1] = 0
        self.valuer.prepare(64, seed=9)
        self.id2i = {int(p): i for i, p in enumerate(self.pool.espn_id)}
        self.rosters = {}     # team_id -> list of espn_id
        self.names = {}
        self.my = league.my_team_id
        self.load_rosters()

    # ------------------------------------------------------------ rosters
    def load_rosters(self):
        if self.lg.platform == "espn":
            e = ESPN(self.lg.league_id)
            s = e.settings(); self.names = {t["id"]: t["name"] for t in s["teams"]}
            r = e.rosters()
            self.roster_slots = {}
            if len(r) == 0:
                self.rosters = {t["id"]: [] for t in s["teams"]}
                self.owned = set(); return
            for tid, g in r.groupby("team_id"):
                self.rosters[int(tid)] = [int(x) for x in g.espn_id if int(x) in self.id2i]
            self.roster_slots = {int(x.espn_id): x.slot for x in r.itertuples()}
        else:   # yahoo: from data/players.csv owner column (name match)
            csv = pd.read_csv(DATA / "players.csv")
            from .projections import norm_name
            csv["key"] = csv.name.map(norm_name)
            self.pool["key"] = self.pool.name.map(norm_name)
            # defenses: csv "Cowboys"/pos DEF  vs  ESPN "Cowboys D/ST"/pos DST -> match on team
            from .projections import TEAM_FIX
            dmap = {r.team: r.owner for r in csv[csv.pos.isin(["DEF", "DST"])].itertuples()}
            dmap = {TEAM_FIX.get(k, k): v for k, v in dmap.items()}
            is_d = self.pool.pos == "DST"
            self.pool.loc[is_d, "key"] = "dst_" + self.pool.loc[is_d, "nfl"]
            csv.loc[csv.pos.isin(["DEF", "DST"]), "key"] = "dst_" + csv.loc[csv.pos.isin(["DEF", "DST"]), "team"].map(lambda t: TEAM_FIX.get(t, t))
            if "status" not in csv: csv["status"] = ""
            m = self.pool.merge(csv[["key", "owner", "status"]], on="key", how="left")
            # yahoo injury flags override ESPN's when present
            ymap = {"Q": "QUESTIONABLE", "D": "DOUBTFUL", "O": "OUT", "IR": "INJURY_RESERVE", "SUSP": "SUSPENSION"}
            st = m.set_index("espn_id").status.map(lambda v: ymap.get(str(v).strip().upper(), None))
            upd = st.dropna()
            if len(upd):
                self.pool.loc[self.pool.espn_id.isin(upd.index), "injury"] = self.pool.espn_id.map(upd).fillna(self.pool.injury)
            for owner, g in m.dropna(subset=["owner"]).groupby("owner"):
                if not str(owner).strip().isdigit() or int(owner) == 0: continue
                self.rosters[int(owner)] = [int(x) for x in g.espn_id]
            self.roster_slots = {}
            import re
            try:
                txt = (DATA / "draft-results.md").read_text()
                for ln in txt.splitlines():
                    mm = re.match(r"^(\d+)\. (.+?)(?: · |$)", ln.strip())
                    for part in re.findall(r"(\d+)\. ([^·]+?)(?= · \d+\.|$)", ln.strip()):
                        self.names[int(part[0])] = part[1].strip(" *")
            except Exception: pass
        self.owned = {pid for ids in self.rosters.values() for pid in ids}

    def idx(self, ids): return [self.id2i[i] for i in ids if i in self.id2i]
    def value(self, ids) -> float: return self.valuer.value_idx(self.idx(ids))
    def team_df(self, tid): return self.pool.iloc[self.idx(self.rosters.get(tid, []))]

    # ------------------------------------------------------------ lineup
    def weekly_points(self, row) -> float:
        """This week's projection: ESPN weekly if present, scaled by our season-model ratio."""
        wk = row.espn_weekly or {}
        e = wk.get(str(self.week))
        ratio = float(row.proj) / max(float(row.espn_proj_s), 1) if row.espn_proj_s and row.espn_proj_s > 0 else 1.0
        ratio = float(np.clip(ratio, 0.7, 1.3))
        if e is not None and e > 0: return float(e) * ratio
        if row.bye == self.week: return 0.0
        return float(row.proj) / 17.0 * ratio

    def lineup(self, tid=None) -> pd.DataFrame:
        tid = tid or self.my
        df = self.team_df(tid).copy()
        df["wk_pts"] = [self.weekly_points(r) for r in df.itertuples()]
        df.loc[df.injury.isin(["OUT", "INJURY_RESERVE", "SUSPENSION"]), "wk_pts"] = 0
        df.loc[df.injury.isin(["DOUBTFUL"]), "wk_pts"] *= 0.25
        df.loc[df.injury.isin(["QUESTIONABLE"]), "wk_pts"] *= 0.85
        df = df.sort_values("wk_pts", ascending=False)
        used, rows = set(), []
        for p in ["QB", "RB", "WR", "TE", "K", "DST"]:
            for _ in range(self.lg.starter_counts[p]):
                c = df[(df.pos == p) & ~df.espn_id.isin(used)]
                if len(c): used.add(int(c.espn_id.iloc[0])); rows.append((p, c.iloc[0]))
                else: rows.append((p, None))
        for _ in range(self.lg.starter_counts["FLEX"]):
            c = df[df.pos.isin(FLEX_POS) & ~df.espn_id.isin(used)]
            if len(c): used.add(int(c.espn_id.iloc[0])); rows.append(("FLEX", c.iloc[0]))
            else: rows.append(("FLEX", None))
        for r in df[~df.espn_id.isin(used)].itertuples(): rows.append(("BN", df.loc[r.Index]))
        out = pd.DataFrame([{"slot": s, "name": (r["name"] if r is not None else "— empty —"), "pos": (r.pos if r is not None else s),
                             "nfl": (r.nfl if r is not None else ""), "wk_pts": (r.wk_pts if r is not None else 0), "injury": (r.injury if r is not None else ""),
                             "bye": (r.bye if r is not None else None), "espn_id": (int(r.espn_id) if r is not None else None)} for s, r in rows])
        return out

    # ------------------------------------------------------------ waivers
    def waivers(self, top=25, tid=None) -> pd.DataFrame:
        tid = tid or self.my
        mine = self.rosters.get(tid, [])
        base = self.value(mine)
        fa = self.pool[~self.pool.espn_id.isin(self.owned)]
        fa = fa[(fa.injury != "INJURY_RESERVE")].head(220)
        rows = []
        # drop cost for each of my players
        drop_cost = {pid: base - self.value([x for x in mine if x != pid]) for pid in mine}
        cheapest = sorted(drop_cost, key=drop_cost.get)
        for r in fa.itertuples():
            pid = int(r.espn_id)
            add_v = self.value(mine + [pid]) - base
            # best add/drop pair: maximize (add value - drop cost) over 4 cheapest drops, keeping roster legal
            best_pair, best_net = None, -1e9
            for d in cheapest[:6]:
                net = self.value([x for x in mine if x != d] + [pid]) - base
                if net > best_net: best_net, best_pair = net, d
            rows.append(dict(espn_id=pid, name=r.name, pos=r.pos, nfl=r.nfl, bye=r.bye, proj=r.proj, ceiling=r.ceiling, injury=r.injury,
                             add_value=add_v, drop=self.pool["name"].iloc[self.id2i[best_pair]] if best_pair else None, net=best_net))
        return pd.DataFrame(rows).sort_values("net", ascending=False).head(top).reset_index(drop=True)

    def drop_list(self, tid=None) -> pd.DataFrame:
        tid = tid or self.my
        mine = self.rosters.get(tid, []); base = self.value(mine)
        rows = [dict(espn_id=pid, name=self.pool["name"].iloc[self.id2i[pid]], pos=self.pool.pos.iloc[self.id2i[pid]],
                     proj=self.pool.proj.iloc[self.id2i[pid]], drop_cost=base - self.value([x for x in mine if x != pid])) for pid in mine]
        return pd.DataFrame(rows).sort_values("drop_cost").reset_index(drop=True)

    # ------------------------------------------------------------ trades
    def trades(self, top=15, max_give=2, max_get=2, tid=None) -> pd.DataFrame:
        """Find 1-for-1 and 2-for-1/1-for-2 trades that improve BOTH teams (so they're acceptable)."""
        tid = tid or self.my
        mine = self.rosters.get(tid, []); my_base = self.value(mine)
        my_df = self.pool.iloc[self.idx(mine)]
        # tradeable: my players with low drop cost relative to projection (surplus) + any starter-level player
        give_c = [int(x) for x in my_df.sort_values("proj", ascending=False).espn_id.head(12)]
        rows = []
        for otid, theirs in self.rosters.items():
            if otid == tid or not theirs: continue
            their_base = self.value(theirs)
            their_df = self.pool.iloc[self.idx(theirs)]
            get_c = [int(x) for x in their_df.sort_values("proj", ascending=False).espn_id.head(10)]
            combos = [([g], [r]) for g in give_c for r in get_c]
            if max_give >= 2:
                combos += [([g1, g2], [r]) for i, g1 in enumerate(give_c[:8]) for g2 in give_c[i + 1:8] for r in get_c[:6]]
            if max_get >= 2:
                combos += [([g], [r1, r2]) for g in give_c[:6] for i, r1 in enumerate(get_c[:8]) for r2 in get_c[i + 1:8]]
            for give, get in combos:
                nm = [x for x in mine if x not in give] + get
                nt = [x for x in theirs if x not in get] + give
                if len(nm) > self.lg.roster.size or len(nt) > self.lg.roster.size: continue
                dm = self.value(nm) - my_base
                if dm <= 2: continue
                dt = self.value(nt) - their_base
                if dt <= -3: continue
                rows.append(dict(partner=self.names.get(otid, f"Team {otid}"), give=", ".join(self.pool["name"].iloc[self.id2i[g]] for g in give),
                                 get=", ".join(self.pool["name"].iloc[self.id2i[g]] for g in get), me=dm, them=dt, fairness=min(dm, dt)))
        if not rows: return pd.DataFrame(columns=["partner", "give", "get", "me", "them", "fairness"])
        df = pd.DataFrame(rows).sort_values(["me"], ascending=False).drop_duplicates(["give", "get"])
        df = df[df.groupby("partner").cumcount() < 3]          # at most 3 ideas per partner
        return df.head(top).reset_index(drop=True)

    # ------------------------------------------------------------ league
    def power(self) -> pd.DataFrame:
        rows = []
        for tid, ids in self.rosters.items():
            df = self.team_df(tid)
            rows.append(dict(team_id=tid, team=self.names.get(tid, f"Team {tid}"), value=self.value(ids), n=len(ids),
                             **{p: round(df[df.pos == p].proj.nlargest(self.lg.starter_counts[p] or 1).sum()) for p in ["QB", "RB", "WR", "TE"]}))
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    import sys
    s = Season(LEAGUES[sys.argv[1] if len(sys.argv) > 1 else "yahoo12"])
    pd.set_option("display.width", 220)
    print(f"week {s.week} · teams {len(s.rosters)} · my roster {len(s.rosters.get(s.my, []))}")
    print("\nPOWER"); print(s.power().round(0).to_string())
    print("\nLINEUP"); print(s.lineup().round(1).to_string())
    print("\nWAIVERS"); print(s.waivers(12).round(1).to_string())
    print("\nTRADES"); print(s.trades(10).round(1).to_string())
