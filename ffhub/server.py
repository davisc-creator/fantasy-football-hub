"""Live draft room: local HTTP server + auto-sync of picks from ESPN.

    python run.py draft espn10 [--slot N] [--port 8765]
"""
from __future__ import annotations
import json, threading, time, webbrowser, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np, pandas as pd
from .config import LEAGUES, CACHE, ROOT, OUT
from .draft import Engine, DraftState, snake_team, my_picks
from .espn import ESPN

class Room:
    def __init__(self, league_key: str, slot: int | None):
        self.lg = LEAGUES[league_key]
        self.key = league_key
        self.pool = pd.read_parquet(CACHE / f"proj_{league_key}.parquet")
        self.eng = Engine(self.lg, self.pool)
        self.slot = slot or 1
        self.state = DraftState(self.lg, self.slot)
        self.tiers = self.eng.tiers().set_index("espn_id").tier.to_dict()
        self.rec = None; self.surv = None; self.rec_for_pick = 0
        self.lock = threading.Lock()
        self.computing = False
        self.autosync = self.lg.platform == "espn"
        self.team_names = {}
        self.order_map = None   # ESPN teamId -> draft slot
        self.log = []
        self.sims = 96
        self._load()
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------------- persistence
    def _path(self): return OUT / f"draft_{self.key}.json"
    def _save(self):
        json.dump({"slot": self.slot, "order": self.state.order}, open(self._path(), "w"))
    def _load(self):
        p = self._path()
        if p.exists():
            d = json.load(open(p)); self.slot = d["slot"]; self.state = DraftState(self.lg, self.slot)
            for pick, s, pid in d["order"]: self.state.pick(pid, s)

    # ---------------- ESPN sync
    def sync(self) -> str:
        if self.lg.platform != "espn": return "not espn"
        try:
            e = ESPN(self.lg.league_id)
            s = e.settings()
            self.team_names = {t["id"]: t["name"] for t in s["teams"]}
            order = s["draft"].get("order") or []
            if order: self.order_map = {tid: i + 1 for i, tid in enumerate(order)}
            if self.order_map and self.lg.my_team_id in self.order_map:
                new_slot = self.order_map[self.lg.my_team_id]
                if new_slot != self.slot: self.set_slot(new_slot)
            picks = e.draft_picks()
            dd = s.get("draft_detail", {})
            if len(picks) == 0 or not (picks.espn_id > 0).any(): return f"ESPN: draft {'in progress' if dd.get('inProgress') else 'not started'}, no picks yet"
            picks = picks.sort_values("overall")
            with self.lock:
                cur = [pid for _, _, pid in self.state.order]
                new = [int(x) for x in picks.espn_id.tolist() if int(x) > 0]
                if new[:len(cur)] != cur:       # divergence -> rebuild from ESPN
                    self.state = DraftState(self.lg, self.slot)
                    cur = []
                for pid in new[len(cur):]:
                    slot = snake_team(self.state.current_pick, self.lg.teams)
                    self.state.pick(pid, slot)
                self._save()
            return f"synced {len(new)} picks"
        except Exception as ex:
            return f"sync error: {ex}"

    def set_slot(self, slot: int):
        with self.lock:
            self.slot = slot
            st = DraftState(self.lg, slot)
            for pick, s, pid in self.state.order: st.pick(pid, s)
            self.state = st; self.rec_for_pick = 0; self._save()

    # ---------------- compute loop
    def _loop(self):
        last_sync = 0
        while True:
            try:
                if self.autosync and time.time() - last_sync > 6:
                    msg = self.sync(); last_sync = time.time()
                    self.log = (self.log + [f"{time.strftime('%H:%M:%S')} {msg}"])[-5:]
                if self.state.current_pick != self.rec_for_pick and self.state.current_pick <= self.eng.total_picks:
                    self.computing = True
                    pick = self.state.current_pick
                    st_copy = DraftState(self.lg, self.slot, dict(self.state.taken), list(self.state.order))
                    on_me = st_copy.on_clock == self.slot
                    rec = self.eng.recommend(st_copy, n_cand=12 if on_me else 8, sims=self.sims if on_me else 24, surv_sims=150 if on_me else 80)
                    surv = self.eng.survival(st_copy, sims=120)
                    with self.lock:
                        if self.state.current_pick == pick:
                            self.rec, self.surv, self.rec_for_pick = rec, surv, pick
                    self.computing = False
                time.sleep(0.3)
            except Exception as ex:
                self.computing = False
                self.log = (self.log + [f"compute error: {ex}"])[-5:]
                time.sleep(1)

    # ---------------- API payload
    def payload(self) -> dict:
        st = self.state
        taken = st.taken
        pool = self.pool
        surv = self.surv.set_index("espn_id") if self.surv is not None and len(self.surv) else None
        avail = pool[~pool.espn_id.isin(taken.keys())].copy()
        avail["tier"] = avail.espn_id.map(self.tiers)
        if surv is not None:
            avail["surv_next"] = avail.espn_id.map(surv.surv_next); avail["surv_next2"] = avail.espn_id.map(surv.surv_next2)
        else:
            avail["surv_next"] = np.nan; avail["surv_next2"] = np.nan
        avail["vorp"] = avail.proj - avail.pos.map(self.eng.base)
        cols = ["espn_id", "name", "pos", "nfl", "bye", "proj", "floor", "ceiling", "adp", "fp_ecr", "adp_c", "tier", "surv_next", "surv_next2", "vorp", "injury", "pos_rank"]
        best = avail.sort_values("vorp", ascending=False).head(250)[cols]
        rosters = {}
        for s in range(1, self.lg.teams + 1):
            ids = st.roster_of(s)
            r = pool[pool.espn_id.isin(ids)][["espn_id", "name", "pos", "nfl", "bye", "proj"]]
            rosters[s] = r.to_dict("records")
        rec = self.rec.to_dict("records") if self.rec is not None and self.rec_for_pick == st.current_pick else []
        names = {}
        if self.order_map:
            for tid, s in self.order_map.items(): names[s] = self.team_names.get(tid, f"Team {tid}")
        return {
            "league": {"key": self.key, "name": self.lg.name, "teams": self.lg.teams, "rounds": self.lg.draft_rounds, "slots": self.lg.starter_counts, "bench": self.lg.roster.BENCH},
            "slot": self.slot, "current_pick": st.current_pick, "on_clock": st.on_clock if st.current_pick <= self.eng.total_picks else None,
            "my_next": st.next_my_picks(3), "my_picks": my_picks(self.slot, self.lg.teams, self.lg.draft_rounds),
            "order": [{"pick": p, "slot": s, "espn_id": pid, "name": pool.set_index("espn_id")["name"].get(pid, str(pid))} for p, s, pid in st.order],
            "rec": _clean(rec), "best": _clean(best.to_dict("records")), "rosters": rosters, "team_names": names,
            "computing": self.computing, "rec_ready": self.rec_for_pick == st.current_pick, "autosync": self.autosync, "log": self.log,
            "repl": self.eng.valuer.repl, "base": self.eng.base,
        }

def _clean(rows):
    out = []
    for r in rows:
        out.append({k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)) for k, v in r.items()})
    return out

ROOM: Room | None = None
HTML = (ROOT / "draft.html")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj, default=float).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/state": return self._json(ROOM.payload())
        if u.path in ("/", "/draft.html"):
            b = HTML.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        self._json({"error": "not found"}, 404)
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/pick":
            with ROOM.lock:
                ROOM.state.pick(int(body["espn_id"]), body.get("slot")); ROOM._save()
        elif u.path == "/api/undo":
            with ROOM.lock: ROOM.state.undo(); ROOM._save()
        elif u.path == "/api/slot": ROOM.set_slot(int(body["slot"]))
        elif u.path == "/api/sync": ROOM.log.append(ROOM.sync())
        elif u.path == "/api/autosync": ROOM.autosync = bool(body["on"])
        elif u.path == "/api/reset":
            with ROOM.lock: ROOM.state = DraftState(ROOM.lg, ROOM.slot); ROOM.rec_for_pick = 0; ROOM._save()
        elif u.path == "/api/sims": ROOM.sims = int(body["sims"])
        self._json({"ok": True})

def serve(league_key: str, slot: int | None = None, port: int = 8765, open_browser=True):
    global ROOM
    ROOM = Room(league_key, slot)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(f"Draft room for {ROOM.lg.name} at {url}  (slot {ROOM.slot}, autosync={ROOM.autosync})")
    if open_browser: webbrowser.open(url)
    srv.serve_forever()
