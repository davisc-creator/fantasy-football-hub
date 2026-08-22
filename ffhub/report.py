"""Build the dashboard (out/hub.html) and the weekly text report."""
from __future__ import annotations
import json, datetime as dt
import numpy as np, pandas as pd
from .config import LEAGUES, CACHE, OUT, ROOT, POSITIONS
from .draft import Engine
from .season import Season

def _recs(df: pd.DataFrame, cols=None):
    df = df if cols is None else df[[c for c in cols if c in df]]
    out = []
    for r in df.to_dict("records"):
        out.append({k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)) for k, v in r.items()})
    return out

def league_payload(key: str, with_season=True) -> dict:
    lg = LEAGUES[key]
    pool = pd.read_parquet(CACHE / f"proj_{key}.parquet")
    pool = pool[pool.pos.isin(POSITIONS)]
    eng = Engine(lg, pool)
    tiers = eng.tiers().set_index("espn_id").tier
    cheat = eng.pool.copy(); cheat["tier"] = cheat.espn_id.map(tiers); cheat["vorp"] = eng.vorp
    cheat = cheat.sort_values("vorp", ascending=False).head(260)
    d = {"key": key, "name": lg.name, "platform": lg.platform, "teams": lg.teams, "draft_date": lg.draft_date, "my_team": lg.my_team_name,
         "slots": lg.starter_counts, "repl": eng.valuer.repl, "base": eng.base,
         "cheat": _recs(cheat, ["espn_id", "name", "pos", "nfl", "bye", "tier", "proj", "floor", "ceiling", "games", "vorp", "adp", "fp_ecr", "adp_c", "espn_proj_s", "ml_pts", "injury", "pos_rank", "age"])}
    if with_season:
        try:
            s = Season(lg)
            if s.rosters and any(s.rosters.values()):
                d["week"] = s.week
                d["power"] = _recs(s.power())
                d["lineup"] = _recs(s.lineup())
                d["waivers"] = _recs(s.waivers(20))
                d["drops"] = _recs(s.drop_list())
                d["trades"] = _recs(s.trades(12))
                d["my_team_id"] = s.my
                d["rosters"] = {str(t): _recs(s.team_df(t).sort_values("proj", ascending=False), ["name", "pos", "nfl", "bye", "proj", "injury"]) for t in s.rosters}
                d["names"] = {str(k): v for k, v in s.names.items()}
        except Exception as ex:
            d["season_error"] = str(ex)
    return d

def build(keys=None) -> dict:
    keys = keys or list(LEAGUES)
    payload = {"built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "leagues": {k: league_payload(k) for k in keys}}
    try:
        bts = [pd.read_csv(f) for f in CACHE.glob("backtest_*_espn10.csv")]
        bt = pd.concat(bts)
        bt["mode"] = bt["mode"].map({"engine": "Hub draft engine", "adp_bot": "Draft by ADP (typical league-mate)", "espn_greedy": "Draft by ESPN projections"})
        payload["backtest"] = _recs(bt.groupby("mode").agg(mean_rank=("rank", "mean"), mean_score=("score", "mean"), top3=("rank", lambda x: (x <= 3).mean()), n=("rank", "size")).reset_index().sort_values("mean_rank"))
        payload["backtest_years"] = sorted(bt.year.unique().tolist())
    except Exception: pass
    tpl = (ROOT / "hub_template.html").read_text()
    html = tpl.replace("/*__DATA__*/", "const DATA=" + json.dumps(payload, default=float) + ";")
    (OUT / "hub.html").write_text(html)
    (OUT / "hub_data.json").write_text(json.dumps(payload, default=float))
    docs = ROOT / "docs"; docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(html)          # GitHub Pages copy
    return payload

def weekly_text(key: str) -> str:
    d = league_payload(key)
    L = [f"{d['name']} — week {d.get('week','?')} report ({dt.date.today()})", ""]
    if "lineup" in d:
        L.append("START THIS LINEUP"); L += [f"  {r['slot']:<5} {r['name']:<24} {r['pos']:<3} {r['nfl']:<4} {r['wk_pts']:5.1f} {r['injury'] if r['injury'] not in ('ACTIVE','') else ''}" for r in d["lineup"] if r["slot"] != "BN"]
        L.append(""); L.append("BENCH"); L += [f"  {r['name']:<24} {r['pos']:<3} {r['wk_pts']:5.1f}" for r in d["lineup"] if r["slot"] == "BN"]
        L.append(""); L.append("WAIVER TARGETS (add → drop, net rest-of-season gain)")
        L += [f"  {r['name']:<22} {r['pos']:<3} proj {r['proj']:5.0f}  → drop {r['drop']}: {r['net']:+.1f}" for r in d["waivers"][:8]]
        L.append(""); L.append("TRADE IDEAS (both sides gain)")
        L += [f"  {r['partner']}: give {r['give']} for {r['get']}  (me {r['me']:+.0f}, them {r['them']:+.0f})" for r in d["trades"][:6]]
        L.append(""); L.append("POWER RANKINGS")
        L += [f"  {i+1:>2}. {r['team']:<28} {r['value']:.0f}" for i, r in enumerate(d["power"])]
    else:
        L.append("(no rosters yet — draft pending)")
    return "\n".join(L)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "text":
        print(weekly_text(sys.argv[2] if len(sys.argv) > 2 else "yahoo12"))
    else:
        p = build(); print("built out/hub.html", {k: list(v.keys()) for k, v in p["leagues"].items()})
