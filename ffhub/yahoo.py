"""Yahoo Fantasy scraper (no API): reads league roster pages with your browser's session cookie.

.env: YAHOO_CURL=<'Copy as cURL' of any logged-in fantasysports.yahoo.com request>  (cookie is extracted)
      or YAHOO_COOKIE=<raw Cookie header>
Writes data/players.csv (name,pos,team,bye,proj,owner,status); proj = last-season Yahoo pts (informational only).
"""
from __future__ import annotations
import re, time, urllib.request, urllib.error
import pandas as pd
from bs4 import BeautifulSoup
from .config import load_env, DATA, LEAGUES

BASE = "https://football.fantasysports.yahoo.com/f1"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"
POS_FIX = {"DEF": "DEF", "D/ST": "DEF", "W/R/T": "", "BN": "", "IR": ""}

def cookie() -> str:
    env = load_env()
    if env.get("YAHOO_COOKIE"): return env["YAHOO_COOKIE"]
    c = env.get("YAHOO_CURL", "")
    m = re.search(r"-H\s+['\"](?:cookie|Cookie):\s*([^'\"]+)['\"]", c) or re.search(r"-b\s+['\"]([^'\"]+)['\"]", c)
    if not m: raise RuntimeError("No Yahoo cookie: add YAHOO_CURL (Copy as cURL) or YAHOO_COOKIE to .env")
    return m.group(1).strip()

def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Cookie": cookie(), "Accept": "text/html"})
    r = urllib.request.urlopen(req, timeout=30)
    html = r.read().decode("utf-8", "ignore")
    if "login.yahoo.com" in r.geturl() or "Sign in to Yahoo" in html[:5000]:
        raise RuntimeError("Yahoo session expired — re-copy the cURL into .env")
    return html

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("\u2019", "'"))

def parse_roster(html: str) -> tuple[str, list[dict]]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text() if soup.title else ""
    m = re.search(r" - (.+?) \| Fantasy Football", title)
    team_name = m.group(1).strip() if m else ""
    rows = []
    for table in soup.select("table"):
        ths = table.select("thead tr")
        if not ths: continue
        hdr = [th.get_text(" ", strip=True) for th in ths[-1].select("th")]
        def col(*names):
            for n in names:
                if n in hdr: return hdr.index(n)
            return None
        i_bye, i_pts = col("Bye"), col("Fan Pts", "Proj Pts", "Pts")
        for tr in table.select("tbody tr"):
            a = tr.select_one("a[href*='/nfl/players/'], a[href*='/nfl/teams/']")
            if not a: continue
            name = a.get_text(strip=True)
            team, pos = "", ""
            for sp in tr.select("span.Fz-xxs"):
                mt = re.match(r"\s*([A-Za-z]{2,4})\s*-\s*([A-Za-z/,]+)", sp.get_text(" ", strip=True))
                if mt: team, pos = mt.group(1).upper(), mt.group(2).split(",")[0].upper(); break
            st = tr.select_one(".ysf-player-status span")
            status = st.get_text(strip=True) if st else ""
            tds = [td.get_text(" ", strip=True) for td in tr.select("td")]
            bye = tds[i_bye] if i_bye is not None and i_bye < len(tds) and re.fullmatch(r"\d{1,2}", tds[i_bye]) else ""
            proj = tds[i_pts].replace(",", "") if i_pts is not None and i_pts < len(tds) and re.fullmatch(r"[\d,]+\.?\d*", tds[i_pts]) else ""
            rows.append(dict(name=name, pos=pos, team=team, bye=bye, proj=proj, status=status))
    return team_name, rows

def refresh(league_key="yahoo12", sleep=1.0) -> pd.DataFrame:
    lg = LEAGUES[league_key]
    # league slot numbering (draft order) by team name, from data/draft-results.md
    import re as _re
    slots = {_norm(n): i for i, n in enumerate(["Woody dih fit in yo armpit", "Wheel Route Shrivas", "Love Thy Nabers", "McRun McCatch McBlock",
             "Dick'em Down Dicker", "Olave Garden", "The Strib Club", "Sunday Kickoff Jackson", "James Express", "Joe's Big TetTees",
             "I Chase Brown Kids", "Brock Hard"], 1)}
    allrows = []; names = {}
    for tid in range(1, lg.teams + 1):
        html = fetch(f"{BASE}/{lg.league_id}/{tid}")   # default view: bye + last-season pts (projections come from our pipeline)
        tname, rs = parse_roster(html)
        owner = slots.get(_norm(tname), tid)
        names[owner] = tname
        for r in rs: r["owner"] = owner
        allrows += rs; time.sleep(sleep)
        print(f"yahoo team {tid} = {tname!r} -> slot {owner}: {len(rs)} players")
    # free agents: top available by season projection
    for start in (0, 25, 50):
        html = fetch(f"{BASE}/{lg.league_id}/players?status=A&pos=O&cut_type=9&stat1=S_PS_2026&sort=PTS&sdir=1&count={start}")
        _, rs = parse_roster(html)
        for r in rs: r["owner"] = "FA"
        allrows += rs; time.sleep(sleep)
    df = pd.DataFrame(allrows).drop_duplicates("name")
    df["pos"] = df.pos.replace(POS_FIX)
    df.loc[df.pos == "DEF", "name"] = df.loc[df.pos == "DEF", "name"].str.replace(r"\s+D/ST$", "", regex=True)
    out = DATA / "players.csv"
    prev = out.read_bytes() if out.exists() else b""
    (DATA / "players_prev.csv").write_bytes(prev)
    df = df[df.owner.astype(str).str.isdigit() | (df.owner == "FA")]
    df[["name", "pos", "team", "bye", "proj", "owner", "status"]].to_csv(out, index=False)
    import json; (DATA / "yahoo_teams.json").write_text(json.dumps(names))
    print(f"wrote {out} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    refresh()
