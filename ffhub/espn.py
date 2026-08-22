"""ESPN Fantasy API client (read-only). Needs ESPN_S2 + SWID for private leagues."""
from __future__ import annotations
import json, time, urllib.request, urllib.error
import pandas as pd
from .config import load_env, SEASON, CACHE

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
POS_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
SLOT_ID = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 23: "FLEX", 17: "K", 16: "DST", 20: "BENCH", 21: "IR"}
PRO_TEAM = {0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
            11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
            21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
            33: "BAL", 34: "HOU"}
# ESPN stat ids -> neutral keys
STAT_ID = {"3": "pass_yd", "4": "pass_td", "20": "pass_int", "19": "pass_2pt", "23": "rush_att", "24": "rush_yd", "25": "rush_td",
           "26": "rush_2pt", "42": "rec_yd", "43": "rec_td", "44": "rec_2pt", "53": "rec", "58": "targets", "72": "fum_lost",
           "210": "games", "0": "pass_att", "1": "pass_cmp",
           "74": "fg_50_plus", "77": "fg_40_49", "80": "fg_0_39", "85": "fg_miss", "86": "xp",
           "89": "dst_pa_0", "90": "dst_pa_1_6", "91": "dst_pa_7_13", "92": "dst_pa_14_17", "95": "dst_int", "96": "dst_fr",
           "97": "dst_blk", "98": "dst_safety", "99": "dst_sack", "101": "dst_td", "103": "dst_ret_td", "123": "dst_pa_28_34",
           "124": "dst_pa_35_45", "125": "dst_pa_46_plus"}


class ESPN:
    def __init__(self, league_id: str, season: int = SEASON):
        env = load_env()
        self.league_id, self.season = str(league_id), season
        self.headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        if env.get("ESPN_S2") and env.get("SWID"):
            self.headers["Cookie"] = f"espn_s2={env['ESPN_S2']}; SWID={env['SWID']}"

    def _get(self, url: str, filt: dict | None = None, retries: int = 3):
        h = dict(self.headers)
        if filt:
            h["x-fantasy-filter"] = json.dumps(filt)
        for i in range(retries):
            try:
                return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60))
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise RuntimeError("ESPN auth failed — refresh ESPN_S2/SWID in .env") from e
                time.sleep(1 + i)
        raise RuntimeError(f"ESPN request failed: {url}")

    def league_url(self, season: int | None = None) -> str:
        return f"{BASE}/seasons/{season or self.season}/segments/0/leagues/{self.league_id}"

    # ------------------------------------------------------------ league
    def settings(self) -> dict:
        d = self._get(self.league_url() + "?view=mSettings&view=mTeam&view=mDraftDetail&view=mStatus")
        s = d["settings"]
        teams = [{"id": t["id"], "name": t.get("name") or f"{t.get('location','')} {t.get('nickname','')}".strip(),
                  "owner": (t.get("owners") or [None])[0]} for t in d.get("teams", [])]
        return {
            "name": s["name"], "size": s["size"], "teams": teams,
            "lineup_slots": {SLOT_ID.get(int(k), k): v for k, v in s["rosterSettings"]["lineupSlotCounts"].items() if v},
            "scoring_items": s["scoringSettings"]["scoringItems"],
            "draft": {"type": s["draftSettings"]["type"], "date_ms": s["draftSettings"].get("date"),
                      "order": s["draftSettings"].get("pickOrder"), "rounds": s["rosterSettings"].get("lineupSlotCounts") and sum(
                          v for k, v in s["rosterSettings"]["lineupSlotCounts"].items() if int(k) != 21)},
            "playoff_teams": s["scheduleSettings"]["playoffTeamCount"],
            "reg_weeks": s["scheduleSettings"]["matchupPeriodCount"],
            "acquisition": s["acquisitionSettings"],
            "draft_detail": d.get("draftDetail", {}),
            "status": d.get("status", {}),
        }

    def rosters(self) -> pd.DataFrame:
        d = self._get(self.league_url() + "?view=mRoster&view=mTeam")
        rows = []
        for t in d["teams"]:
            for e in t.get("roster", {}).get("entries", []):
                p = e["playerPoolEntry"]["player"]
                rows.append({"team_id": t["id"], "espn_id": p["id"], "name": p["fullName"],
                             "pos": POS_ID.get(p["defaultPositionId"], "?"), "nfl": PRO_TEAM.get(p["proTeamId"], "?"),
                             "slot": SLOT_ID.get(e["lineupSlotId"], e["lineupSlotId"]),
                             "injury": p.get("injuryStatus", "ACTIVE")})
        return pd.DataFrame(rows)

    def draft_picks(self) -> pd.DataFrame:
        d = self._get(self.league_url() + "?view=mDraftDetail")
        picks = d.get("draftDetail", {}).get("picks", [])
        return pd.DataFrame([{"overall": p["overallPickNumber"], "round": p["roundId"], "round_pick": p["roundPickNumber"],
                              "team_id": p["teamId"], "espn_id": p["playerId"], "keeper": p.get("keeper", False)} for p in picks])

    # ------------------------------------------------------------ players
    def players(self, limit: int = 600, season: int | None = None) -> pd.DataFrame:
        """Player pool with ESPN ADP, ESPN rank, season projection, last-season actual, weekly projections."""
        season = season or self.season
        url = (self.league_url(season) if season == self.season else f"{BASE}/seasons/{season}/segments/0/leaguedefaults/3") + "?view=kona_player_info"
        filt = {"players": {"limit": limit, "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "PPR"}}}
        d = self._get(url, filt)
        rows = []
        for x in d["players"]:
            p = x.get("player", x)
            own = p.get("ownership", {}) or {}
            rk = (p.get("draftRanksByRankType", {}) or {}).get("PPR", {}) or {}
            row = {"espn_id": p["id"], "name": p["fullName"], "pos": POS_ID.get(p.get("defaultPositionId"), "?"),
                   "nfl": PRO_TEAM.get(p.get("proTeamId"), "?"), "injury": p.get("injuryStatus", "ACTIVE"),
                   "adp": own.get("averageDraftPosition"), "pct_owned": own.get("percentOwned"),
                   "espn_rank": rk.get("rank"), "auction": rk.get("auctionValue"),
                   "on_team": x.get("onTeamId", 0), "status": x.get("status"), "outlook": p.get("seasonOutlook", "")}
            weekly = {}
            for s in p.get("stats", []):
                sid = s.get("id", "")
                if s.get("statSplitTypeId") == 0 and sid == f"10{season}":
                    row["espn_proj"] = s.get("appliedTotal"); row["espn_proj_stats"] = {STAT_ID.get(k, k): v for k, v in s.get("stats", {}).items()}
                elif s.get("statSplitTypeId") == 0 and sid == f"00{season}":
                    row["actual"] = s.get("appliedTotal")
                elif s.get("statSplitTypeId") == 0 and sid == f"00{season-1}":
                    row["last_actual"] = s.get("appliedTotal")
                elif s.get("statSplitTypeId") == 1 and s.get("statSourceId") == 1 and sid.startswith("11"):
                    weekly[s.get("scoringPeriodId")] = s.get("appliedTotal")
            row["espn_weekly"] = weekly
            rows.append(row)
        return pd.DataFrame(rows)

    def free_agents(self, limit: int = 300) -> pd.DataFrame:
        url = self.league_url() + "?view=kona_player_info"
        filt = {"players": {"filterStatus": {"value": ["FREEAGENT", "WAIVERS"]}, "limit": limit,
                            "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
        d = self._get(url, filt)
        return pd.DataFrame([{"espn_id": x["player"]["id"], "name": x["player"]["fullName"],
                              "pos": POS_ID.get(x["player"]["defaultPositionId"], "?"), "status": x.get("status")} for x in d["players"]])


if __name__ == "__main__":
    import sys
    e = ESPN(sys.argv[1] if len(sys.argv) > 1 else "980527775")
    s = e.settings(); print(s["name"], s["size"], s["lineup_slots"], s["draft"])
    print(e.players(20)[["name", "pos", "adp", "espn_rank", "espn_proj"]])
