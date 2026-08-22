"""League definitions, paths, and scoring rules.

Scoring is expressed in a platform-neutral stat vocabulary so nflverse history,
ESPN projections, and Yahoo projections can all be scored identically.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
OUT = ROOT / "out"
MODELS = ROOT / "models"
for _d in (CACHE, OUT, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

SEASON = 2026
REG_WEEKS = 18          # NFL weeks in 2026 regular season
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
FLEX_POS = ["RB", "WR", "TE"]


def load_env() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        raw = p.read_text().replace("\\\n", " ")          # join backslash line-continuations (pasted cURL)
        key = None
        for line in raw.splitlines():
            if "=" in line and not line.lstrip().startswith("#") and not line.lstrip().startswith("-"):
                k, v = line.split("=", 1)
                key = k.strip(); env[key] = v.strip().strip('"').strip("'")
            elif key == "YAHOO_CURL" and line.strip():           # cURL lines without '=' continue the value
                env[key] += " " + line.strip()
    env.update({k: v for k, v in os.environ.items() if k in ("ESPN_S2", "SWID")})
    return env


# ---------------------------------------------------------------- scoring
# Neutral stat keys: pass_yd pass_td pass_int rush_yd rush_td rec rec_yd rec_td
# fum_lost two_pt  bonus_rush100 bonus_rush125(?) big_play40 ...
ESPN_PPR_SCORING = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0, "pass_2pt": 2.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
    "rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
    "fum_lost": -2.0,
    # kicking
    "fg_0_39": 3.0, "fg_40_49": 4.0, "fg_50_plus": 5.0, "fg_miss": -1.0, "xp": 1.0,
}

YAHOO_12_SCORING = {
    **ESPN_PPR_SCORING,
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    # league-specific: +2 for any 40+ yd run or reception, +1 at 100 rush, +2 at 125 rush
    "rush_40plus": 2.0, "rec_40plus": 2.0, "rush_100_bonus": 1.0, "rush_125_bonus": 2.0,
}


@dataclass
class Roster:
    QB: int = 1; RB: int = 2; WR: int = 2; TE: int = 1; FLEX: int = 1
    K: int = 1; DST: int = 1; BENCH: int = 7; IR: int = 1

    @property
    def starters(self) -> int:
        return self.QB + self.RB + self.WR + self.TE + self.FLEX + self.K + self.DST

    @property
    def size(self) -> int:
        return self.starters + self.BENCH


@dataclass
class League:
    key: str
    name: str
    platform: str               # espn | yahoo
    league_id: str
    teams: int
    my_team_id: int | None
    my_team_name: str
    roster: Roster
    scoring: dict
    playoff_teams: int
    reg_season_weeks: int       # matchup periods in regular season
    playoff_weeks: list[int]
    draft_rounds: int
    draft_date: str | None = None
    notes: str = ""

    @property
    def starter_counts(self) -> dict:
        r = self.roster
        return {"QB": r.QB, "RB": r.RB, "WR": r.WR, "TE": r.TE, "FLEX": r.FLEX, "K": r.K, "DST": r.DST}


LEAGUES: dict[str, League] = {
    "espn10": League(
        key="espn10", name="Blues Backshots 2026", platform="espn", league_id="980527775",
        teams=10, my_team_id=2, my_team_name="Carson's Competitive Team",
        roster=Roster(), scoring=ESPN_PPR_SCORING, playoff_teams=6,
        reg_season_weeks=14, playoff_weeks=[15, 16, 17], draft_rounds=16,
        draft_date="2026-08-31 19:00 PDT",
    ),
    "espn14": League(
        key="espn14", name="Blues Fantasy Football", platform="espn", league_id="2112605200",
        teams=14, my_team_id=6, my_team_name="Carson's Competitive Team",
        roster=Roster(), scoring=ESPN_PPR_SCORING, playoff_teams=6,
        reg_season_weeks=14, playoff_weeks=[15, 16, 17], draft_rounds=16,
        draft_date=None,
    ),
    "yahoo12": League(
        key="yahoo12", name="Sean's Swell League 2.0", platform="yahoo", league_id="1356748",
        teams=12, my_team_id=12, my_team_name="Brock Hard",
        roster=Roster(BENCH=6, IR=2), scoring=YAHOO_12_SCORING, playoff_teams=4,
        reg_season_weeks=15, playoff_weeks=[16, 17], draft_rounds=15,
        draft_date="2026-08-19", notes="Already drafted. 40+yd big-play bonuses; rush yardage bonuses.",
    ),
}
