# Fantasy Football Hub — 2026

Python data pipeline + ML projections + draft and season engines for three leagues:

| Key | League | Platform | Teams | Status |
|---|---|---|---|---|
| `espn10` | Blues Backshots 2026 | ESPN | 10 | drafts **Aug 31, 7 PM PDT** (order randomized 1h before) |
| `espn14` | Blues Fantasy Football | ESPN | 14 | draft not scheduled yet |
| `yahoo12` | Sean's Swell League 2.0 | Yahoo | 12 | drafted Aug 19 — in season mode |

## Commands

```bash
.venv/bin/python run.py refresh            # pull ESPN + FantasyPros + nflverse, rebuild projections (all leagues)
.venv/bin/python run.py draft espn10       # LIVE DRAFT ROOM → http://127.0.0.1:8765 (auto-syncs picks from ESPN)
.venv/bin/python run.py report             # rebuild out/hub.html + print weekly advice for every league
.venv/bin/python run.py weekly             # refresh + report + email (what the scheduler runs)
.venv/bin/python run.py schedule           # install Tue 7am / Sun 9am launchd job
.venv/bin/python run.py mock espn10 [slot] # terminal mock draft
.venv/bin/python run.py backtest 2024      # replay a past draft, score by actual results
.venv/bin/python run.py train              # retrain models, print validation
```

Open **`out/hub.html`** for the dashboard (cheat sheets, my team, waivers, trades, power rankings).

## Draft day (ESPN)

1. `python run.py refresh` an hour before (fresh ADP + injuries).
2. `python run.py draft espn10` — the room opens in your browser. It polls ESPN every 6 s, picks up your draft slot once the order is randomized, and records every pick automatically. You can also type a name + Enter to record a pick manually, and Undo.
3. On your pick, the left panel ranks candidates by **expected final-roster value** (it simulates the rest of the draft for each one). "Lasts to next" is the probability the player is still there at your following pick — take the best player now unless the alternative is very likely to last.
4. Best-available panel shows tiers, VORP, ADP/ECR and survival odds; the right panel shows your roster and byes.

## How the numbers are made

- **Data**: ESPN league API (settings, ADP, projections, rosters, live draft), FantasyPros expert consensus (ECR + spread, via nflverse), nflverse play-by-play-derived weekly stats 2015–2025, schedules with Vegas lines, player ID crosswalk.
- **Projections** = 70 % *market curve* + 30 % *ML model*.
  - *Market curve*: position-specific curve fit on 2018–2025 mapping ADP/ECR positional rank → actual points. Validated as the best within-position ranker (Spearman RB .62 / WR .56 vs ESPN's .42 / .36).
  - *ML model*: gradient boosting (scikit-learn HistGradientBoosting) predicting points-per-game deviation from ESPN and games played, from ESPN projection, ADP, two prior seasons of usage (targets, carries, air-yard share, EPA), age, experience, draft capital. Leave-one-season-out: MAE 48.7 vs ESPN 55.4. Adds ~3 % MAE improvement on top of the market. Floor / ceiling from 20th / 80th-percentile models.
- **Roster value**: expected points of the optimal weekly lineup over 18 weeks with byes, injury availability, per-player season-outcome draws (upside has option value), playoff weeks ×1.5, replacement-level streaming for empty/weak slots.
- **Draft engine**: each player's draft time ~ Normal(ADP, sd); opponents take the earliest available player their roster allows. Survival % from simulations; the recommendation maximises expected final-roster value. Validated by replaying 2024 (see Model tab in the dashboard).
- **Season engine**: lineups (ESPN weekly projection × our season tilt, injury-adjusted), waivers (net rest-of-season gain after the best drop), trades (only deals where both sides gain), power rankings.

## Files

```
ffhub/        config.py espn.py nfl.py features.py market.py models.py projections.py
              draft.py server.py season.py report.py backtest.py mock.py emailer.py
draft.html    live draft room UI      hub_template.html  dashboard template
data/cache/   parquet caches (nflverse, ESPN history, projections)   models/  trained models
out/          hub.html, hub_data.json, draft state, logs
.env          ESPN_S2, SWID (+ GMAIL_USER, GMAIL_APP_PASSWORD, REPORT_TO for email)
```

Yahoo rosters currently come from `data/players.csv` (owner column) — refresh by asking Claude to re-read the league, or set up the Yahoo API.
