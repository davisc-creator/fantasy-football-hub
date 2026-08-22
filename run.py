#!/usr/bin/env python
"""Fantasy Football Hub CLI.

  python run.py refresh [league]     pull fresh ESPN/nflverse data, rebuild projections for all leagues
  python run.py train                retrain the season model (and print validation)
  python run.py draft <league> [--slot N] [--port P]   open the live draft room
  python run.py mock <league> [slot] run a mock draft in the terminal
  python run.py backtest [year]      replay a past season's draft and score by actual results
  python run.py report [league]      rebuild out/hub.html (all leagues) and print the weekly text report
  python run.py weekly               refresh + report + email (what the scheduler runs)
  python run.py schedule             install a launchd job: Tue 7am (waivers) + Sun 9am (lineups)
  python run.py publish              rebuild dashboard and git commit+push (GitHub Pages)
"""
import sys, argparse

def publish():
    """Commit docs/ (the dashboard) and push, so GitHub Pages / the phone sees fresh data."""
    import subprocess, pathlib, datetime as dt
    root = pathlib.Path(__file__).resolve().parent
    if not (root / ".git").exists(): print("publish skipped: not a git repo"); return
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
    run("add", "-A")
    if run("diff", "--cached", "--quiet").returncode == 0: print("publish: nothing changed"); return
    run("commit", "-m", f"Weekly update {dt.datetime.now():%Y-%m-%d %H:%M}")
    r = run("push"); print("publish:", "pushed" if r.returncode == 0 else r.stderr.strip()[-200:])

def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"): print(__doc__); return
    cmd, rest = a[0], a[1:]
    if cmd == "refresh":
        from ffhub.config import LEAGUES
        from ffhub import nfl, projections
        if "--data" in rest:
            for fn in (nfl.weekly_stats, nfl.schedules, nfl.playerids, nfl.fp_rankings, nfl.rosters_now):
                fn(refresh=True)
        keys = [k for k in rest if k in LEAGUES] or list(LEAGUES)
        if "yahoo12" in keys:
            try:
                from ffhub import yahoo; yahoo.refresh()
            except Exception as ex: print("yahoo refresh skipped:", ex)
        for k in keys:
            t = projections.build(LEAGUES[k]); print(f"{k}: {len(t)} players projected")
    elif cmd == "train":
        from ffhub import models; models.__name__ == "__main__"
        import runpy; runpy.run_module("ffhub.models", run_name="__main__")
    elif cmd == "draft":
        p = argparse.ArgumentParser(); p.add_argument("league"); p.add_argument("--slot", type=int); p.add_argument("--port", type=int, default=8765)
        o = p.parse_args(rest)
        from ffhub.server import serve; serve(o.league, o.slot, o.port)
    elif cmd == "mock":
        from ffhub.mock import run; run(rest[0] if rest else "espn10", int(rest[1]) if len(rest) > 1 else None)
    elif cmd == "backtest":
        from ffhub.backtest import run; run(int(rest[0]) if rest else 2024)
    elif cmd == "report":
        from ffhub import report
        report.build()
        from ffhub.config import LEAGUES
        for k in (rest or [k for k in LEAGUES]):
            print(report.weekly_text(k)); print("\n" + "=" * 70 + "\n")
    elif cmd == "weekly":
        import datetime as dt
        from ffhub.config import LEAGUES
        from ffhub import projections, report, emailer
        try:
            from ffhub import yahoo; yahoo.refresh()
        except Exception as ex: print("yahoo refresh skipped:", ex)
        for k in LEAGUES: projections.build(LEAGUES[k])
        report.build()
        texts = [report.weekly_text(k) for k in LEAGUES]
        body = ("\n\n" + "=" * 70 + "\n\n").join(texts)
        html = "<pre style='font:13px Menlo,monospace'>" + body.replace("<", "&lt;") + "</pre><p>Full dashboard attached (hub.html).</p>"
        print(emailer.send(f"Fantasy Hub — {dt.date.today():%a %b %d}", body, html))
        publish()
    elif cmd == "publish":
        from ffhub import report; report.build(); publish()
    elif cmd == "schedule":
        import os, subprocess, pathlib
        root = pathlib.Path(__file__).resolve().parent
        plist = pathlib.Path.home() / "Library/LaunchAgents/com.carson.ffhub.plist"
        py = root / ".venv/bin/python"
        cal = "".join(f"<dict><key>Weekday</key><integer>{d}</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>0</integer></dict>" for d, h in ((2, 7), (0, 9)))
        plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.carson.ffhub</string>
<key>ProgramArguments</key><array><string>{py}</string><string>{root/'run.py'}</string><string>weekly</string></array>
<key>WorkingDirectory</key><string>{root}</string>
<key>StartCalendarInterval</key><array>{cal}</array>
<key>StandardOutPath</key><string>{root/'out/weekly.log'}</string><key>StandardErrorPath</key><string>{root/'out/weekly.log'}</string>
</dict></plist>""")
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        r = subprocess.run(["launchctl", "load", str(plist)], capture_output=True, text=True)
        print("installed", plist, r.stderr or "ok", "— runs Tue 07:00 and Sun 09:00")
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
