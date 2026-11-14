"""
INDEX-ONLY backtest for first_candle_poll_live.py -- results in NIFTY INDEX POINTS.

Why: the option backtest can only price contracts Fyers still has history for
(current week), so it sees very few trades and hits the 5 req/s limit. The strategy's
exits are defined in index points anyway (SL / TP from the entry spot), so the
strategy can be tested on NIFTY 1-min data alone, over 60+ days, with a handful of
history calls.

What it does (same rules as the live bot):
  * Signals come from first_candle_poll_live.generate_today_events (imported, zero drift):
    Setup A (C1/C2/C3 reversal at the first-candle high/low) and Setup B (full-body
    09:15 candle -> immediate entry at 09:16).
  * Entry = OPEN of the NIFTY bar at the signal time (the bar after the signal candle).
  * CE (BUY): target when NIFTY >= entry + TP; stop = C1 low - C1_SL_OFFSET_POINTS (live default,
    SL_MODE "C1_OFFSET") or entry - SL_INDEX_POINTS (SL_MODE "FIXED"). PE (SELL): mirrored (C1 high + offset). SL wins if both are touched in the same bar. A bar that gaps
    through a level fills at its open. EOD exit at EOD_EXIT_TIME at the bar open.
  * One position at a time; direction alternation follows Config.ALTERNATE_DIRECTION.
  * Costs/slippage are NOT included -- points are gross. The rupee line is a rough
    estimate: points x APPROX_DELTA x lot size.

It then prints the results for the live Config settings, breakdowns (setup, direction,
hour, exit reason) and, if SWEEP is on, a grid over SL / TP / alternation with a
first-half vs second-half expectancy so you can see whether a result is consistent or
a one-period fluke. A grid search on a short window overfits easily -- prefer settings
that are positive in BOTH halves and have a decent number of trades.

Run from the folder holding fyers_appid.txt / fyers_token.txt:  python first_candle_index_backtest.py
"""
import os, sys, time, warnings, logging, requests, urllib3, pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_candle_poll_live import Config as LiveConfig, generate_today_events, profit_lock_points

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - FirstCandleIndexBT - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("FirstCandleIndexBT")


# ══════════════════════════════════════════════════════════════════════════════
#  Everything not listed here is inherited from first_candle_poll_live.Config
#  (SL_INDEX_POINTS, TP_INDEX_POINTS, ALTERNATE_DIRECTION, HEALTHY_BODY_FRAC, ...).
# ══════════════════════════════════════════════════════════════════════════════
class IndexBacktestConfig(LiveConfig):
    # ---- PURE INDEX-POINT RULES (override the live bot's SL mode) ----
    SL_MODE = "FIXED"            # "FIXED" = SL_INDEX_POINTS from entry | "C1_OFFSET" = C1 low/high -/+ offset
    SL_INDEX_POINTS = 40.0       # NIFTY falls 40 pts (CE) / rises 40 pts (PE) -> LOSS
    # TP_INDEX_POINTS is NOT overridden here: it follows first_candle_poll_live.Config.TP_INDEX_POINTS,
    # so editing the target in the live file changes this backtest too.
    # Variant table: same signals, same stop, different exit rules: (label, profit_lock_on, target_pts or None)
    VARIANTS = [("Lock ON,  no target", True, None),
                ("Lock OFF, TP 50", False, 50),
                ("Lock OFF, TP 80", False, 80),
                ("Lock ON,  TP 80", True, 80),
                ("Lock ON,  TP 100", True, 100)]    # set VARIANTS = None to skip
    HIST_DAYS = 90               # calendar days back from today
    CHUNK_DAYS = 45              # days per history request (Fyers 1-min limit is ~100 days)
    APPROX_DELTA = 0.6           # rupee estimate only: points x delta x lot

    SWEEP = False                # True = also grid-search SL/TP (can overfit a short window)
    SWEEP_C1_OFFSET = [10, 20, 30, 40, 50, 60]   # used when SL_MODE == "C1_OFFSET" (live default)
    SWEEP_SL = [20, 25, 30, 40, 50]              # used when SL_MODE == "FIXED"
    SWEEP_TP = [40, 60, 80, 100, 120]
    SWEEP_ALTERNATE = [False, True]
    SWEEP_TOP = 12               # rows to print


class Fyers:
    T1 = "https://api-t1.fyers.in/api/v3"; T2 = "https://api-t1.fyers.in/data"
    MIN_INTERVAL = 0.25; MAX_RETRIES = 6; _last_call = 0.0

    def __init__(self, app, tok):
        auth = tok if ":" in tok else f"{app}:{tok}"
        self.s = requests.Session(); self.s.verify = False
        self.s.headers.update({"Authorization": auth})

    def profile(self):
        return self.s.get(f"{self.T1}/profile").json()

    def history(self, symbol, res, d_from, d_to):
        js = {}
        for attempt in range(self.MAX_RETRIES):
            wait = self.MIN_INTERVAL - (time.monotonic() - self._last_call)
            if wait > 0: time.sleep(wait)
            self._last_call = time.monotonic()
            r = self.s.get(f"{self.T2}/history", params={
                "symbol": symbol, "resolution": res, "date_format": "1",
                "range_from": d_from, "range_to": d_to, "cont_flag": "0"})
            try: js = r.json()
            except Exception: return {}
            if js.get("code") == -360 or "requests/second" in str(js.get("message", "")):
                time.sleep(1.0 + attempt); continue
            return js
        return js


def fetch_nifty(fy: Fyers, cfg) -> pd.DataFrame:
    extra = cfg.EMA_LOOKBACK_DAYS if cfg.USE_EMA_FILTER else 0
    end = datetime.now(IST).date()
    start = end - timedelta(days=cfg.HIST_DAYS + extra)
    frames, d0 = [], start
    while d0 <= end:
        d1 = min(d0 + timedelta(days=cfg.CHUNK_DAYS - 1), end)
        js = fy.history(cfg.UNDERLYING, cfg.BASE_RES, d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d"))
        if js.get("s") != "ok":
            logger.warning(f"history {d0}..{d1}: {js}")
        elif js.get("candles"):
            frames.append(pd.DataFrame(js["candles"],
                          columns=["timestamp", "open", "high", "low", "close", "volume"]))
        d0 = d1 + timedelta(days=1)
    if not frames:
        raise ValueError("No NIFTY history returned (check token / dates).")
    df = pd.concat(frames)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    df = df.iloc[df.index.indexer_between_time(cfg.DATA_START, cfg.SESSION_END)]
    if cfg.USE_EMA_FILTER:
        df["ema200"] = df["close"].ewm(span=cfg.EMA_PERIOD, adjust=False).mean()
    first_day = end - timedelta(days=cfg.HIST_DAYS)
    return df[df.index.date >= first_day].copy()


def build_events(df: pd.DataFrame, cfg) -> List[dict]:
    events: List[dict] = []
    for _d, day_df in df.groupby(df.index.date):
        events.extend(generate_today_events(day_df, cfg))
    events.sort(key=lambda e: e["timestamp"])
    return events


def day_arrays(df: pd.DataFrame) -> Dict[object, dict]:
    days = {}
    for d, g in df.groupby(df.index.date):
        days[d] = {"ts": g.index, "o": g["open"].to_numpy(), "h": g["high"].to_numpy(),
                   "l": g["low"].to_numpy(), "c": g["close"].to_numpy(),
                   "pos": {t: i for i, t in enumerate(g.index)}}
    return days


def simulate(days, events, cfg, sl: float, tp: Optional[float], alternate: bool,
             c1_offset: Optional[float] = None, lock: Optional[bool] = None) -> List[dict]:
    """tp=None -> no fixed target. lock=None -> use cfg.PROFIT_LOCK_ENABLED.
    Profit lock: stop ratchets to entry +/- profit_lock_points(max favourable move so far). The lock is
    updated AFTER a bar's exit checks (it uses the best price of bars already completed), i.e. within one
    bar the older, looser stop applies -- conservative and free of look-ahead."""
    use_lock = cfg.PROFIT_LOCK_ENABLED if lock is None else lock
    trades: List[dict] = []
    busy_until = None; last_dir = None; cur_day = None
    for ev in events:
        ts = ev["timestamp"]; d = ts.date()
        if d != cur_day:
            cur_day = d; last_dir = None; busy_until = None
        if busy_until is not None and busy_until >= ts:
            continue                                   # still holding when the signal arrives
        if alternate and last_dir == ev["dir"]:
            continue
        if not (cfg.TRADE_START <= ts.time() <= cfg.TRADE_END):
            continue
        day = days.get(d)
        i0 = day["pos"].get(ts) if day else None
        if i0 is None:
            continue
        entry = float(day["o"][i0]); sgn = 1 if ev["dir"] == "BUY" else -1
        if c1_offset is not None and ev.get("sl_ref") is not None:
            sl_lvl = ev["sl_ref"] - sgn * c1_offset            # C1 low - off (CE) / C1 high + off (PE)
        else:
            sl_lvl = entry - sgn * sl
        risk = abs(entry - sl_lvl); tp_lvl = (entry + sgn * tp) if tp else None
        if risk <= 0:
            continue
        pts = None; reason = None; exit_ts = None; locked = 0.0
        n = len(day["ts"])
        for i in range(i0, n):
            t = day["ts"][i]; o = float(day["o"][i]); hi = float(day["h"][i]); lo = float(day["l"][i])
            if t.time() >= cfg.EOD_EXIT_TIME:
                pts, reason, exit_ts = sgn * (o - entry), "eod_exit", t; break
            stop_lvl, stop_kind = sl_lvl, "stop_loss"
            if locked > 0:                                  # profit-lock stop is tighter than the original
                stop_lvl, stop_kind = entry + sgn * locked, "profit_lock"
            if sgn == 1: sl_hit, tp_hit = lo <= stop_lvl, (tp_lvl is not None and hi >= tp_lvl)
            else:        sl_hit, tp_hit = hi >= stop_lvl, (tp_lvl is not None and lo <= tp_lvl)
            if sl_hit:
                gap = i > i0 and ((sgn == 1 and o <= stop_lvl) or (sgn == -1 and o >= stop_lvl))
                pts, reason, exit_ts = (sgn * (o - entry) if gap else sgn * (stop_lvl - entry)), stop_kind, t; break
            if tp_hit:
                gap = i > i0 and ((sgn == 1 and o >= tp_lvl) or (sgn == -1 and o <= tp_lvl))
                pts, reason, exit_ts = (sgn * (o - entry) if gap else tp), "target", t; break
            if use_lock:                                    # ratchet from this bar's best price
                fav = (hi - entry) if sgn == 1 else (entry - lo)
                locked = max(locked, profit_lock_points(fav, cfg))
        if pts is None:                                # data ended before EOD
            pts, reason, exit_ts = sgn * (float(day["c"][-1]) - entry), "data_end", day["ts"][-1]
        trades.append({"date": d, "direction": ev["dir"], "setup": ev["name"], "level": ev.get("level"),
                       "entry_time": ts, "entry_spot": round(entry, 2), "exit_time": exit_ts,
                       "touch": (",".join(ev["touched"]) if ev.get("touched") is not None else None),
                       "cands_txt": (" ; ".join(f"{n} {c[0]} {c[3]:.1f}-{c[2]:.1f}" for n, c in zip(("C1", "C2", "C3"), ev["cands"]))
                                     if ev.get("cands") else ""),
                       "exit_reason": reason, "risk_pts": round(risk, 1), "locked_pts": locked, "pnl_pts": round(pts, 2),
                       "result": ("PROFIT" if pts > 0 else "LOSS" if pts < 0 else "FLAT")})
        busy_until = exit_ts; last_dir = ev["dir"]
    return trades


def stats(trades: List[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0,
                "net_pts": 0.0, "profit_factor": 0.0, "max_dd": 0.0, "avg_risk": 0.0}
    p = np.array([t["pnl_pts"] for t in trades], dtype=float)
    wins, losses = p[p > 0], p[p < 0]
    cum = np.cumsum(p)
    return {"trades": n, "win_rate": round(100.0 * len(wins) / n, 1),
            "avg_win": round(float(wins.mean()), 1) if len(wins) else 0.0,
            "avg_loss": round(float(losses.mean()), 1) if len(losses) else 0.0,
            "expectancy": round(float(p.mean()), 2), "net_pts": round(float(p.sum()), 1),
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) else float("inf"),
            "max_dd": round(float((np.maximum.accumulate(cum) - cum).max()), 1),
            "avg_risk": round(float(np.mean([t["risk_pts"] for t in trades])), 1)}


def half_expectancy(trades: List[dict]):
    if len(trades) < 4:
        return None, None
    dates = sorted({t["date"] for t in trades}); mid = dates[len(dates) // 2]
    a = [t["pnl_pts"] for t in trades if t["date"] < mid]
    b = [t["pnl_pts"] for t in trades if t["date"] >= mid]
    return (round(float(np.mean(a)), 2) if a else None, round(float(np.mean(b)), 2) if b else None)


def print_breakdown(trades: List[dict], key: str, title: str):
    df = pd.DataFrame(trades)
    print(f"  {title}:")
    for k, g in df.groupby(key):
        wr = 100.0 * (g["pnl_pts"] > 0).mean()
        print(f"    {str(k):<24} n={len(g):<4} win%={wr:5.1f}  net={g['pnl_pts'].sum():8.1f} pts  "
              f"avg={g['pnl_pts'].mean():7.2f}")


def main():
    cfg = IndexBacktestConfig()
    for f in ("fyers_appid.txt", "fyers_token.txt"):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found in {os.getcwd()} -- run from the folder that has it")
    fy = Fyers(open("fyers_appid.txt").read().strip(), open("fyers_token.txt").read().strip())
    p = fy.profile()
    if p.get("s") != "ok":
        raise ConnectionError(f"Fyers auth failed: {p}")
    logger.info(f"Fyers connected -- {p['data']['name']}")

    df = fetch_nifty(fy, cfg)
    n_days = df.index.normalize().nunique()
    logger.info(f"NIFTY 1-min: {len(df)} bars over {n_days} trading days "
                f"({df.index[0].date()} .. {df.index[-1].date()})")
    events = build_events(df, cfg)
    days = day_arrays(df)
    n_b = sum(1 for e in events if e["name"] == "first_candle_full_body")
    logger.info(f"Signals: {len(events)} total ({n_b} full-body 9:15, {len(events) - n_b} line-pattern) "
                f"| healthy_frac={cfg.HEALTHY_BODY_FRAC} C1={cfg.REQUIRE_HEALTHY_C1} C3={cfg.REQUIRE_HEALTHY_C3} "
                f"prox={cfg.LEVEL_PROXIMITY_PCT}% EMA={cfg.USE_EMA_FILTER}")

    sl, tp, alt = cfg.SL_INDEX_POINTS, cfg.TP_INDEX_POINTS, cfg.ALTERNATE_DIRECTION
    c1 = cfg.C1_SL_OFFSET_POINTS if cfg.SL_MODE == "C1_OFFSET" else None
    trades = simulate(days, events, cfg, sl, tp, alt, c1)
    s = stats(trades)
    risk_avg = s["avg_risk"] if c1 is not None and trades else sl
    be = (100.0 * risk_avg / (risk_avg + tp)) if tp else None
    tp_txt = f"TP {tp:g} pts (RR 1:{tp / risk_avg:.2f})" if tp else "NO fixed target"
    lock_txt = (f"profit-lock +{cfg.PROFIT_LOCK_FIRST_PROFIT:g}@{cfg.PROFIT_LOCK_START_POINTS:g} then "
                f"+{cfg.PROFIT_LOCK_STEP_POINTS:g}/{cfg.PROFIT_LOCK_STEP_POINTS:g}pts" if cfg.PROFIT_LOCK_ENABLED else "profit-lock OFF")
    sl_txt = (f"SL = C1 extreme -/+ {c1:g} (avg risk {risk_avg:g} pts)" if c1 is not None else f"SL {sl:g}")
    rupees = s["net_pts"] * cfg.APPROX_DELTA * cfg.LOT
    print("\n" + "=" * 70)
    print(f"  LIVE CONFIG: {sl_txt} / {tp_txt} / {lock_txt} | alternate={alt} | {n_days} days")
    print(f"  Trades {s['trades']} | win% {s['win_rate']} (break-even {('%.1f%%' % be) if be else 'n/a - no fixed target'}) | "
          f"avg win {s['avg_win']} / avg loss {s['avg_loss']} pts")
    print(f"  Expectancy {s['expectancy']} pts/trade | net {s['net_pts']} pts | PF {s['profit_factor']} | "
          f"max DD {s['max_dd']} pts")
    print(f"  ~Rs {rupees:,.0f} gross (rough: pts x delta {cfg.APPROX_DELTA} x lot {cfg.LOT}; no costs)")
    if trades:
        print("-" * 70)
        for t in trades:
            t["hour"] = t["entry_time"].hour
        print("  Trade list (exit: target / stop_loss / profit_lock = trailed stop hit / eod_exit; PROFIT or LOSS by points):")
        print("  (touch = which of C1/C2/C3 reached the line; C1/C2/C3 = bar start time and low-high)")
        for t in trades:
            print(f"    {t['entry_time']:%Y-%m-%d %H:%M} {t['direction']:<4} {t['setup']:<22} entry {t['entry_spot']:>9.2f} "
                  f"-> {t['exit_reason']:<9} {t['pnl_pts']:+8.2f} pts  {t['result']}")
            if t.get("touch") is not None:
                print(f"        line {t['level']:.2f} | touch: {t['touch'] or 'NONE'} | {t['cands_txt']}")
        print("-" * 70)
        print_breakdown(trades, "setup", "By setup")
        print_breakdown(trades, "direction", "By direction")
        print_breakdown(trades, "hour", "By entry hour")
        print_breakdown(trades, "exit_reason", "By exit reason")
        pd.DataFrame(trades).to_csv("FirstCandle_Index_Trades.csv", index=False)
        print("  Trades saved -> FirstCandle_Index_Trades.csv")
    print("=" * 70)

    if cfg.VARIANTS:
        print(f"\nSame signals and stop ({sl_txt}), different exit rules:")
        print(f"  {'variant':<22} {'trades':>6} {'win%':>6} {'net pts':>9} {'exp':>7} {'PF':>5} {'maxDD':>7}")
        for label, lk, tp_ in cfg.VARIANTS:
            st = stats(simulate(days, events, cfg, sl, tp_, alt, c1, lk))
            print(f"  {label:<22} {st['trades']:>6} {st['win_rate']:>6} {st['net_pts']:>9} "
                  f"{st['expectancy']:>7} {st['profit_factor']:>5} {st['max_dd']:>7}")
        print("  (different exits change which signals overlap, so trade counts can differ)")

    if cfg.SWEEP:
        rows = []
        modes = [("C1", o_) for o_ in cfg.SWEEP_C1_OFFSET] if c1 is not None else [("FIXED", s_) for s_ in cfg.SWEEP_SL]
        for a in cfg.SWEEP_ALTERNATE:
            for mode, v in modes:
                for t_ in cfg.SWEEP_TP:
                    tr = simulate(days, events, cfg, v, t_, a, v if mode == "C1" else None)
                    st = stats(tr); h1, h2 = half_expectancy(tr)
                    ar = st["avg_risk"] if mode == "C1" else v
                    rows.append({"alternate": a, "SL": (f"C1-{v:g}" if mode == "C1" else v), "TP": t_,
                                 "RR": round(t_ / ar, 2) if ar else 0.0, **st,
                                 "exp_first_half": h1, "exp_second_half": h2,
                                 "breakeven_win_rate": round(100.0 * ar / (ar + t_), 1) if ar else 0.0})
        sw = pd.DataFrame(rows).sort_values("expectancy", ascending=False)
        sw.to_csv("FirstCandle_Index_Sweep.csv", index=False)
        cols = ["alternate", "SL", "TP", "avg_risk", "trades", "win_rate", "breakeven_win_rate", "expectancy",
                "exp_first_half", "exp_second_half", "profit_factor", "max_dd"]
        print(f"\nSweep (top {cfg.SWEEP_TOP} by expectancy, pts/trade; full grid -> FirstCandle_Index_Sweep.csv):")
        print(sw[cols].head(cfg.SWEEP_TOP).to_string(index=False))
        pos_both = sw[(sw["exp_first_half"] > 0) & (sw["exp_second_half"] > 0) & (sw["trades"] >= 15)]
        print(f"\nCombos positive in BOTH halves with >= 15 trades: {len(pos_both)} of {len(sw)}")
        print("Caution: more combos tried on a short window = more chance a 'best' one is luck.")


if __name__ == "__main__":
    main()
