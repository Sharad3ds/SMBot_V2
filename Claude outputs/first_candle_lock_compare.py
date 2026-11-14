"""
PROFIT-LOCK COMPARISON for first_candle_poll_live.py  (index points, NIFTY 1-min only)

Question it answers:  keep it simple (fixed SL + fixed TP, no lock)  or add a profit lock -- and if a lock,
which START level?  Early locks (+10 at +50) make tiny wins that option costs/theta wipe out, so later starts
(+20 at +60, +30 at +70 ...) are tested too.

It reuses first_candle_index_backtest.py (same signals, same fills, same bar-walk) so nothing drifts.
Run from the folder holding fyers_appid.txt / fyers_token.txt:   python first_candle_lock_compare.py

What it prints, for each stop style (FIXED 40 from entry, and live C1-offset stop):
  * one row per (lock rule x target):  trades, win%, net pts, expectancy, PF, max DD,
    first-half / second-half expectancy (is it consistent?), small-win count, and COST-ADJUSTED net
    (net minus COST_PTS_PER_TRADE per trade -- a rough stand-in for option slippage/theta/brokerage).
  * a VERDICT block comparing the simple baseline (no lock, TP 50) with the best lock rule.
Saved to FirstCandle_Lock_Compare.csv.

Caution: ~40 rows tried on ~3 months of data -> the 'best' row can be luck. The verdict therefore asks for a
clear margin AND positive results in both halves before it recommends a lock.
"""
import os, sys, copy, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import first_candle_index_backtest as B

logger = logging.getLogger("FirstCandleLockCompare")


class CompareConfig(B.IndexBacktestConfig):
    HIST_DAYS = 90
    # (label, start_pts, first_profit_pts, step_pts)  -- None start = lock OFF
    LOCK_RULES = [("Lock OFF",                  None, 0,  10),
                  ("+10 @50, +10/10 (current)", 50,  10, 10),
                  ("+20 @60, +10/10",           60,  20, 10),
                  ("+30 @70, +10/10",           70,  30, 10),
                  ("+40 @80, +10/10",           80,  40, 10),
                  ("+20 @60, +20/20",           60,  20, 20),
                  ("+30 @70, +20/20",           70,  30, 20)]
    TARGETS = [50, 100, None]            # None = no fixed target (only the lock/stop/EOD exits)
    STOP_STYLES = ["FIXED", "C1_OFFSET"] # FIXED = entry -/+ SL_INDEX_POINTS | C1_OFFSET = live stop
    COST_PTS_PER_TRADE = 5.0             # rough index-point equivalent of option slippage+theta+brokerage per trade
    SMALL_WIN_MAX = 30.0                 # "small win" = 0 < pnl <= this
    CLEAR_MARGIN_PTS = 150.0             # lock must beat baseline by this many cost-adjusted pts to be 'clear'
    BASELINE = ("Lock OFF", 50)          # simple version: no lock, TP 50
    HALF_MIN_TRADES = 4


def run_rule(days, events, cfg, stop_style, rule, tp):
    label, start, first, step = rule
    c = copy.copy(cfg)
    c.PROFIT_LOCK_ENABLED = start is not None
    if start is not None:
        c.PROFIT_LOCK_START_POINTS = float(start)
        c.PROFIT_LOCK_FIRST_PROFIT = float(first)
        c.PROFIT_LOCK_STEP_POINTS = float(step)
    c1 = cfg.C1_SL_OFFSET_POINTS if stop_style == "C1_OFFSET" else None
    return B.simulate(days, events, c, cfg.SL_INDEX_POINTS, tp, cfg.ALTERNATE_DIRECTION, c1, c.PROFIT_LOCK_ENABLED)


def summarize(trades, cfg):
    st = B.stats(trades)
    h1, h2 = B.half_expectancy(trades)
    p = np.array([t["pnl_pts"] for t in trades], dtype=float) if trades else np.array([])
    small = int(((p > 0) & (p <= cfg.SMALL_WIN_MAX)).sum()) if len(p) else 0
    return {**st, "exp_h1": h1, "exp_h2": h2, "small_wins": small,
            "net_after_cost": round(st["net_pts"] - cfg.COST_PTS_PER_TRADE * st["trades"], 1)}


def compare(days, events, cfg):
    rows = []
    for style in cfg.STOP_STYLES:
        for rule in cfg.LOCK_RULES:
            for tp in cfg.TARGETS:
                if rule[1] is None and tp is None:
                    continue                          # no lock + no target = never exits except stop/EOD; skip
                if rule[1] is not None and tp is not None and rule[1] >= tp:
                    continue                          # lock starts at/after the target -> identical to 'Lock OFF', skip
                tr = run_rule(days, events, cfg, style, rule, tp)
                rows.append({"stop": style, "lock": rule[0], "tp": tp if tp else "none", **summarize(tr, cfg)})
    return pd.DataFrame(rows)


def print_tables(df, cfg):
    cols = ["lock", "tp", "trades", "win_rate", "net_pts", "net_after_cost", "expectancy", "profit_factor",
            "max_dd", "exp_h1", "exp_h2", "small_wins"]
    for style in cfg.STOP_STYLES:
        sub = df[df["stop"] == style]
        if sub.empty:
            continue
        risk = sub["avg_risk"].iloc[0]
        sl_txt = (f"SL = entry -/+ {cfg.SL_INDEX_POINTS:g}" if style == "FIXED"
                  else f"SL = C1 extreme -/+ {cfg.C1_SL_OFFSET_POINTS:g} (avg risk ~{risk:g})")
        print("\n" + "=" * 118)
        print(f"  {sl_txt} | alternate={cfg.ALTERNATE_DIRECTION} | cost proxy {cfg.COST_PTS_PER_TRADE:g} pts/trade")
        print("=" * 118)
        print(sub.sort_values("net_after_cost", ascending=False)[cols].to_string(index=False))


def verdict(df, cfg):
    print("\n" + "#" * 118)
    print("  VERDICT  (baseline = simple version: " f"{cfg.BASELINE[0]}, TP {cfg.BASELINE[1]})")
    print("#" * 118)
    for style in cfg.STOP_STYLES:
        sub = df[df["stop"] == style]
        base = sub[(sub["lock"] == cfg.BASELINE[0]) & (sub["tp"] == cfg.BASELINE[1])]
        locked = sub[sub["lock"] != "Lock OFF"]
        if base.empty or locked.empty:
            continue
        b = base.iloc[0]
        # candidates: lock rules positive in BOTH halves, ranked by cost-adjusted net
        ok = locked[(locked["exp_h1"].fillna(-1) > 0) & (locked["exp_h2"].fillna(-1) > 0)]
        pool = ok if not ok.empty else locked
        best = pool.sort_values("net_after_cost", ascending=False).iloc[0]
        gain = best["net_after_cost"] - b["net_after_cost"]
        print(f"\n  [{style}]")
        print(f"   baseline : net {b['net_pts']:+.0f} pts, after-cost {b['net_after_cost']:+.0f}, PF {b['profit_factor']}, "
              f"maxDD {b['max_dd']}, halves {b['exp_h1']}/{b['exp_h2']}")
        print(f"   best lock: {best['lock']} + TP {best['tp']} -> net {best['net_pts']:+.0f} pts, "
              f"after-cost {best['net_after_cost']:+.0f}, PF {best['profit_factor']}, maxDD {best['max_dd']}, "
              f"halves {best['exp_h1']}/{best['exp_h2']}, small wins {best['small_wins']}")
        print(f"   difference (after-cost): {gain:+.0f} pts over {int(best['trades'])} trades")
        if ok.empty:
            print("   NOTE: no lock rule was positive in both halves - treat any lock result as fragile.")
        if gain >= cfg.CLEAR_MARGIN_PTS:
            print(f"   -> CLEAR WIN for the lock (>= {cfg.CLEAR_MARGIN_PTS:g} pts): use '{best['lock']}', TP {best['tp']}.")
        elif gain >= cfg.CLEAR_MARGIN_PTS / 3:
            print(f"   -> lock helps a bit but not clearly: marginal. Prefer the simpler version unless drawdown matters to you.")
        else:
            print(f"   -> no meaningful gain: keep it SIMPLE (no lock).")
    print("\n  Reminder: index points, 1-min bars, grid on ~3 months -> confirm on fresh data / paper trading before going live.")


def main():
    cfg = CompareConfig()
    for f in ("fyers_appid.txt", "fyers_token.txt"):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found in {os.getcwd()} -- run from the folder that has it")
    fy = B.Fyers(open("fyers_appid.txt").read().strip(), open("fyers_token.txt").read().strip())
    p = fy.profile()
    if p.get("s") != "ok":
        raise ConnectionError(f"Fyers auth failed: {p}")
    logger.info(f"Fyers connected -- {p['data']['name']}")
    df = B.fetch_nifty(fy, cfg)
    logger.info(f"NIFTY 1-min: {len(df)} bars over {df.index.normalize().nunique()} days "
                f"({df.index[0].date()} .. {df.index[-1].date()})")
    events = B.build_events(df, cfg)
    days = B.day_arrays(df)
    logger.info(f"Signals: {len(events)}")
    res = compare(days, events, cfg)
    print_tables(res, cfg)
    verdict(res, cfg)
    res.to_csv("FirstCandle_Lock_Compare.csv", index=False)
    print("\n  Saved -> FirstCandle_Lock_Compare.csv")


if __name__ == "__main__":
    main()
