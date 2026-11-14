"""
Backtest for first_candle_poll_live.py (First-Candle High/Low support-resistance
+ 3-candle reversal, SL / TP measured in NIFTY index points).

Signal logic is IMPORTED VERBATIM from the live bot to prevent drift:
Config, generate_today_events (-> detect_signal, get_first_candle_levels,
within_band). Only the execution machinery (bar-walking, option-side risk
simulation, EOD boundary, CSV export) lives here -- same structure as
pattern_live_backtest.py.

STRATEGY (see first_candle_poll_live.py):
  GREEN line = high of the day's 09:15 1-min candle, RED line = its low.
  Setup A -- CE BUY at GREEN: C1 red, C2 green, C3 green (+ rising closes OR body ratio);
             PE BUY at RED  : C1 green, C2 red, C3 red   (+ falling closes OR body ratio).
             C3 must be healthy (C1 need not be); the C1/C2 extreme (pullback low for CE,
             retest high for PE) must be within LEVEL_PROXIMITY_PCT of the line; C3 closes on
             the correct side of it.
  Setup B -- if the 09:15 candle is full-body (>= FIRST_CANDLE_FULL_BODY_FRAC of its range),
             enter at the open of the next candle in its direction (green -> CE, red -> PE).

RISK (matches live, RISK_MODE = "INDEX_POINTS"): SL = C1 low/high -/+ C1_SL_OFFSET_POINTS
  (SL_MODE "C1_OFFSET", default) or SL_INDEX_POINTS from entry (SL_MODE "FIXED"); TP =
  TP_INDEX_POINTS NIFTY points from the entry spot (CE: down/up, PE: up/down).
  Checked on each NIFTY 1-min bar's high/low; SL wins if both are touched in the
  same bar. The entry spot is the NIFTY bar OPEN at the entry minute.
  Exit price = the OPTION's price at the moment the index touched the level,
  estimated by interpolating the option's open -> low/high over the NIFTY bar's
  move (open -> low/high), then EXIT_SLIPPAGE_PCT is taken off (bot-sent market
  exit). Gaps through the level fill at the bar open. This is an estimate: the 1-min
  data does not say WHEN inside the minute the level was touched.
  RISK_MODE = "PREMIUM_PCT" keeps the old premium-% logic (exact SL/TP price fills).
  EOD exit at EOD_EXIT_TIME on the ENTRY day. One position at a time.

Differences vs pattern_live_backtest.py (deliberate):
  * The entry bar itself is checked for SL/TP (fill is at its OPEN, so the rest
    of that bar's range happens after the fill).
  * Bars are walked only up to the next signal's time, so a signal that arrives
    while the trade is still open is ignored (as live does).
  * The LAST trade is walked bar-by-bar too (the old flush closed it at EOD
    price without checking SL/TP).
  * PROFIT LOCK (live Config.PROFIT_LOCK_*): once NIFTY is +50 pts in favour the stop moves to entry +10,
    +60 -> +20, +70 -> +30 ... (index points, ratchet). TP_INDEX_POINTS = None = no fixed target. The lock
    is updated from completed bars, so within one bar the older stop applies (conservative).
  * Option symbol uses the expiry for each SIGNAL DATE (ExpiryResolver, with the
    Tuesday fallback) instead of the current week's code for every date.
  * The backtest window starts at midnight of the first day, so that day's
    09:15 candle (the level) is never cut off.
  * Entries are skipped if the option bar at the signal time is missing by
    more than MAX_ENTRY_DELAY_MIN (thin / absent data).
  * Direction alternation follows Config.ALTERNATE_DIRECTION (off by default).

KNOWN LIMITATION: Fyers generally does not serve 1-min history for EXPIRED
option contracts, so trades whose weekly contract has expired are skipped
("no option data" in the funnel). Longer windows need a different data source.

Config OVERRIDES: edit BacktestConfig below without touching the live bot.
"""
import os, sys, time, warnings, logging, requests, urllib3, pytz
import pandas as pd
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Optional, List

# ── Import EVERYTHING signal-related VERBATIM from the live bot ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_candle_poll_live import profit_lock_points
from first_candle_poll_live import (
    Config as LiveConfig,
    generate_today_events,
    expiry_code,
    get_next_tuesday,
)
from expiry_resolver import ExpiryResolver

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - FirstCandleBacktest - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("FirstCandleBacktest")


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST-SPECIFIC CONFIG OVERRIDES
#  Everything not listed here is inherited from first_candle_poll_live.Config.
# ══════════════════════════════════════════════════════════════════════════════
class BacktestConfig(LiveConfig):
    # ── Backtest window (calendar days back from today) ──
    HIST_DAYS = 30
    MAX_ENTRY_DELAY_MIN = 5         # skip entry if option bar at signal time is missing by more than this
    ENTRY_SLIPPAGE_PCT = 0.5         # % added to the option open to model the market-order fill

    EXIT_SLIPPAGE_PCT = 0.5          # % taken off the option price on bot-managed market exits (INDEX_POINTS mode)

    # ── Risk: RISK_MODE / SL_INDEX_POINTS / TP_INDEX_POINTS are inherited from the live Config.
    #    Override here to A/B, e.g.  SL_INDEX_POINTS = 25.0 ;  TP_INDEX_POINTS = 75.0
    #    PREMIUM_PCT mode: TP follows SL x RISK_REWARD. ──
    STOP_LOSS_PCT   = LiveConfig.STOP_LOSS_PCT
    RISK_REWARD     = LiveConfig.RISK_REWARD
    TAKE_PROFIT_PCT = STOP_LOSS_PCT * RISK_REWARD

    # ── Common A/B knobs (uncomment to override live defaults) ──
    # LEVEL_PROXIMITY_PCT = 0.10    # % of the level
    # ENABLE_FIRST_CANDLE_ENTRY = True   # False -> Setup A only
    # FIRST_CANDLE_FULL_BODY_FRAC = 0.85
    # BODY_RATIO_MAX      = 1.2
    # HEALTHY_BODY_FRAC   = 0.8
    # REQUIRE_HEALTHY_C1  = False
    # REQUIRE_HEALTHY_C3  = True
    # ENABLE_BUY          = True    # False -> PE-only
    # ENABLE_SELL         = True    # False -> CE-only
    # USE_EMA_FILTER      = False   # True -> CE only above EMA200, PE only below
    # ALTERNATE_DIRECTION = False
    # TRADE_START         = dtime(9, 19)
    # LOT = 65; MONEYNESS = "ITM"; TF_RULE = "1min"


# ---------- Fyers HTTP client ----------
class Fyers:
    T1 = "https://api-t1.fyers.in/api/v3"; T2 = "https://api-t1.fyers.in/data"

    def __init__(self, app, tok):
        auth = tok if ":" in tok else f"{app}:{tok}"
        self.s = requests.Session(); self.s.verify = False
        self.s.headers.update({"Authorization": auth})

    def profile(self):
        return self.s.get(f"{self.T1}/profile").json()

    # Fyers Standard plan allows 5 requests/second (error code -360 above that). Stay at ~4/s
    # and retry with back-off if Fyers still says "slow down".
    MIN_INTERVAL = 0.25
    MAX_RETRIES = 6
    _last_call = 0.0

    def _throttle(self):
        wait = self.MIN_INTERVAL - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def history(self, symbol, res, d_from, d_to):
        js = {}
        for attempt in range(self.MAX_RETRIES):
            self._throttle()
            r = self.s.get(f"{self.T2}/history", params={
                "symbol": symbol, "resolution": res, "date_format": "1",
                "range_from": d_from, "range_to": d_to, "cont_flag": "0"})
            try: js = r.json()
            except Exception: return {}
            if js.get("code") == -360 or "requests/second" in str(js.get("message", "")):
                time.sleep(1.0 + attempt)          # rate limited -> back off and retry
                continue
            return js
        return js


# ---------- backtest bot ----------
class FirstCandleBacktestBot:
    def __init__(self):
        self.cfg = BacktestConfig()
        self.fy = self._init_fyers()
        self.expiry_resolver = ExpiryResolver()
        self._expiry_cache: Dict[object, str] = {}
        self.skipped_ema = 0
        self.fetch_errors: Dict[str, str] = {}
        logger.info(
            f"BacktestConfig | TF={self.cfg.TF_RULE} | first-candle {self.cfg.FIRST_CANDLE_TIME.strftime('%H:%M')} "
            f"H/L levels (line prox={self.cfg.LEVEL_PROXIMITY_PCT}%) | "
            f"body_ratio<{self.cfg.BODY_RATIO_MAX}xC2 | "
            f"EMA={'EMA'+str(self.cfg.EMA_PERIOD) if self.cfg.USE_EMA_FILTER else 'off'} | "
            f"{self.cfg.risk_description()} | "
            f"BUY={self.cfg.ENABLE_BUY} SELL={self.cfg.ENABLE_SELL} | "
            f"alternate_dir={self.cfg.ALTERNATE_DIRECTION}"
        )

    def _init_fyers(self) -> Fyers:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
        app = open("fyers_appid.txt").read().strip()
        tok = open("fyers_token.txt").read().strip()
        fy = Fyers(app, tok); p = fy.profile()
        if p.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed -- re-run the token generator.\nResponse: {p}")
        logger.info(f"Fyers connected -- {p['data']['name']}")
        return fy

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end = datetime.now(IST); start = end - timedelta(days=days)
        resp = self.fy.history(symbol, self.cfg.BASE_RES,
                               start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if resp.get("s") != "ok":
            raise ValueError(f"History API error for {symbol}: {resp}")
        candles = resp.get("candles", [])
        if not candles:
            raise ValueError(f"No historical data for {symbol}")
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        return df

    def get_index_data(self) -> pd.DataFrame:
        c = self.cfg
        # Extra history before the window so EMA200 (if enabled) has a warm-up.
        fetch_days = c.HIST_DAYS + (c.EMA_LOOKBACK_DAYS if c.USE_EMA_FILTER else 0)
        end = datetime.now(IST)
        # Midnight of the first day: its 09:15 candle (the level) must not be cut off.
        backtest_start = IST.localize(datetime.combine(
            (end - timedelta(days=c.HIST_DAYS)).date(), dtime(0, 0)))
        df = self._fetch_history(c.UNDERLYING, fetch_days)
        df = df.iloc[df.index.indexer_between_time(c.DATA_START, c.SESSION_END)]
        if c.USE_EMA_FILTER:
            df[f"ema{c.EMA_PERIOD}"] = df["close"].ewm(span=c.EMA_PERIOD, adjust=False).mean()
        df = df.loc[df.index >= backtest_start].copy()
        logger.info(f"Loaded NIFTY50 1-min index data: {len(df)} rows across {c.HIST_DAYS} days | "
                    f"EMA{c.EMA_PERIOD}={'ON' if c.USE_EMA_FILTER else 'OFF'}")
        return df

    def _fetch_option_1min(self, symbol: str) -> pd.DataFrame:
        return self._fetch_history(symbol, self.cfg.HIST_DAYS).sort_index()

    def _expiry_for(self, d) -> str:
        if d not in self._expiry_cache:
            code, _src = self.expiry_resolver.resolve_expiry_code(d, expiry_code, get_next_tuesday)
            self._expiry_cache[d] = code
        return self._expiry_cache[d]

    def _opt_symbol(self, spot: float, direction: str, trade_date) -> str:
        c = self.cfg
        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{self._expiry_for(trade_date)}{strike:05d}{typ}"

    def _print_fetch_errors(self):
        if not self.fetch_errors:
            return
        rl = [s for s, m in self.fetch_errors.items() if "requests/second" in m or "-360" in m]
        other = {s: m for s, m in self.fetch_errors.items() if s not in rl}
        print(f"\nOption history could not be loaded for {len(self.fetch_errors)} symbol(s); "
              f"their signals were skipped.")
        if rl:
            print(f"  {len(rl)} symbol(s) still hit the Fyers rate limit after retries -- re-run the backtest.")
        for sym, msg in list(other.items())[:10]:
            print(f"  {sym}: {msg[:110]}")
        if other:
            print("  (Fyers generally has no 1-min history for EXPIRED weekly contracts, so signals from "
                  "earlier weeks usually cannot be priced.)")

    # ── Signal generation: per-day using the live bot's generate_today_events ──
    def generate_signals(self, df_1min: pd.DataFrame) -> List[dict]:
        c = self.cfg
        all_events: List[dict] = []
        skipped_ema = 0
        for _day, day_df in df_1min.groupby(df_1min.index.date):
            # The backtest owns the EMA gate (avoids double-filtering).
            ema_enabled = c.USE_EMA_FILTER
            c.USE_EMA_FILTER = False
            try:
                day_events = generate_today_events(day_df, c)
            finally:
                c.USE_EMA_FILTER = ema_enabled
            for ev in day_events:
                if c.USE_EMA_FILTER:
                    signal_bar_ts = ev["timestamp"] - pd.Timedelta(c.TF_RULE)
                    try:
                        row = day_df.loc[signal_bar_ts]
                        ema = float(row[f"ema{c.EMA_PERIOD}"]); spot = float(row["close"])
                    except (KeyError, TypeError, ValueError):
                        skipped_ema += 1; continue
                    if ev["dir"] == "SELL" and spot >= ema:
                        skipped_ema += 1; continue
                    if ev["dir"] == "BUY" and spot <= ema:
                        skipped_ema += 1; continue
                all_events.append(ev)
        self.skipped_ema = skipped_ema
        logger.info(f"Generated {len(all_events)} valid signal events across all days "
                    f"| EMA-skipped={skipped_ema}")
        return all_events

    # ── backtest engine (premium SL/TP 1:3, EOD, single position) ──
    def run_backtest(self) -> pd.DataFrame:
        c = self.cfg
        print(f"Starting first_candle_poll_live backtest ({c.risk_description()})...")
        df_1min = self.get_index_data()
        self.nifty_df = df_1min
        events = self.generate_signals(df_1min)
        if not events:
            print("No signals generated")
            return pd.DataFrame()
        events.sort(key=lambda e: e["timestamp"])
        for ev in events:
            ev["option_symbol"] = self._opt_symbol(ev["spot"], ev["dir"], ev["timestamp"].date())

        option_cache: Dict[str, pd.DataFrame] = {}

        def get_option_df(sym):
            if sym not in option_cache:
                try:
                    option_cache[sym] = self._fetch_option_1min(sym)
                except ValueError as e:
                    logger.warning(f"Skipping {sym}: {e}")
                    self.fetch_errors[sym] = str(e)
                    option_cache[sym] = pd.DataFrame()
            return option_cache[sym]

        def entry_bar(opt_df, ts):
            """Option bar starting at/after ts -> (open, bar_start) or (None, None)."""
            if opt_df.empty: return None, None
            pos = opt_df.index.searchsorted(ts, side="left")
            if pos >= len(opt_df): return None, None
            return float(opt_df.iloc[int(pos)]["open"]), opt_df.index[int(pos)]

        def eod_price(opt_df, eod_ts):
            """(price, ts, reached) for the EOD exit on eod_ts's day.
            Open of the bar at EOD time if present, else the last bar before it
            (reached=False when the data ends before EOD, e.g. a same-day run)."""
            day = eod_ts.date()
            day_df = opt_df[opt_df.index.date == day]
            if day_df.empty: return None, None, False
            at_or_after = day_df[day_df.index >= eod_ts]
            if not at_or_after.empty:
                return float(at_or_after.iloc[0]["open"]), at_or_after.index[0], True
            return float(day_df.iloc[-1]["close"]), day_df.index[-1], False

        trades = []
        has = False
        sl_abs = None
        lock_pts = 0.0
        symbol = direction = entry_price = entry_time = pattern = level = entry_spot = None
        last_trade_dir: Optional[str] = None
        current_day = None
        skipped_dir_alt = skipped_window = skipped_holding = skipped_no_data = 0

        def reset_pos():
            nonlocal has, symbol, direction, entry_price, entry_time, pattern, level, entry_spot
            has = False
            symbol = direction = entry_price = entry_time = pattern = level = entry_spot = None

        def record_trade(exit_time, exit_price, reason):
            pnl_pts = exit_price - entry_price
            trades.append({
                "direction": direction, "option_symbol": symbol, "pattern": pattern,
                "level": round(level, 2) if level is not None else None,
                "entry_spot": round(entry_spot, 2) if entry_spot is not None else None,
                "sl_spot": round(sl_abs, 2) if sl_abs is not None else None,
                "entry_time": entry_time, "entry_price": round(entry_price, 2),
                "exit_time": exit_time, "exit_price": round(exit_price, 2),
                "exit_reason": reason,
                "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * c.LOT, 2),
                "result": "profit" if pnl_pts > 0 else "loss",
            })

        def nifty_open_at(ts):
            nd = self.nifty_df
            pos = nd.index.searchsorted(ts, side="left")
            if pos < len(nd) and nd.index[pos] == ts:
                return float(nd.iloc[int(pos)]["open"])
            return None

        def option_at(opt_df, bar_ts):
            """(o, h, l, c) of the option bar starting at bar_ts, else the last bar before it
            (flat), else None."""
            pos = int(opt_df.index.searchsorted(bar_ts, side="left"))
            if pos < len(opt_df) and opt_df.index[pos] == bar_ts:
                r = opt_df.iloc[pos]
                return float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
            if pos > 0:
                px = float(opt_df.iloc[pos - 1]["close"])
                return px, px, px, px
            return None

        def idx_exit_option_price(opt_df, bar_ts, nb, kind, trigger):
            """Option price when NIFTY touched `trigger` inside bar `nb` (see module docstring)."""
            op = float(nb["open"]); hi = float(nb["high"]); lo = float(nb["low"])
            if direction == "BUY":
                ext = lo if kind == "SL" else hi
            else:
                ext = hi if kind == "SL" else lo
            denom = ext - op
            f = 1.0 if denom == 0 else min(1.0, max(0.0, (trigger - op) / denom))
            ob = option_at(opt_df, bar_ts)
            if ob is None: return None
            o, h, l, _cl = ob
            opt_ext = l if kind == "SL" else h        # long option: loss -> its low, win -> its high
            return (o + f * (opt_ext - o)) * (1 - c.EXIT_SLIPPAGE_PCT / 100.0)

        def manage_position_index(limit_ts):
            """INDEX_POINTS mode: walk NIFTY 1-min bars in [entry_time, min(limit_ts, entry-day EOD))."""
            nonlocal lock_pts
            if not has: return
            opt_df = get_option_df(symbol)
            entry_eod_ts = IST.localize(datetime.combine(entry_time.date(), c.EOD_EXIT_TIME))
            window_end = min(limit_ts, entry_eod_ts)
            nd = self.nifty_df
            window = nd[(nd.index >= entry_time) & (nd.index < window_end)]
            for bar_ts, nb in window.iterrows():
                hi = float(nb["high"]); lo = float(nb["low"])
                kind = "stop_loss"
                if direction == "BUY":
                    sl_lvl = sl_abs if sl_abs is not None else entry_spot - c.SL_INDEX_POINTS
                    if lock_pts > 0: sl_lvl, kind = entry_spot + lock_pts, "profit_lock"
                    tp_lvl = (entry_spot + c.TP_INDEX_POINTS) if c.TP_INDEX_POINTS else None
                    sl_hit = lo <= sl_lvl; tp_hit = tp_lvl is not None and hi >= tp_lvl
                else:
                    sl_lvl = sl_abs if sl_abs is not None else entry_spot + c.SL_INDEX_POINTS
                    if lock_pts > 0: sl_lvl, kind = entry_spot - lock_pts, "profit_lock"
                    tp_lvl = (entry_spot - c.TP_INDEX_POINTS) if c.TP_INDEX_POINTS else None
                    sl_hit = hi >= sl_lvl; tp_hit = tp_lvl is not None and lo <= tp_lvl
                if (sl_abs is not None or c.SL_INDEX_POINTS or lock_pts > 0) and sl_hit:     # SL first (conservative)
                    px = idx_exit_option_price(opt_df, bar_ts, nb, "SL", sl_lvl)
                    if px is not None:
                        record_trade(bar_ts, px, kind); reset_pos(); return
                if c.TP_INDEX_POINTS and tp_hit:
                    px = idx_exit_option_price(opt_df, bar_ts, nb, "TP", tp_lvl)
                    if px is not None:
                        record_trade(bar_ts, px, "target"); reset_pos(); return
                if c.PROFIT_LOCK_ENABLED:                           # ratchet from this (completed) bar
                    fav = (hi - entry_spot) if direction == "BUY" else (entry_spot - lo)
                    lock_pts = max(lock_pts, profit_lock_points(fav, c))
            if limit_ts >= entry_eod_ts:
                px, pxt, reached = eod_price(opt_df, entry_eod_ts)
                if px is not None:
                    record_trade(pxt, px * (1 - c.EXIT_SLIPPAGE_PCT / 100.0),
                                 "eod_exit" if reached else "backtest_end")
                    reset_pos()

        def manage_position(limit_ts):
            if c.RISK_MODE == "INDEX_POINTS":
                return manage_position_index(limit_ts)
            return manage_position_premium(limit_ts)

        def manage_position_premium(limit_ts):
            """PREMIUM_PCT mode: walk the option's bars in [entry_time, min(limit_ts, entry-day EOD))
            for SL/TP; if limit_ts reaches EOD without one, exit at EOD."""
            if not has: return
            opt_df = get_option_df(symbol)
            entry_eod_ts = IST.localize(datetime.combine(entry_time.date(), c.EOD_EXIT_TIME))
            window_end = min(limit_ts, entry_eod_ts)
            sl_price = entry_price * (1 - c.STOP_LOSS_PCT / 100.0)
            tp_price = entry_price * (1 + c.TAKE_PROFIT_PCT / 100.0)
            window = opt_df[(opt_df.index >= entry_time) & (opt_df.index < window_end)]
            for bar_ts, obar in window.iterrows():
                if c.USE_PREMIUM_SL and float(obar["low"]) <= sl_price:      # SL first (conservative)
                    record_trade(bar_ts, sl_price, "stop_loss"); reset_pos(); return
                if c.USE_PREMIUM_TP and float(obar["high"]) >= tp_price:
                    record_trade(bar_ts, tp_price, "target"); reset_pos(); return
            if limit_ts >= entry_eod_ts:
                px, pxt, reached = eod_price(opt_df, entry_eod_ts)
                if px is not None:
                    record_trade(pxt, px, "eod_exit" if reached else "backtest_end")
                    reset_pos()

        max_delay = timedelta(minutes=c.MAX_ENTRY_DELAY_MIN)
        for ev in events:
            ts = ev["timestamp"]
            if current_day is None or ts.date() != current_day:
                current_day = ts.date()
                last_trade_dir = None

            manage_position(ts)      # closes SL/TP/EOD that happened BEFORE this signal
            if has:
                skipped_holding += 1
                continue
            if c.ALTERNATE_DIRECTION and last_trade_dir is not None and ev["dir"] == last_trade_dir:
                skipped_dir_alt += 1
                continue
            if not (c.TRADE_START <= ts.time() <= c.TRADE_END):
                skipped_window += 1
                continue

            sym = ev["option_symbol"]
            px, pxt = entry_bar(get_option_df(sym), ts)
            if px is None or pxt - ts > max_delay or pxt.date() != ts.date():
                skipped_no_data += 1
                continue
            has = True
            symbol = sym; direction = ev["dir"]; pattern = ev["name"]; level = ev.get("level")
            entry_price = px * (1 + c.ENTRY_SLIPPAGE_PCT / 100.0)
            entry_time = pxt
            entry_spot = nifty_open_at(pxt)
            if entry_spot is None:
                entry_spot = ev["spot"]          # NIFTY bar missing at entry -> signal candle close
            last_trade_dir = ev["dir"]
            # absolute NIFTY stop: C1 low - offset (CE) / C1 high + offset (PE); None -> fixed SL points
            sl_abs = None; lock_pts = 0.0
            if c.RISK_MODE == "INDEX_POINTS" and c.SL_MODE == "C1_OFFSET" and ev.get("sl_ref") is not None:
                sl_abs = ev["sl_ref"] - c.C1_SL_OFFSET_POINTS if ev["dir"] == "BUY" else ev["sl_ref"] + c.C1_SL_OFFSET_POINTS

        # Last trade: walk it bar-by-bar through its own day (SL/TP/EOD), not just flush it.
        if has:
            manage_position(IST.localize(datetime.combine(entry_time.date(), c.EOD_EXIT_TIME)))
        if has:   # still open -> no data covering EOD
            opt_df = get_option_df(symbol)
            if not opt_df.empty:
                record_trade(opt_df.index[-1], float(opt_df.iloc[-1]["close"]), "backtest_end")
                reset_pos()

        trades_df = pd.DataFrame(trades)
        funnel = (f"signals={len(events)}, skipped_ema={self.skipped_ema}, "
                  f"skipped_dir_alt={skipped_dir_alt}, skipped_window={skipped_window}, "
                  f"skipped_holding={skipped_holding}, skipped_no_option_data={skipped_no_data}")
        if trades_df.empty:
            print("No trades executed")
            print(f"\n(For diagnostics: {funnel})")
            self._print_fetch_errors()
            return trades_df

        trades_df["cum_pnl_rs"] = trades_df["pnl_rs"].cumsum()
        total = len(trades_df)
        wins = int((trades_df["result"] == "profit").sum())
        losses = int((trades_df["result"] == "loss").sum())
        net = trades_df["pnl_rs"].sum()
        wr = round(wins / total * 100, 1)
        gross_win = trades_df.loc[trades_df["pnl_rs"] > 0, "pnl_rs"].sum()
        gross_loss = -trades_df.loc[trades_df["pnl_rs"] < 0, "pnl_rs"].sum()
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
        max_dd = float((trades_df["cum_pnl_rs"].cummax() - trades_df["cum_pnl_rs"]).max())
        if c.RISK_MODE == "INDEX_POINTS":
            if c.SL_MODE == "C1_OFFSET" and "sl_spot" in trades_df and trades_df["sl_spot"].notna().any():
                risk = (trades_df["entry_spot"] - trades_df["sl_spot"]).abs().dropna()
                avg_risk = float(risk.mean())
                rr = (c.TP_INDEX_POINTS / avg_risk) if c.TP_INDEX_POINTS else float("nan")
                sl_txt = f"C1 extreme -/+ {c.C1_SL_OFFSET_POINTS:g} (avg risk {avg_risk:.0f} NIFTY pts)"
            else:
                rr = (c.TP_INDEX_POINTS / c.SL_INDEX_POINTS) if c.TP_INDEX_POINTS else float("nan")
                sl_txt = f"{c.SL_INDEX_POINTS:g} NIFTY pts"
            tp_txt = (f"{c.TP_INDEX_POINTS:g} NIFTY pts" if c.TP_INDEX_POINTS else "no fixed target") + \
                     (" + profit-lock" if c.PROFIT_LOCK_ENABLED else "")
            exit_slip = c.EXIT_SLIPPAGE_PCT
        else:
            rr = c.TAKE_PROFIT_PCT / c.STOP_LOSS_PCT
            sl_txt, tp_txt = f"-{c.STOP_LOSS_PCT:g}% premium", f"+{c.TAKE_PROFIT_PCT:g}% premium"
            exit_slip = 0.0
        breakeven_wr = round(100.0 / (1 + rr), 1) if rr == rr else float("nan")

        csv_path = "FirstCandle_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")
        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Break-even Win Rate (%),{breakeven_wr}",
                f"Net PnL (Rs),{net:.2f}", f"Profit Factor,{pf}", f"Max Drawdown (Rs),{max_dd:.2f}",
                f"TF Rule,{c.TF_RULE}",
                f"First Candle,{c.FIRST_CANDLE_TIME.strftime('%H:%M')}",
                f"Line Proximity %,{c.LEVEL_PROXIMITY_PCT}",
                f"First-candle Full-body Entry,{c.ENABLE_FIRST_CANDLE_ENTRY} (>= {c.FIRST_CANDLE_FULL_BODY_FRAC})",
                f"Body Ratio Max,{c.BODY_RATIO_MAX}",
                f"Healthy Body Frac,{c.HEALTHY_BODY_FRAC}",
                f"Require Healthy C1,{c.REQUIRE_HEALTHY_C1}",
                f"Require Healthy C3,{c.REQUIRE_HEALTHY_C3}",
                f"EMA Filter,{c.USE_EMA_FILTER}",
                f"BUY Enabled,{c.ENABLE_BUY}", f"SELL Enabled,{c.ENABLE_SELL}",
                f"Risk Mode,{c.RISK_MODE}", f"SL,{sl_txt}", f"TP,{tp_txt}",
                f"Risk Reward,1:{rr:.2f}", f"Exit Slippage %,{exit_slip}",
                f"Alternate Direction,{c.ALTERNATE_DIRECTION}",
                f"Signals Generated,{len(events)}",
                f"Skipped EMA,{self.skipped_ema}",
                f"Skipped Dir-Alt,{skipped_dir_alt}",
                f"Skipped Window,{skipped_window}",
                f"Skipped Holding,{skipped_holding}",
                f"Skipped No Option Data,{skipped_no_data}",
                f"Entry Slippage %,{c.ENTRY_SLIPPAGE_PCT}",
                f"Lot Size,{c.LOT}", f"Hist Days,{c.HIST_DAYS}",
            ]:
                f.write(line + "\n")

        by_reason = trades_df.groupby("exit_reason")["pnl_rs"].agg(["count", "sum"]).reset_index()
        by_dir = trades_df.groupby("direction")["pnl_rs"].agg(["count", "sum"]).reset_index()
        print(f"\n{'='*62}")
        print(f"  Contract Type       : {c.MONEYNESS}  |  TF: {c.TF_RULE}")
        print(f"  Risk (SL / TP)      : {sl_txt} / {tp_txt}  (RR 1:{rr:.2f}, mode {c.RISK_MODE})")
        print(f"  Levels              : {c.FIRST_CANDLE_TIME.strftime('%H:%M')} candle H/L  (line prox={c.LEVEL_PROXIMITY_PCT}%)")
        print(f"  EMA Filter          : {c.USE_EMA_FILTER}  (period={c.EMA_PERIOD})")
        print(f"  Total Trades        : {total}   (W {wins} / L {losses})")
        print(f"  Win Rate            : {wr}%   " + (f"(break-even at RR 1:{rr:.2f} = {breakeven_wr}%, before costs)"
              if rr == rr else "(no fixed target: break-even depends on the trailed exits)"))
        print(f"  Net PnL             : Rs {net:,.2f}")
        print(f"  Profit Factor       : {pf}   |  Max Drawdown: Rs {max_dd:,.2f}")
        print(f"  Best / Worst        : Rs {trades_df['pnl_rs'].max():,.2f} / Rs {trades_df['pnl_rs'].min():,.2f}")
        print(f"{'-'*62}")
        print("  Signal funnel:")
        print(f"    Signals generated        : {len(events)}")
        print(f"    Skipped EMA              : {self.skipped_ema}")
        print(f"    Skipped dir-alternation  : {skipped_dir_alt}")
        print(f"    Skipped trade window     : {skipped_window}")
        print(f"    Skipped (in-position)    : {skipped_holding}")
        print(f"    Skipped (no option data) : {skipped_no_data}")
        print(f"    Actual entries           : {total}")
        print("  Exit-reason breakdown:")
        for _, row in by_reason.iterrows():
            print(f"    {row['exit_reason']:<15} count={int(row['count']):<3}  sum=Rs {row['sum']:>12,.2f}")
        print("  Direction breakdown:")
        for _, row in by_dir.iterrows():
            print(f"    {str(row['direction']):<8} count={int(row['count']):<3}  sum=Rs {row['sum']:>12,.2f}")
        print(f"{'='*62}")
        self._print_fetch_errors()
        print(f"Results saved -> {csv_path}")
        return trades_df


# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    bot = FirstCandleBacktestBot()
    trades = bot.run_backtest()

