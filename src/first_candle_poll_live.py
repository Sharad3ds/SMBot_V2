"""
FIRST-CANDLE HIGH/LOW support-resistance REVERSAL LIVE polling bot (NIFTY options).

Built from pattern_poll_live.py: broker layer (FyersHTTP, OrderManager, state
persistence, broker-side SL/TP, EOD exit) is unchanged. Only the signal filter,
the data window and the risk numbers differ.

LEVELS (recomputed from the day's 09:15 1-minute candle):
  GREEN line = high of the day's first 1-min candle
  RED   line = low  of the day's first 1-min candle

SETUP A -- 3-candle reversal at a line (C1, C2, C3 = confirmation/entry):
  CE BUY  at the GREEN line (high acts as SUPPORT on a pullback):
          C1 red, C2 green, C3 green; fires if EITHER (a) closes rise
          C1 < C2 < C3, OR (b) C1 body < BODY_RATIO_MAX x C2 body.
  PE BUY  at the RED line (low acts as RESISTANCE on a retest from below):
          C1 green, C2 red, C3 red; fires if EITHER (a) closes fall
          C1 > C2 > C3, OR (b) C1 body < BODY_RATIO_MAX x C2 body.
  C3 must be healthy (body >= HEALTHY_BODY_FRAC of its range); C1 need not be
  (REQUIRE_HEALTHY_C1 / REQUIRE_HEALTHY_C3 flags are in Config).
  Line test: for a CE the pullback low (lowest low of C1/C2) must be within
  LEVEL_PROXIMITY_PCT of the green line; for a PE the retest high (highest high
  of C1/C2) must be within LEVEL_PROXIMITY_PCT of the red line.
  C3 must close on the correct side of the line (above green / below red).
  The first candle itself is never part of the pattern (C1 is the 2nd candle or later).

SETUP B -- full-body first candle (immediate entry): if the 09:15 candle's body
  is >= FIRST_CANDLE_FULL_BODY_FRAC of its range, enter at the very next candle
  in the candle's direction: GREEN -> CE buy, RED -> PE buy. (Single position,
  so a later Setup A signal is ignored while this trade is open.)

PROFIT LOCK (INDEX_POINTS mode): once NIFTY is +50 pts in favour the stop moves to entry +10; +60 -> +20;
  +70 -> +30 ... (ratchet, never lowered). TP_INDEX_POINTS = None means no fixed target. Config: PROFIT_LOCK_*.

RISK (RISK_MODE = "INDEX_POINTS", default): TP = 80 NIFTY INDEX POINTS from the entry spot.
  SL_MODE = "C1_OFFSET" (default): stop = C1 candle low - 40 pts (CE) / C1 high + 40 pts (PE);
  SL_MODE = "FIXED": stop = SL_INDEX_POINTS from the entry spot. Description below is the FIXED form:
    CE: stop when NIFTY <= entry - SL, target when NIFTY >= entry + TP
    PE: stop when NIFTY >= entry + SL, target when NIFTY <= entry - TP
  Fyers cannot attach index-point stops to an option position, so the BOT watches
  the NIFTY index (polling every POSITION_POLL_INTERVAL_SECS while a trade is open)
  and exits with DELETE /api/v3/positions. NO broker-side SL/TP is attached in this
  mode: if the bot process stops, the position is unprotected until you act.
  RISK_MODE = "PREMIUM_PCT" restores the old option-premium % SL/TP attached
  broker-side via PATCH /api/v3/positions (legType=2).
  EOD close (15:20) always uses DELETE /api/v3/positions.

Single position at a time. Signals older than MAX_SIGNAL_AGE_MIN are discarded
(prevents entering a stale signal after a mid-day restart).
DRY_RUN defaults to True for this new strategy -- flip it deliberately.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
import requests
import logging
import os
from typing import Optional, List
import warnings
import pytz
import calendar
import time
import json
import urllib3
from expiry_resolver import ExpiryResolver

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
IST = pytz.timezone("Asia/Kolkata")

STATE_FILE = "firstcandle_live_position_state.json"

def ist_time(*args):
    return datetime.now(IST).timetuple()
logging.Formatter.converter = ist_time

try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - FirstCandlePoll - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("FirstCandlePoll")

# --------------------------- EXPIRY HELPERS (identical convention) ---------------------------
def get_next_tuesday(now=None):
    now = now or datetime.now(IST)
    x = datetime(now.year, now.month, now.day)
    while x.weekday() != 1: x += timedelta(days=1)
    return x

def monthly_expiry(y, m):
    last = calendar.monthrange(y, m)[1]; x = datetime(y, m, last)
    while x.weekday() != 1: x -= timedelta(days=1)
    return x

def expiry_code(dt):
    yy = dt.year % 100; me = monthly_expiry(dt.year, dt.month)
    if dt.date() == me.date(): return f"{yy:02d}{dt.strftime('%b').upper()}"
    mc = str(dt.month) if dt.month <= 9 else {10: 'O', 11: 'N', 12: 'D'}[dt.month]
    return f"{yy:02d}{mc}{dt.day:02d}"

# --------------------------- CONFIG (matches pattern_backtest_swingzone.py) ---------------------------
class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"
    TF_RULE = "1min"

    # ── 3-candle reversal pattern (same definition as pattern_poll_live.py) ──
    HEALTHY_BODY_FRAC = 0.65       # a "healthy" candle: body >= this fraction of its high-low range
    REQUIRE_HEALTHY_C1 = False     # C1 need NOT be healthy (set True to require it)
    REQUIRE_HEALTHY_C2 = True      # C2 must be healthy
    REQUIRE_HEALTHY_C3 = False     # C3 need NOT be healthy (set True to require it)
    # BREAKOUT-RETEST filter: the line must first be broken, then retested.
    #   CE: before C1, price must have traded at least RETEST_MIN_BREAK_POINTS ABOVE the green line
    #       (broke out), then pulled back to the line (C1/C2 touch it) and bounced (support).
    #   PE: before C1, price must have traded at least that far BELOW the red line, then rallied back to it.
    # Stops the bot buying the very first candles after the open, before any breakout/pullback exists.
    REQUIRE_PRIOR_BREAKOUT = True
    RETEST_MIN_BREAK_POINTS = 30.0
    # LINE-TOUCH rule: at least one of C1 / C2 / C3 must actually TOUCH the line (its high-low range
    # reaches the line, +/- tolerance): CE -> the green line, PE -> the red line.
    # When True this replaces the looser LEVEL_PROXIMITY_PCT band check on C1/C2.
    REQUIRE_LINE_TOUCH = True 
    TOUCH_TOLERANCE_POINTS = 0.0
    # C1 must have very little / no wick: (upper wick + lower wick) <= this fraction of C1's
    # high-low range (0.0 = no wick at all, 0.15 = body is at least 85% of the range). None = off.
    C1_MAX_WICK_FRAC = None        # OFF for now (set e.g. 0.15 to require a near wick-less C1)
    CONFIRM_MUST_BE_COLORED = True # PE: C3 red / CE: C3 green
    ENABLE_BUY = True              # CE trades at the green line (first-candle high)
    ENABLE_SELL = True             # PE trades at the red line (first-candle low)
    BODY_RATIO_MAX = 0.8           # fires if C1 body < this x C2 body (OR three rising/falling closes)

    # ── first-candle levels ──
    FIRST_CANDLE_TIME = dtime(9, 15)   # bar whose high/low define the green/red lines
    LEVEL_PROXIMITY_PCT = 0.10         # how close (% of level, ~25 pts) the C1/C2 extreme must be to the line

    # Setup B: full-body 09:15 candle -> immediate entry on the next candle in its direction.
    ENABLE_FIRST_CANDLE_ENTRY = True
    FIRST_CANDLE_FULL_BODY_FRAC = 0.85 # 09:15 body >= this fraction of its high-low range = "full body"

    # Optional 200 EMA direction filter (not part of the PPT strategy -> OFF).
    USE_EMA_FILTER = False
    EMA_PERIOD = 200
    EMA_LOOKBACK_DAYS = 10

    ALTERNATE_DIRECTION = True     # old bot forced CE/PE alternation; the PPT strategy does not
    MAX_SIGNAL_AGE_MIN = 2         # ignore signals whose C3 closed longer ago than this

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"

    DATA_START = dtime(9, 15)      # must include the 09:15 candle (old bot cut bars before 09:20)
    SESSION_START = dtime(9, 15); SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 16)   # Setup B fires when the 09:15 candle closes (09:16); Setup A cannot fire before 09:19
    TRADE_START = dtime(9, 16); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    # ── risk management ──
    # "INDEX_POINTS": SL/TP in NIFTY index points from the entry spot; the bot watches the
    #                 index and exits itself (no broker-side SL/TP possible for index points).
    # "PREMIUM_PCT" : option-premium % SL/TP attached broker-side (previous behaviour).
    RISK_MODE = "INDEX_POINTS"
    SL_INDEX_POINTS = 30.0         # CE: NIFTY falls this far below entry / PE: rises this far above
    TP_INDEX_POINTS = 50.0         # fixed target in index points (CE: above entry / PE: below). None = NO fixed
                                   # target (exit only by stop / profit-lock stop / EOD) -- needed for an open-ended lock ladder
    # PROFIT LOCK (index points, ratchets only upward): once NIFTY has moved PROFIT_LOCK_START_POINTS in your
    # favour the stop is moved to entry +PROFIT_LOCK_FIRST_PROFIT; every further PROFIT_LOCK_STEP_POINTS of
    # favourable move lifts the locked profit by PROFIT_LOCK_STEP_POINTS.
    #   move 50 -> lock +10 | 60 -> +20 | 70 -> +30 | 80 -> +40 ...  (PE: mirrored, stop moves DOWN)
    PROFIT_LOCK_ENABLED = False
    PROFIT_LOCK_START_POINTS = 50.0
    PROFIT_LOCK_FIRST_PROFIT = 10.0
    PROFIT_LOCK_STEP_POINTS = 10.0
    # SL placement in INDEX_POINTS mode:
    #   "C1_OFFSET": CE stop = C1 candle LOW - C1_SL_OFFSET_POINTS ; PE stop = C1 candle HIGH + offset
    #                (full-body 09:15 entry has no C1: the 09:15 candle's low/high is used)
    #   "FIXED"    : stop = SL_INDEX_POINTS from the entry spot (old behaviour)
    SL_MODE = "C1_OFFSET"
    C1_SL_OFFSET_POINTS = 40.0
    POSITION_POLL_INTERVAL_SECS = 3    # faster polling while a trade is open (exit latency)

    # PREMIUM_PCT mode only (TP = SL x RISK_REWARD):
    USE_PREMIUM_SL = True
    STOP_LOSS_PCT = 10.0
    RISK_REWARD = 3.0
    USE_PREMIUM_TP = True
    TAKE_PROFIT_PCT = STOP_LOSS_PCT * RISK_REWARD   # 30%

    POLL_INTERVAL_SECS = 20
    DRY_RUN = True

    # Fyers V3 broker-side protection is attached after the INTRADAY market
    # order is filled/visible in the Positions API.
    PROTECTION_RETRIES = 10
    PROTECTION_RETRY_DELAY_SECS = 0.5

    def tp_text(self) -> str:
        return f"TP={self.TP_INDEX_POINTS:g}" if self.TP_INDEX_POINTS else "no fixed target"

    def lock_text(self) -> str:
        if not (self.RISK_MODE == "INDEX_POINTS" and self.PROFIT_LOCK_ENABLED):
            return ""
        return (f" | profit-lock: +{self.PROFIT_LOCK_FIRST_PROFIT:g} @ {self.PROFIT_LOCK_START_POINTS:g}, "
                f"then +{self.PROFIT_LOCK_STEP_POINTS:g} per {self.PROFIT_LOCK_STEP_POINTS:g} pts")

    def risk_description(self) -> str:
        if self.RISK_MODE == "INDEX_POINTS":
            if self.SL_MODE == "C1_OFFSET":
                return (f"SL=C1 extreme -/+ {self.C1_SL_OFFSET_POINTS:g} pts / {self.tp_text()} "
                        f"NIFTY index points{self.lock_text()}")
            rr = (f" (RR 1:{self.TP_INDEX_POINTS / self.SL_INDEX_POINTS:.2f})"
                  if self.TP_INDEX_POINTS and self.SL_INDEX_POINTS else "")
            return f"SL={self.SL_INDEX_POINTS:g} / {self.tp_text()} NIFTY index points{rr}{self.lock_text()}"
        return (f"SL=-{self.STOP_LOSS_PCT}% TP=+{self.TAKE_PROFIT_PCT}% option premium "
                f"(RR 1:{self.RISK_REWARD:g})")

    def __init__(self):
        exp_dt = get_next_tuesday()
        self.EXPIRY_CODE = expiry_code(exp_dt)
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | First-candle H/L reversal "
                    f"(C1/C2/C3, {self.TF_RULE} bars, line prox={self.LEVEL_PROXIMITY_PCT}%) | "
                    f"first-candle full-body entry={'ON(>='+str(self.FIRST_CANDLE_FULL_BODY_FRAC)+')' if self.ENABLE_FIRST_CANDLE_ENTRY else 'off'} | "
                    f"EMA={'EMA'+str(self.EMA_PERIOD) if self.USE_EMA_FILTER else 'off'} | "
                    f"{self.risk_description()} | "
                    f"live_exit={('bot-managed index exits + DELETE /positions' if self.RISK_MODE == 'INDEX_POINTS' else 'DELETE /positions + broker SL/TP') if not self.DRY_RUN else 'dry-run sim'} | "
                    f"alternate_dir={self.ALTERNATE_DIRECTION} | DRY_RUN={self.DRY_RUN}")

# --------------------------- FYERS HTTP CLIENT ---------------------------
class FyersHTTP:
    T1 = "https://api-t1.fyers.in/api/v3"; T2 = "https://api-t1.fyers.in/data"

    def __init__(self, app, tok):
        auth = tok if ":" in tok else f"{app}:{tok}"
        self.s = requests.Session(); self.s.verify = False
        self.s.headers.update({"Authorization": auth})

    def profile(self): return self.s.get(f"{self.T1}/profile").json()

    def history(self, symbol, res, d_from, d_to):
        r = self.s.get(f"{self.T2}/history", params={
            "symbol": symbol, "resolution": res, "date_format": "1",
            "range_from": d_from, "range_to": d_to, "cont_flag": "0"})
        try: return r.json()
        except Exception: return {}

    def quotes(self, symbol: str) -> dict:
        try:
            return self.s.get(f"{self.T2}/quotes", params={"symbols": symbol}).json()
        except Exception:
            return {}

    def place_order(self, data: dict) -> dict:
        """POST /api/v3/orders/sync -- documented Fyers V3 sync order API."""
        try:
            return self.s.post(f"{self.T1}/orders/sync", json=data).json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def exit_order(self, data: dict) -> dict:
        """Cancel a pending order via DELETE /api/v3/orders/sync/{id}."""
        try:
            r = self.s.delete(f"{self.T1}/orders/sync/{data['id']}")
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def exit_position(self, position_id: str) -> dict:
        """Close a specific open position via DELETE /api/v3/positions."""
        try:
            r = self.s.delete(f"{self.T1}/positions", json={"id": position_id})
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def set_stop_loss_take_profit(
        self,
        position_id: str,
        stop_loss: float,
        take_profit: float,
        qty: int,
        leg_type: int = 2,
    ) -> dict:
        """PATCH /api/v3/positions -- attach SL/TP to an open position.

        Fyers V3 documents legType=1 as offset and legType=2 as percentage.
        This bot uses percentage protection so the configured STOP_LOSS_PCT and
        TAKE_PROFIT_PCT can be applied without depending on the exact fill price.
        """
        try:
            payload = {
                "positionId": position_id,
                "takeProfit": float(take_profit),
                "stopLoss": float(stop_loss),
                "qty": int(qty),
                "legType": int(leg_type),
            }
            r = self.s.patch(f"{self.T1}/positions", json=payload)
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def positions(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/positions").json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def get_position(self, symbol: str) -> Optional[dict]:
        """Return the current INTRADAY position for symbol, if present."""
        resp = self.positions()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return None
        for pos in resp.get("netPositions", []):
            if pos.get("symbol") == symbol and pos.get("productType") == "INTRADAY":
                return pos
        return None

    def get_order(self, order_id: str) -> Optional[dict]:
        """Return a current-day order by exchange order id."""
        if not order_id:
            return None
        resp = self.orderbook()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return None
        for order in resp.get("orderBook", []):
            if order.get("id") == order_id:
                return order
        return None

    def orderbook(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/orders").json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def tradebook(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/trades").json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

# --------------------------- FIRST-CANDLE LEVELS + REVERSAL DETECTION ---------------------------
def get_first_candle_levels(bars: pd.DataFrame, cfg):
    """(green_line, red_line) = (high, low) of the day's 09:15 1-min bar, else None.
    Returns None if the first bar is not the 09:15 bar (e.g. missing data), so the
    bot stays out rather than using the wrong candle."""
    if bars is None or bars.empty:
        return None
    if bars.index[0].time() != cfg.FIRST_CANDLE_TIME:
        return None
    return float(bars["high"].iloc[0]), float(bars["low"].iloc[0])


def within_band(value, level, pct):
    band = level * pct / 100.0
    return (level - band) <= value <= (level + band)


def profit_lock_points(move, cfg):
    """Profit (index points above/below entry) the stop is locked at once NIFTY has moved `move` points
    in our favour. 0.0 = no lock yet. Ratchet is applied by the caller (it only ever increases)."""
    if not cfg.PROFIT_LOCK_ENABLED or move < cfg.PROFIT_LOCK_START_POINTS:
        return 0.0
    steps = int((move - cfg.PROFIT_LOCK_START_POINTS + 1e-9) // cfg.PROFIT_LOCK_STEP_POINTS)
    return cfg.PROFIT_LOCK_FIRST_PROFIT + steps * cfg.PROFIT_LOCK_STEP_POINTS


def detect_signal(o, h, l, c, cfg, levels):
    """Last bar (index i) is C3. Returns ("BUY"|"SELL", name, c1, c2, level) or None.
    BUY (CE) only at the green line, SELL (PE) only at the red line."""
    n = len(c)
    if n < 4: return None            # first candle + C1,C2,C3
    i = n - 1
    c1, c2 = i - 2, i - 1
    if c1 < 1: return None           # first candle is the level, not part of the pattern
    green_line, red_line = levels
    green = lambda k: c[k] > o[k]
    red   = lambda k: c[k] < o[k]
    body  = lambda k: abs(c[k] - o[k])

    def healthy(k):
        rng = h[k] - l[k]
        return rng > 0 and body(k) >= cfg.HEALTHY_BODY_FRAC * rng

    def c1_wick_ok():
        if cfg.C1_MAX_WICK_FRAC is None:
            return True
        rng = h[c1] - l[c1]
        return rng > 0 and (rng - body(c1)) <= cfg.C1_MAX_WICK_FRAC * rng + 1e-9

    healthy_c1_c3 = ((not cfg.REQUIRE_HEALTHY_C1 or healthy(c1)) and
                     (not cfg.REQUIRE_HEALTHY_C2 or healthy(c2)) and
                     (not cfg.REQUIRE_HEALTHY_C3 or healthy(i)) and c1_wick_ok())

    broke_below = broke_above = True
    if cfg.REQUIRE_PRIOR_BREAKOUT:
        pts = cfg.RETEST_MIN_BREAK_POINTS
        broke_below = c1 > 1 and float(np.min(l[1:c1])) <= red_line - pts
        broke_above = c1 > 1 and float(np.max(h[1:c1])) >= green_line + pts

    def touches(line):
        t = cfg.TOUCH_TOLERANCE_POINTS
        return any((l[k] - t) <= line <= (h[k] + t) for k in (c1, c2, i))

    # ---- PE: retest of the RED line (first-candle low) from below -> rejection ----
    sell_confirm_ok = (not cfg.CONFIRM_MUST_BE_COLORED) or red(i)
    if cfg.ENABLE_SELL and green(c1) and red(c2) and sell_confirm_ok and healthy_c1_c3 and broke_below:
        cond_a = c[c1] > c[c2] > c[i]
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)
        retest_high = max(h[c1], h[c2])
        near = touches(red_line) if cfg.REQUIRE_LINE_TOUCH else within_band(retest_high, red_line, cfg.LEVEL_PROXIMITY_PCT)
        if (cond_a or cond_b) and near and c[i] < red_line:
            return ("SELL", "red_line_pe", c1, c2, red_line)

    # ---- CE: pullback to the GREEN line (first-candle high) -> support holds ----
    buy_confirm_ok = (not cfg.CONFIRM_MUST_BE_COLORED) or green(i)
    if cfg.ENABLE_BUY and red(c1) and green(c2) and buy_confirm_ok and healthy_c1_c3 and broke_above:
        cond_a = c[c1] < c[c2] < c[i]
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)
        pullback_low = min(l[c1], l[c2])
        near = touches(green_line) if cfg.REQUIRE_LINE_TOUCH else within_band(pullback_low, green_line, cfg.LEVEL_PROXIMITY_PCT)
        if (cond_a or cond_b) and near and c[i] > green_line:
            return ("BUY", "green_line_ce", c1, c2, green_line)

    return None


def generate_today_events(day_df: pd.DataFrame, cfg: Config) -> List[dict]:
    """Rebuilds signal events from today's closed bars (causal, re-derived every poll)."""
    r = day_df.resample(cfg.TF_RULE).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    levels = get_first_candle_levels(r, cfg)
    if levels is None or len(r) < 1:
        return []

    o = r["open"].to_numpy(); h = r["high"].to_numpy()
    l = r["low"].to_numpy(); cl = r["close"].to_numpy()

    ema200 = None
    if cfg.USE_EMA_FILTER and "ema200" in day_df.columns:
        ema200 = day_df["ema200"].reindex(r.index)

    events: List[dict] = []

    def ema_ok(i, direction):
        if not cfg.USE_EMA_FILTER:
            return True
        ema_value = ema200.iloc[i] if ema200 is not None else np.nan
        if pd.isna(ema_value):
            return False
        if direction == "SELL" and cl[i] >= float(ema_value):
            return False
        if direction == "BUY" and cl[i] <= float(ema_value):
            return False
        return True

    # Setup B: full-body 09:15 candle -> immediate entry at the next candle, in its direction.
    if cfg.ENABLE_FIRST_CANDLE_ENTRY:
        rng0 = h[0] - l[0]
        body0 = abs(cl[0] - o[0])
        if rng0 > 0 and body0 >= cfg.FIRST_CANDLE_FULL_BODY_FRAC * rng0 and cl[0] != o[0]:
            direction0 = "BUY" if cl[0] > o[0] else "SELL"
            t0 = r.index[0] + pd.Timedelta(cfg.TF_RULE)
            if (cfg.SESSION_START <= t0.time() <= cfg.SESSION_END
                    and t0.time() >= cfg.SIGNAL_START_TIME and ema_ok(0, direction0)):
                events.append({
                    "timestamp": t0, "dir": direction0, "name": "first_candle_full_body",
                    "spot": float(cl[0]), "level": float(cl[0]),
                    "sl_ref": float(l[0]) if direction0 == "BUY" else float(h[0]),
                })

    for i in range(3, len(r)):
        bar_close_time = r.index[i] + pd.Timedelta(cfg.TF_RULE)
        ts_time = bar_close_time.time()
        if not (cfg.SESSION_START <= ts_time <= cfg.SESSION_END):
            continue
        if ts_time < cfg.SIGNAL_START_TIME:
            continue

        sig = detect_signal(o[:i+1], h[:i+1], l[:i+1], cl[:i+1], cfg, levels)
        if sig is None:
            continue
        direction, name, c1, c2, level = sig

        if not ema_ok(i, direction):
            continue

        events.append({
            "timestamp": bar_close_time,
            "dir": direction,
            "name": name,
            "spot": float(cl[i]),
            "level": level,
            "sl_ref": float(l[c1]) if direction == "BUY" else float(h[c1]),   # C1 low (CE) / high (PE)
            # evidence for diagnostics: (bar open time, O, H, L, C) of C1, C2, C3 and which touched the line
            "cands": [(r.index[k].strftime("%H:%M"), float(o[k]), float(h[k]), float(l[k]), float(cl[k]))
                      for k in (c1, c2, i)],
            "touched": [n for n, k in (("C1", c1), ("C2", c2), ("C3", i)) if l[k] <= level <= h[k]],
        })

    return events


# --------------------------- ORDER MANAGER (dry-run + live INTRADAY) ---------------------------
class OrderManager:
    def __init__(self, fyers: FyersHTTP, cfg: Config):
        self.fyers = fyers; self.cfg = cfg
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.pattern = None; self.oid = None; self.position_id = None
        self.protection_attached = False
        self.entry_confirmed = False
        self.exit_pending_reason = None
        self.entry_spot = None          # NIFTY spot at entry (index-point SL/TP reference)
        self.sl_spot = None             # absolute NIFTY stop level (C1_OFFSET mode)
        self.locked_pts = 0.0           # profit-lock ratchet: stop is at entry +/- this many index points

    # ── state persistence ──
    def _save_state(self):
        try:
            state = {
                "has": self.has, "symbol": self.symbol, "direction": self.direction,
                "entry_price": self.entry_price, "entry_spot": self.entry_spot,
                "sl_spot": self.sl_spot, "locked_pts": self.locked_pts,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None,
                "pattern": self.pattern, "oid": self.oid,
                "position_id": self.position_id,
                "protection_attached": self.protection_attached,
                "entry_confirmed": self.entry_confirmed,
                "exit_pending_reason": self.exit_pending_reason,
                "saved": datetime.now(IST).isoformat(),
            }
            tmp = STATE_FILE + ".tmp"
            json.dump(state, open(tmp, "w"))
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"[OM] save state: {e}")

    def _clear_state(self):
        try:
            if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
        except Exception: pass

    def _reset_state(self):
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.pattern = None; self.oid = None; self.position_id = None
        self.protection_attached = False
        self.entry_confirmed = False
        self.exit_pending_reason = None
        self.entry_spot = None
        self.sl_spot = None
        self.locked_pts = 0.0

    def restore(self):
        if not os.path.exists(STATE_FILE): return
        try:
            st = json.load(open(STATE_FILE))
            saved = datetime.fromisoformat(st["saved"])
            if saved.date() != datetime.now(IST).date():
                self._clear_state(); return
            if st.get("has"):
                self.has = True; self.symbol = st["symbol"]; self.direction = st["direction"]
                self.entry_price = st["entry_price"]
                self.entry_spot = st.get("entry_spot")
                self.sl_spot = st.get("sl_spot")
                self.locked_pts = float(st.get("locked_pts") or 0.0)
                self.entry_time = datetime.fromisoformat(st["entry_time"]) if st.get("entry_time") else None
                self.pattern = st.get("pattern"); self.oid = st.get("oid")
                self.position_id = st.get("position_id")
                self.protection_attached = bool(st.get("protection_attached", False))
                self.entry_confirmed = bool(st.get("entry_confirmed", False))
                self.exit_pending_reason = st.get("exit_pending_reason")
                if not self.position_id and self.symbol:
                    self.position_id = f"{self.symbol}-INTRADAY"
                logger.info(f"[OM] Restored {self.symbol} {self.direction} @ {self.entry_price} "
                            f"(oid={self.oid}, position_id={self.position_id})")
        except Exception as e:
            logger.error(f"[OM] restore: {e}")

    # ── helpers ──
    def _risk_desc(self) -> str:
        return self.cfg.risk_description()

    def _build_trade_rec(self, exit_ltp: float, reason: str) -> dict:
        pnl_pts = float(exit_ltp) - self.entry_price
        pnl_rs  = pnl_pts * self.cfg.LOT
        return {
            "direction": self.direction, "option_symbol": self.symbol, "pattern": self.pattern,
            "entry_spot": self.entry_spot, "sl_spot": self.sl_spot, "locked_pts": self.locked_pts,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(float(exit_ltp), 2), "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "exit_time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "result": "profit" if pnl_pts > 0 else "loss",
        }

    def _current_ltp(self) -> float:
        """Best-effort current option LTP used for exit logging."""
        if not self.symbol:
            return 0.0
        try:
            resp = self.fyers.quotes(self.symbol)
            v = resp["d"][0]["v"].get("lp")
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    # ── open (dry-run + live INTRADAY market order) ──
    def open(self, symbol: str, direction: str, ltp: float, pattern: str,
             spot: Optional[float] = None, sl_spot: Optional[float] = None) -> bool:
        if self.has:
            logger.warning("[OM] position already open, open() ignored")
            return False

        entry_price = float(ltp)

        if self.cfg.DRY_RUN:
            # Dry-run: simulate fill at LTP; caller simulates SL/TP using live option LTP.
            self.has = True; self.symbol = symbol; self.direction = direction
            self.entry_price = entry_price; self.entry_time = datetime.now(IST)
            self.pattern = pattern
            self.entry_spot = spot; self.sl_spot = sl_spot; self.locked_pts = 0.0
            self.oid = f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            self.entry_confirmed = True
            logger.info(f"[OM][TRADE] BUY {symbol} ({direction}) @ {ltp:.2f} | pattern={pattern}")
            logger.info(f"[OM][TRADE] Entry details: qty={self.cfg.LOT} "
                        f"order_type=MARKET productType=INTRADAY "
                        f"{self._risk_desc()} entry_spot={self.entry_spot} sl_spot={self.sl_spot}")
            self._save_state()
            return True

        # LIVE: documented Fyers V3 market order + broker-side percentage SL/TP.
        # BO/CO are deprecated; INTRADAY is the supported product type.
        data = {
            "symbol": symbol,
            "qty": int(self.cfg.LOT),
            "type": 2,               # Market order
            "side": 1,               # Buy
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0,
            "orderTag": "FCRev",
            "isSliceOrder": False,
        }
        logger.info(f"[OM][LIVE] Placing INTRADAY MARKET BUY: symbol={symbol} dir={direction} "
                    f"qty={self.cfg.LOT} quote_ltp={entry_price:.2f}")
        resp = self.fyers.place_order(data=data)
        logger.info(f"[OM][LIVE] place_order response: {resp}")

        if not (isinstance(resp, dict) and resp.get("s") == "ok"):
            logger.error(f"[OM][LIVE] INTRADAY order rejected: {resp}")
            return False

        self.has = True; self.symbol = symbol; self.direction = direction
        self.entry_price = entry_price; self.entry_time = datetime.now(IST)
        self.pattern = pattern
        self.entry_spot = spot; self.sl_spot = sl_spot; self.locked_pts = 0.0
        self.oid = resp.get("id")
        self.position_id = None
        self.protection_attached = False
        self.entry_confirmed = False
        self.exit_pending_reason = None

        # The market order response gives us the order id. The position id is
        # obtained from the Positions API once the fill becomes visible.
        pos = None
        for attempt in range(self.cfg.PROTECTION_RETRIES):
            pos = self.fyers.get_position(symbol)
            if pos and int(pos.get("netQty", 0)) != 0:
                break
            time.sleep(self.cfg.PROTECTION_RETRY_DELAY_SECS)

        if not pos or int(pos.get("netQty", 0)) == 0:
            logger.error(f"[OM][LIVE] Position not visible after order {self.oid}; "
                         "broker-side SL/TP could not be attached.")
            # Keep local state so the next poll can reconcile the open position.
            self.position_id = f"{symbol}-INTRADAY"
            self.protection_attached = False
            self.entry_confirmed = False
            self._save_state()
            return True

        self.position_id = pos.get("id") or f"{symbol}-INTRADAY"
        self.entry_confirmed = True
        actual_entry = pos.get("buyAvg") or pos.get("netAvg") or entry_price
        try:
            self.entry_price = float(actual_entry)
        except (TypeError, ValueError):
            self.entry_price = entry_price

        if self.cfg.RISK_MODE == "PREMIUM_PCT":
            # Fyers V3 PATCH /positions: legType=2 means percentage.
            protection = self.fyers.set_stop_loss_take_profit(
                position_id=self.position_id,
                stop_loss=self.cfg.STOP_LOSS_PCT,
                take_profit=self.cfg.TAKE_PROFIT_PCT,
                qty=int(pos.get("qty", self.cfg.LOT)),
                leg_type=2,
            )
            logger.info(f"[OM][LIVE] SL/TP attach response: {protection}")

            if not (isinstance(protection, dict) and protection.get("s") == "ok"):
                logger.error(f"[OM][LIVE] Failed to attach broker SL/TP to {self.position_id}: {protection}")
                logger.error("[OM][LIVE] Position remains open; reconciliation will retry protection on the next poll.")
            else:
                self.protection_attached = True
                logger.info(f"[OM][LIVE] Broker protection attached: position={self.position_id} "
                            f"SL={self.cfg.STOP_LOSS_PCT}% TP={self.cfg.TAKE_PROFIT_PCT}%")
        else:
            logger.warning(f"[OM][LIVE] INDEX_POINTS mode: NO broker-side SL/TP attached; the bot exits at "
                           f"{self._risk_desc()} from entry_spot={self.entry_spot} sl_spot={self.sl_spot}.")

        logger.info(f"[OM][LIVE] INTRADAY order placed successfully, order_id={self.oid}")
        logger.info(f"[OM][TRADE] BUY {symbol} ({direction}) @ ~{self.entry_price:.2f} | pattern={pattern}")
        logger.info(f"[OM][TRADE] Entry details: qty={self.cfg.LOT} order_type=MARKET "
                    f"productType=INTRADAY position_id={self.position_id} "
                    f"{self._risk_desc()} entry_spot={self.entry_spot} sl_spot={self.sl_spot}")
        self._save_state()
        return True

    # ── close (dry-run uses provided ltp; live uses DELETE /positions) ──
    def close(self, ltp: float, reason: str) -> Optional[dict]:
        """Close position. LIVE uses the same INTRADAY market-order schema as the
        working S/R bot and then records the exit using the latest available LTP."""
        if not self.has or not self.symbol:
            logger.warning("[OM] no open position, close() ignored")
            return None

        if self.cfg.DRY_RUN:
            exit_ltp = float(ltp)
            rec = self._build_trade_rec(exit_ltp, reason)
            logger.info(f"[OM][TRADE] EXIT {self.symbol} @ {exit_ltp:.2f} reason={reason} "
                        f"entry={self.entry_price:.2f}")
            logger.info(f"[OM][TRADE] PnL Rs {rec['pnl_rs']:,.0f} | "
                        f"points={rec['pnl_pts']:.2f} | result={rec['result']}")
            self._reset_state(); self._clear_state()
            return rec

        # LIVE: verify the broker still reports an open INTRADAY position first.
        # Never send another exit request after netQty reaches 0.
        pos_resp = self.fyers.positions()
        if not isinstance(pos_resp, dict) or pos_resp.get("s") != "ok":
            logger.error(f"[OM][LIVE] Cannot verify position before exit; /positions response: {pos_resp}")
            return None

        pos = None
        for candidate in pos_resp.get("netPositions", []):
            if self.position_id and candidate.get("id") == self.position_id:
                pos = candidate
                break
            if candidate.get("symbol") == self.symbol and candidate.get("productType") == "INTRADAY":
                pos = candidate
                break

        if pos is None:
            logger.info(f"[OM][LIVE] No matching INTRADAY position record; netQty=0 cannot be confirmed. "
                        f"No exit order sent. symbol={self.symbol} reason={reason}")
            return None

        if int(pos.get("netQty", 0)) == 0:
            logger.info(f"[OM][LIVE] Position already closed (netQty=0); no exit order sent. "
                        f"symbol={self.symbol} reason={reason}")
            self._reset_state(); self._clear_state()
            return None

        self.position_id = pos.get("id") or self.position_id or f"{self.symbol}-INTRADAY"
        self.entry_confirmed = True
        actual_entry = pos.get("buyAvg") or pos.get("netAvg") or self.entry_price
        try:
            self.entry_price = float(actual_entry)
        except (TypeError, ValueError):
            pass

        self.exit_pending_reason = reason
        self._save_state()

        logger.info(f"[OM][LIVE] Exiting open INTRADAY position: position_id={self.position_id} "
                    f"symbol={self.symbol} netQty={pos.get('netQty')} reason={reason}")
        resp = self.fyers.exit_position(self.position_id)
        logger.info(f"[OM][LIVE] exit_position response: {resp}")

        if not (isinstance(resp, dict) and resp.get("s") == "ok"):
            logger.error(f"[OM][LIVE] position exit rejected: {resp}")
            self.exit_pending_reason = None
            self._save_state()
            return None

        # Verify the broker position after requesting the exit. A trade is treated
        # as closed only when /positions reports netQty == 0.
        pos_after_resp = self.fyers.positions()
        if not isinstance(pos_after_resp, dict) or pos_after_resp.get("s") != "ok":
            logger.warning(f"[OM][LIVE] Could not verify post-exit netQty; keeping state for reconciliation: "
                           f"{pos_after_resp}")
            return None

        pos_after = None
        for candidate in pos_after_resp.get("netPositions", []):
            if candidate.get("id") == self.position_id:
                pos_after = candidate
                break
            if candidate.get("symbol") == self.symbol and candidate.get("productType") == "INTRADAY":
                pos_after = candidate
                break

        if pos_after is None:
            logger.info(f"[OM][LIVE] Post-exit position record not available; netQty=0 not confirmed. "
                        f"No additional exit order will be sent; reconciliation will continue.")
            return None

        if int(pos_after.get("netQty", 0)) != 0:
            logger.info(f"[OM][LIVE] Exit request accepted but position is still open "
                        f"(netQty={pos_after.get('netQty')}). No additional exit order will be sent "
                        f"until a later reconciliation confirms netQty=0.")
            return None

        exit_ltp = self._current_ltp() or float(ltp)
        rec = self._build_trade_rec(exit_ltp, reason)
        logger.info(f"[OM][TRADE] EXIT {self.symbol} @ ~{exit_ltp:.2f} reason={reason} "
                    f"entry={self.entry_price:.2f}")
        logger.info(f"[OM][TRADE] PnL Rs {rec['pnl_rs']:,.0f} | "
                    f"points={rec['pnl_pts']:.2f} | result={rec['result']}")
        self._reset_state(); self._clear_state()
        return rec

    # ── live broker-position reconciliation ──
    def broker_exit_reason(self, exit_ltp: float) -> str:
        """Best-effort classification when broker-side SL/TP has closed the position."""
        if not self.entry_price:
            return "broker_exit"
        if self.cfg.RISK_MODE != "PREMIUM_PCT":
            return "broker_exit"
        sl_price = self.entry_price * (1 - self.cfg.STOP_LOSS_PCT / 100.0)
        tp_price = self.entry_price * (1 + self.cfg.TAKE_PROFIT_PCT / 100.0)
        if exit_ltp <= sl_price:
            return "stop_loss"
        if exit_ltp >= tp_price:
            return "target"
        return "broker_exit"

    def sync_with_broker(self) -> Optional[dict]:
        """LIVE-only. Treat the INTRADAY position as closed only when /positions reports netQty == 0."""
        if not self.has or not self.symbol or self.cfg.DRY_RUN:
            return None

        resp = self.fyers.positions()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.warning(f"[OM] sync_with_broker: positions API error: {resp}")
            return None

        found = None
        for pos in resp.get("netPositions", []):
            if self.position_id and pos.get("id") == self.position_id:
                found = pos
                break
            if pos.get("symbol") == self.symbol and pos.get("productType") == "INTRADAY":
                found = pos
                break

        # The order may be accepted before the fill becomes visible in Positions.
        # Do not treat that transient state as an exit.
        if not self.entry_confirmed:
            if found is not None and int(found.get("netQty", 0)) != 0:
                self.position_id = found.get("id") or f"{self.symbol}-INTRADAY"
                self.entry_confirmed = True
                actual_entry = found.get("buyAvg") or found.get("netAvg") or self.entry_price
                try:
                    self.entry_price = float(actual_entry)
                except (TypeError, ValueError):
                    pass
                logger.info(f"[OM][LIVE] Entry fill confirmed: position_id={self.position_id} "
                            f"entry~{self.entry_price:.2f}")
            else:
                order = self.fyers.get_order(self.oid)
                if order:
                    status = int(order.get("status", 0))
                    # 1=Canceled, 2=Traded/Filled, 5=Rejected, 6=Pending, 7=Expired.
                    if status in (1, 5, 7):
                        logger.error(f"[OM][LIVE] Entry order {self.oid} is not open: {order}")
                        self._reset_state(); self._clear_state()
                        return {"entry_failed": True}
                    logger.info(f"[OM][LIVE] Entry order {self.oid} status={status}; waiting for position visibility.")
                return None

        if found is not None and int(found.get("netQty", 0)) != 0:
            if self.position_id is None:
                self.position_id = found.get("id") or f"{self.symbol}-INTRADAY"

            if self.exit_pending_reason:
                logger.info(f"[OM][LIVE] Exit pending for {self.symbol}; broker still reports "
                            f"netQty={found.get('netQty')}. Waiting for netQty=0 before finalizing the exit.")
                return None

            actual_entry = found.get("buyAvg") or found.get("netAvg") or self.entry_price
            try:
                self.entry_price = float(actual_entry)
            except (TypeError, ValueError):
                pass

            if self.cfg.RISK_MODE == "PREMIUM_PCT" and not self.protection_attached:
                protection = self.fyers.set_stop_loss_take_profit(
                    position_id=self.position_id,
                    stop_loss=self.cfg.STOP_LOSS_PCT,
                    take_profit=self.cfg.TAKE_PROFIT_PCT,
                    qty=int(found.get("qty", self.cfg.LOT)),
                    leg_type=2,
                )
                logger.info(f"[OM][LIVE] Retry SL/TP attach response: {protection}")
                if isinstance(protection, dict) and protection.get("s") == "ok":
                    self.protection_attached = True
                    self._save_state()
                    logger.info(f"[OM][LIVE] Broker protection attached on retry: position={self.position_id}")
            return None

        if found is None:
            logger.info(f"[OM][LIVE] No matching INTRADAY position record for {self.symbol}; "
                        f"waiting until /positions confirms netQty=0.")
            return None

        # Matching position is present with netQty=0 -> broker has closed it.
        exit_ltp = self._current_ltp()
        reason = self.exit_pending_reason or self.broker_exit_reason(exit_ltp)
        logger.info(f"[OM][LIVE] Broker position closed: position_id={self.position_id} "
                    f"symbol={self.symbol} ltp~{exit_ltp:.2f} reason={reason}")
        rec = self._build_trade_rec(exit_ltp, reason)
        logger.info(f"[OM][TRADE] EXIT {self.symbol} @ ~{exit_ltp:.2f} reason={reason} "
                    f"entry={self.entry_price:.2f}")
        logger.info(f"[OM][TRADE] PnL Rs {rec['pnl_rs']:,.0f} | "
                    f"points={rec['pnl_pts']:.2f} | result={rec['result']}")
        self._reset_state(); self._clear_state()
        return rec


# --------------------------- ENGINE ---------------------------
class Engine:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()
        self.expiry_resolver = ExpiryResolver()
        self.om = OrderManager(self.fy, self.cfg)
        self.om.restore()
        self.trades: List[dict] = []
        self._fired_events: set = set()          # timestamps already acted on today
        self._last_trade_dir: Optional[str] = None  # DIRECTION ALTERNATION tracker (resets daily)
        self._today: Optional[object] = None
        self._levels_logged: Optional[object] = None
        self._last_scan = 0.0                    # monotonic time of the last signal scan
        logger.info(f"[ENG] ready. interval={self.cfg.POLL_INTERVAL_SECS}s")
        if self.cfg.DRY_RUN:
            logger.warning("*** DRY-RUN — no real orders ***")
        else:
            logger.warning("*** LIVE MODE — INTRADAY market entry + broker-side SL/TP + DELETE /positions exit ***")

    def _init_fyers(self) -> FyersHTTP:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
        app = open("fyers_appid.txt").read().strip()
        tok = open("fyers_token.txt").read().strip()
        fy = FyersHTTP(app, tok); p = fy.profile()
        if p.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed: {p}")
        logger.info(f"[ENG] connected {p['data']['name']}")
        return fy

    def _fetch_today_bars(self) -> pd.DataFrame:
        now = datetime.now(IST)
        # Fetch historical NIFTY bars only for calculating EMA200.
        # The existing pattern/swing logic still receives TODAY's bars only.
        history_from = now - timedelta(days=self.cfg.EMA_LOOKBACK_DAYS)
        resp = self.fy.history(self.cfg.UNDERLYING, self.cfg.BASE_RES,
                               history_from.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
        if resp.get("s") != "ok":
            logger.warning(f"[ENG] history err: {resp}")
            return pd.DataFrame()
        candles = resp.get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        df = df.iloc[df.index.indexer_between_time(self.cfg.DATA_START, self.cfg.SESSION_END)]
        if self.cfg.USE_EMA_FILTER:
            df["ema200"] = df["close"].ewm(
                span=self.cfg.EMA_PERIOD, adjust=False
            ).mean()
        # Keep the existing pattern/swing calculation restricted to today's bars.
        return df.loc[df.index.date == now.date()].copy()

    def _opt_symbol(self, spot: float, direction: str) -> str:
        c = self.cfg
        exp_code, _ = self.expiry_resolver.resolve_expiry_code(
            datetime.now(IST).date(), expiry_code, get_next_tuesday)
        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{exp_code}{strike:05d}{typ}"

    def _ltp(self, symbol: str) -> Optional[float]:
        r = self.fy.quotes(symbol)
        try:
            v = r["d"][0]["v"].get("lp")
            return float(v) if v is not None else None
        except Exception:
            return None

    def _drop_forming_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trims the still-forming bar so only CLOSED bars feed the signal
        engine. Bar duration is derived from BASE_RES (in minutes, since no
        'S' suffix -- matches Fyers' convention)."""
        if df.empty:
            return df
        try:
            bar_minutes = int(self.cfg.BASE_RES)
        except ValueError:
            bar_minutes = 1
        now = datetime.now(IST)
        last_start = df.index[-1]
        if last_start.tzinfo is None:
            last_start = IST.localize(last_start)
        if last_start + timedelta(minutes=bar_minutes) > now:
            df = df.iloc[:-1]
        return df

    def _check_dryrun_exits(self) -> bool:
        """DRY-RUN only: fetch current option LTP and simulate SL/TP exit."""
        if not self.cfg.DRY_RUN or not self.om.has or self.cfg.RISK_MODE != "PREMIUM_PCT":
            return False

        opt_ltp = self._ltp(self.om.symbol)
        if opt_ltp is None:
            return False

        if self.cfg.USE_PREMIUM_SL and opt_ltp <= self.om.entry_price * (1 - self.cfg.STOP_LOSS_PCT / 100.0):
            rec = self.om.close(opt_ltp, "stop_loss")
            if rec: self.trades.append(rec)
            return True

        if self.cfg.USE_PREMIUM_TP and opt_ltp >= self.om.entry_price * (1 + self.cfg.TAKE_PROFIT_PCT / 100.0):
            rec = self.om.close(opt_ltp, "target")
            if rec: self.trades.append(rec)
            return True

        return False

    def _check_index_exit(self) -> bool:
        """INDEX_POINTS mode (dry-run and live): exit when NIFTY has moved SL/TP points from the
        entry spot. CE profits when NIFTY rises, PE profits when NIFTY falls."""
        c = self.cfg
        if c.RISK_MODE != "INDEX_POINTS" or not self.om.has or self.om.entry_spot is None:
            return False
        if not c.DRY_RUN and not self.om.entry_confirmed:
            return False                       # fill not confirmed yet
        if self.om.exit_pending_reason:
            return False                       # exit already sent; sync_with_broker finalises it
        spot = self._ltp(c.UNDERLYING)
        if spot is None:
            return False
        move = (spot - self.om.entry_spot) if self.om.direction == "BUY" else (self.om.entry_spot - spot)
        # PROFIT LOCK ratchet: raise the locked profit as the favourable move grows (never lowered).
        lock_new = profit_lock_points(move, c)
        if lock_new > self.om.locked_pts:
            self.om.locked_pts = lock_new
            self.om._save_state()
            logger.info(f"[ENG] Profit lock: favourable move {move:+.2f} pts -> stop moved to entry "
                        f"{'+' if self.om.direction == 'BUY' else '-'}{lock_new:g} pts "
                        f"(NIFTY level {self.om.entry_spot + lock_new if self.om.direction == 'BUY' else self.om.entry_spot - lock_new:.2f})")
        reason = None
        if self.om.sl_spot is not None:
            sl_hit = (spot <= self.om.sl_spot) if self.om.direction == "BUY" else (spot >= self.om.sl_spot)
        else:
            sl_hit = bool(c.SL_INDEX_POINTS) and move <= -c.SL_INDEX_POINTS
        if self.om.locked_pts > 0 and move <= self.om.locked_pts:
            reason = "profit_lock"
        elif sl_hit:
            reason = "stop_loss"
        elif c.TP_INDEX_POINTS and move >= c.TP_INDEX_POINTS:
            reason = "target"
        if reason is None:
            return False
        opt_ltp = self._ltp(self.om.symbol) or self.om.entry_price
        logger.info(f"[ENG] Index {reason}: NIFTY {spot:.2f} vs entry {self.om.entry_spot:.2f} "
                    f"({self.om.direction}, favourable move {move:+.2f} pts, sl_spot={self.om.sl_spot}, locked={self.om.locked_pts:g}) -> exiting {self.om.symbol}")
        rec = self.om.close(opt_ltp, reason)
        if rec: self.trades.append(rec)
        return True

    def _check_live_broker_exit(self) -> bool:
        """LIVE only: detect broker-side SL/TP or other automatic closure."""
        if self.cfg.DRY_RUN or not self.om.has:
            return False
        rec = self.om.sync_with_broker()
        if isinstance(rec, dict) and rec.get("entry_failed"):
            return True
        if rec is not None:
            self.trades.append(rec)
            return True
        return False

    def _eod_exit_if_needed(self) -> bool:
        now = datetime.now(IST)
        eod = IST.localize(datetime.combine(now.date(), self.cfg.EOD_EXIT_TIME))
        if now >= eod and self.om.has:
            # close() re-checks /positions and sends DELETE /positions only while
            # the broker still reports an open INTRADAY position (netQty != 0).
            ltp = self._ltp(self.om.symbol) or self.om.entry_price
            rec = self.om.close(ltp, "eod_exit")
            if rec: self.trades.append(rec)
            return True
        return False

    def _poll(self):
        now = datetime.now(IST)

        if self._today != now.date():
            self._today = now.date()
            self._fired_events = set()
            self._last_trade_dir = None   # direction alternation resets each new trading day
            logger.info(f"[ENG] New day {self._today} — state reset.")

        if not (self.cfg.SESSION_START <= now.time() <= self.cfg.SESSION_END):
            return

        # Position monitoring happens before EOD/manual exit. LIVE is considered
        # closed only when GET /positions reports netQty == 0.
        if self.cfg.DRY_RUN:
            if self._check_dryrun_exits():
                return
        else:
            if self._check_live_broker_exit():
                return

        # INDEX_POINTS mode: bot-managed SL/TP on the NIFTY index (no-op in PREMIUM_PCT mode).
        if self._check_index_exit():
            return

        # EOD exit is attempted only after the latest broker-position check.
        # OrderManager.close() performs another /positions check immediately before
        # sending DELETE /positions.
        if self._eod_exit_if_needed():
            return

        # While a trade is open we poll fast for exits, but only rescan signals at the normal pace.
        if self.om.has and (time.monotonic() - self._last_scan) < self.cfg.POLL_INTERVAL_SECS:
            return
        self._last_scan = time.monotonic()

        df = self._fetch_today_bars()
        df = self._drop_forming_bar(df)
        if df.empty:
            return

        if self._levels_logged != self._today:
            lv = get_first_candle_levels(df, self.cfg)
            if lv:
                logger.info(f"[ENG] First candle {self.cfg.FIRST_CANDLE_TIME.strftime('%H:%M')} levels: "
                            f"GREEN(high)={lv[0]:.2f} RED(low)={lv[1]:.2f}")
                self._levels_logged = self._today

        events = generate_today_events(df, self.cfg)
        max_age = timedelta(minutes=self.cfg.MAX_SIGNAL_AGE_MIN)
        events = [e for e in events if now - e["timestamp"] <= max_age]   # drop stale signals
        if not events:
            return

        next_ev = None
        for ev in events:
            if ev["timestamp"] not in self._fired_events:
                next_ev = ev
                break
        if next_ev is None:
            return

        self._fired_events.add(next_ev["timestamp"])

        # single-position engine: ignore all new signals while in a trade
        if self.om.has:
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} ignored — already in position.")
            return

        # DIRECTION ALTERNATION: no CE after CE, no PE after PE
        if self.cfg.ALTERNATE_DIRECTION and self._last_trade_dir is not None and next_ev["dir"] == self._last_trade_dir:
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} skipped — "
                        f"same direction as last trade ({self._last_trade_dir}).")
            return

        if not (self.cfg.TRADE_START <= now.time() <= self.cfg.TRADE_END):
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} outside trade window — skipping.")
            return

        sym = self._opt_symbol(next_ev["spot"], next_ev["dir"])
        ltp = self._ltp(sym)
        if ltp is None:
            logger.warning(f"[ENG] No LTP for {sym} — skipping this signal.")
            return

        logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} "
                    f"pattern={next_ev['name']} level={next_ev['level']:.2f} spot={next_ev['spot']:.2f} "
                    f"-> entering {sym} @ {ltp:.2f}")
        entry_spot = self._ltp(self.cfg.UNDERLYING) or next_ev["spot"]   # SL/TP reference (index points)
        sl_spot = None
        if self.cfg.RISK_MODE == "INDEX_POINTS" and self.cfg.SL_MODE == "C1_OFFSET" and next_ev.get("sl_ref") is not None:
            off = self.cfg.C1_SL_OFFSET_POINTS
            sl_spot = next_ev["sl_ref"] - off if next_ev["dir"] == "BUY" else next_ev["sl_ref"] + off
            logger.info(f"[ENG] C1 stop: sl_ref={next_ev['sl_ref']:.2f} -> sl_spot={sl_spot:.2f} "
                        f"({abs(entry_spot - sl_spot):.1f} pts from entry spot {entry_spot:.2f}, {self.cfg.tp_text()})")
        opened = self.om.open(sym, next_ev["dir"], ltp, next_ev["name"], spot=entry_spot, sl_spot=sl_spot)
        if opened:
            self._last_trade_dir = next_ev["dir"]

    def run(self):
        logger.info("[ENG] starting First-Candle H/L Reversal poll loop")
        try:
            while True:
                t0 = time.monotonic()
                try:
                    self._poll()
                except Exception as e:
                    logger.error(f"poll err: {e}", exc_info=True)
                interval = (self.cfg.POSITION_POLL_INTERVAL_SECS if self.om.has
                            else self.cfg.POLL_INTERVAL_SECS)
                time.sleep(max(0.0, interval - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            logger.info("[ENG] interrupted")
        finally:
            self._shutdown()

    def _shutdown(self):
        if self.om.has:
            ltp = self._ltp(self.om.symbol) or self.om.entry_price
            rec = self.om.close(ltp, "shutdown")
            if rec: self.trades.append(rec)
        self._export()
        logger.info("[ENG] shutdown complete")

    def _export(self):
        if not self.trades:
            logger.info("[ENG] no trades to export")
            return
        try:
            df = pd.DataFrame(self.trades)
            df["cum_pnl_rs"] = df["pnl_rs"].cumsum()
            d = datetime.now(IST).strftime("%Y%m%d")
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(path, exist_ok=True)
            fp = os.path.join(path, f"firstcandle_trades_{d}.xlsx")
            df.to_excel(fp, index=False)
            net = df["pnl_rs"].sum(); wins = (df["pnl_rs"] > 0).sum()
            logger.info(f"[ENG] exported {len(df)} trades -> {fp} | net Rs {net:,.0f} | wins {wins}/{len(df)}")
        except Exception as e:
            logger.error(f"export: {e}")


if __name__ == "__main__":
    Engine().run()
