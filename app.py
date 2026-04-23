# type: ignore
import datetime
import logging
import math
import statistics
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pytz

from wiz_trader import QuotesClient, WizzerClient

# --- CONFIG ------------------------------------------------------------------
WATCHLIST: List[dict] = []

CAPITAL = 10_00_000
SCREEN_LIMIT = 250
HISTORICAL_BATCH_SIZE = 20

TZ = pytz.timezone("Asia/Kolkata")
TRADING_START = datetime.time(9, 15, 0)
UNIVERSE_BUILD_TIME = datetime.time(9, 15, 0)
ENTRY_TIME = datetime.time(10, 0, 0)
ENTRY_WINDOW_END = datetime.time(10, 5, 0)
RECENTER_CUTOFF_TIME = datetime.time(13, 30, 0)
EOD_EXIT_TIME = datetime.time(15, 10, 0)
TRADING_END = datetime.time(15, 30, 0)

DAILY_HORIZONS = (2, 5, 10, 20)
HOURLY_HORIZONS = (2, 4, 8, 12)
DAILY_WEIGHTS = {2: 0.20, 5: 0.25, 10: 0.25, 20: 0.30}
HOURLY_WEIGHTS = {2: 0.20, 4: 0.25, 8: 0.25, 12: 0.30}

CLASS_SIDEWAYS = "SIDEWAYS"
CLASS_POTENTIAL = "POTENTIAL_SIDEWAYS"
CLASS_NOT = "NOT_SIDEWAYS"

PREMIUM_CAPTURE_EXIT = 0.40
PREMIUM_EXPANSION_EXIT = 2.00
DELTA_EXIT_LIMIT = 0.35
IV_EXPANSION_EXIT = 1.35
SIGMA_MOVE_EXIT = 1.25
SIGMA_MOVE_RECENTER = 0.75
SHORT_STRIKE_PROXIMITY = 0.15
MANAGEMENT_COOLDOWN_SEC = 5.0

# --- LOGGING -----------------------------------------------------------------
logging.Formatter.converter = lambda *_: datetime.datetime.now(TZ).timetuple()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("StockOptionIronCondor")


# --- UTILITY -----------------------------------------------------------------
def now_ist() -> datetime.datetime:
    return datetime.datetime.now(TZ)


def is_market_open() -> bool:
    now = now_ist().time()
    return TRADING_START <= now < TRADING_END


def parse_ts(ts_value: Optional[str]) -> Optional[datetime.datetime]:
    if not ts_value:
        return None
    normalized = ts_value.replace("Z", "+00:00")
    return datetime.datetime.fromisoformat(normalized).astimezone(TZ)


def mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def stdev(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    if len(items) < 2:
        return 0.0
    return statistics.pstdev(items)


def log_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return math.log(a / b)


def median(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    if not items:
        return 0.0
    return float(statistics.median(items))


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def previous_weekday(day: datetime.date) -> datetime.date:
    prev = day - datetime.timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= datetime.timedelta(days=1)
    return prev


def latest_completed_hour_candle_start(ts: datetime.datetime) -> datetime.datetime:
    session_start = ts.replace(hour=9, minute=15, second=0, microsecond=0)
    if ts <= session_start + datetime.timedelta(hours=1):
        prev_day = previous_weekday(ts.date())
        return datetime.datetime.combine(prev_day, datetime.time(14, 15), tzinfo=TZ)

    elapsed_minutes = int((ts - session_start).total_seconds() // 60)
    completed_hours = max((elapsed_minutes // 60) - 1, 0)
    candle_start = session_start + datetime.timedelta(hours=completed_hours)

    if candle_start.time() > datetime.time(14, 15):
        candle_start = datetime.datetime.combine(ts.date(), datetime.time(14, 15), tzinfo=TZ)

    return candle_start


def make_identifier(exchange: str, trading_symbol: str, exchange_token: int) -> str:
    return f"{exchange}:{trading_symbol}:{exchange_token}"


def float_or_none(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def exc_message(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("message"):
                return str(payload["message"])
        except Exception:
            text = getattr(response, "text", "")
            if text:
                return str(text)
    return str(exc)


# --- STRATEGY ----------------------------------------------------------------
class Strategy:
    def __init__(self, quotes: QuotesClient, api: WizzerClient, instruments: list):
        self.quotes = quotes
        self.api = api
        self.running = True
        self.ws = None

        self._eod_monitor_started = False
        self._position_monitor_started = False
        self._scheduler_started = False
        self._exit_lock = threading.RLock()
        self._entry_lock = threading.Lock()
        self._order_lock = threading.Lock()
        self._management_lock = threading.Lock()

        self._exited_once = False
        self._last_management_ts = 0.0

        self._session_date: Optional[datetime.date] = None
        self._universe_built_for: Optional[datetime.date] = None
        self._entry_attempted_for: Optional[datetime.date] = None
        self._full_exit_date: Optional[datetime.date] = None

        self.vix_identifier: Optional[str] = None
        self.candidates: List[dict] = []
        self.selected_stock: Optional[dict] = None

        self.ltp_cache: Dict[str, float] = {}
        self.greeks_cache: Dict[str, Dict[str, Optional[float]]] = {}

        self.instruments = {
            make_identifier(i["exchange"], i["trading_symbol"], i["exchange_token"]): {
                **i,
                "position": 0,
                "entry_price": None,
                "last_bar": None,
                "candle": {"open": [], "high": [], "low": [], "close": []},
            }
            for i in instruments
        }

        self.position = {
            "active": False,
            "underlying": None,
            "classification": None,
            "expiry_date": None,
            "spot_entry": None,
            "entry_iv": None,
            "sigma0": None,
            "entry_credit": None,
            "lot_size": None,
            "lots": 0,
            "quantity": 0,
            "recentered": False,
            "roll_count": 0,
            "short_call": None,
            "long_call": None,
            "short_put": None,
            "long_put": None,
        }

    # --- LIFECYCLE -----------------------------------------------------------
    def _api_post(self, endpoint: str, body: dict):
        return self.api._make_request("POST", endpoint, json=body)

    def _get_position_sizing_capital(self) -> float:
        """
        Resolve the capital to use for entry sizing.

        Prefer the SDK strategy cash endpoint so the position size tracks the
        backend allocation model. Fall back to the static capital constant when
        the SDK method is unavailable or returns unusable values.
        """
        fallback_capital = float(CAPITAL)

        if not hasattr(self.api, "get_strategy_cash"):
            logger.warning(
                "[Cash] SDK get_strategy_cash() not available; using fallback capital %.2f",
                fallback_capital,
            )
            return fallback_capital

        try:
            cash = (
                self.api.get_strategy_cash(
                    trading_mode=WizzerClient.TRADING_MODE_PAPER,
                )
                or {}
            )

            available_cash = float(cash.get("availableCash") or 0)
            allocated_capital = float(cash.get("allocatedCapital") or 0)
            net_pnl = float(cash.get("netPnl") or 0)

            if available_cash > 0:
                logger.info(
                    "[Cash] Using availableCash=%.2f for sizing "
                    "(allocatedCapital=%.2f, netPnl=%.2f, tradingMode=%s)",
                    available_cash,
                    allocated_capital,
                    net_pnl,
                    cash.get("tradingMode"),
                )
                return available_cash

            if allocated_capital > 0:
                logger.warning(
                    "[Cash] availableCash=%.2f is not usable; falling back to "
                    "allocatedCapital=%.2f for sizing",
                    available_cash,
                    allocated_capital,
                )
                return allocated_capital

            logger.warning(
                "[Cash] Cash endpoint returned unusable sizing data %s; "
                "using fallback capital %.2f",
                cash,
                fallback_capital,
            )
        except Exception as exc:
            logger.warning(
                "[Cash] Failed to fetch strategy cash via SDK (%s); using "
                "fallback capital %.2f",
                exc,
                fallback_capital,
            )

        return fallback_capital

    def on_connect(self, ws):
        self.ws = ws
        logger.info("Connected. Subscribing instruments.")
        if self.instruments:
            ws.subscribe(list(self.instruments.keys()))

        if not self._eod_monitor_started:
            self._eod_monitor_started = True
            threading.Thread(target=self._eod_exit, daemon=True).start()

        if not self._position_monitor_started:
            self._position_monitor_started = True
            threading.Thread(target=self._position_monitor, daemon=True).start()

        if not self._scheduler_started:
            self._scheduler_started = True
            threading.Thread(target=self._decision_scheduler, daemon=True).start()

    def on_tick(self, ws, tick):
        if tick.get("type") == "greeks":
            identifier = tick.get("identifier")
            if identifier:
                self.greeks_cache[identifier] = {
                    "iv": float_or_none(tick.get("iv")),
                    "delta": float_or_none(tick.get("delta")),
                    "gamma": float_or_none(tick.get("gamma")),
                    "theta": float_or_none(tick.get("theta")),
                    "vega": float_or_none(tick.get("vega")),
                    "rho": float_or_none(tick.get("rho")),
                }
            return

        identifier = tick.get("identifier")
        if not identifier or identifier not in self.instruments:
            return

        ltp = float_or_none(tick.get("ltp"))
        if ltp is not None:
            self.ltp_cache[identifier] = ltp

        if not is_market_open():
            return

        if not self.position["active"]:
            return

        ts = parse_ts(tick.get("ts")) or now_ist()
        if time.time() - self._last_management_ts < MANAGEMENT_COOLDOWN_SEC:
            return

        self._last_management_ts = time.time()
        self._manage_position(ts)

    def run(self):
        self.quotes.on_tick = self.on_tick
        self.quotes.on_connect = self.on_connect
        self.quotes.connect()

    # --- SCHEDULING ----------------------------------------------------------
    def _decision_scheduler(self):
        while self.running:
            now = now_ist()
            self._roll_session_if_needed(now.date())

            if now.time() >= UNIVERSE_BUILD_TIME and self._universe_built_for != now.date():
                self._build_candidate_universe()

            if ENTRY_TIME <= now.time() <= ENTRY_WINDOW_END and self._entry_attempted_for != now.date():
                self._entry_attempted_for = now.date()
                self._attempt_entry()

            time.sleep(5)

    def _roll_session_if_needed(self, session_date: datetime.date):
        if self._session_date == session_date:
            return

        self._session_date = session_date
        self._universe_built_for = None
        self._entry_attempted_for = None
        self.candidates = []
        self.selected_stock = None
        self.greeks_cache.clear()
        self.ltp_cache.clear()
        self._last_management_ts = 0.0
        logger.info("[Session] Rolled strategy state for %s", session_date.isoformat())

    # --- DATA DISCOVERY ------------------------------------------------------
    def _build_candidate_universe(self):
        build_date = now_ist().date()
        logger.info("[Selection] Building F&O stock universe and DT-ELR-SI classifications.")
        try:
            self.vix_identifier = self._resolve_vix_identifier()
            raw_equities = self._fetch_candidate_equities()
            fo_candidates = self._filter_monthly_optionable(raw_equities)
            self.candidates = self._classify_candidates(fo_candidates)
            self.selected_stock = self.candidates[0] if self.candidates else None
            self._universe_built_for = build_date

            if self.selected_stock:
                logger.info(
                    "[Selection] Chosen stock: %s | score=%.4f | drift=%.4f",
                    self.selected_stock["identifier"],
                    self.selected_stock["combined_score"],
                    self.selected_stock["drift_rank"],
                )
            else:
                logger.info("[Selection] No stock qualified as SIDEWAYS for %s.", build_date.isoformat())
        except Exception as exc:
            self._universe_built_for = build_date
            logger.error("[Selection] Universe build failed: %s", exc)

    def _resolve_vix_identifier(self) -> str:
        indices = self.api.get_indices(exchange="NSE") or []
        for item in indices:
            symbol = str(item.get("tradingSymbol") or "").upper().strip()
            if symbol in {"INDIA VIX", "NIFTY INDIA VIX", "VIX"}:
                return make_identifier(item["exchange"], item["tradingSymbol"], int(item["exchangeToken"]))
        raise ValueError("Unable to resolve INDIA VIX identifier from SDK indices.")

    def _fetch_candidate_equities(self) -> List[dict]:
        try:
            derivatives = self._api_post(
                "/datahub/instruments/filter-by-derivatives",
                {"hasOptions": True},
            ) or []
        except Exception as exc:
            logger.warning("[Selection] Derivatives filter endpoint failed, falling back to screener: %s", exc)
            derivatives = []

        if derivatives:
            results = []
            for item in derivatives[:SCREEN_LIMIT]:
                exchange = item.get("exchange") or "NSE"
                trading_symbol = item.get("tradingSymbol")
                exchange_token = item.get("exchangeToken")
                if not trading_symbol or exchange_token is None:
                    continue
                identifier = make_identifier(exchange, trading_symbol, int(exchange_token))
                results.append(
                    {
                        "exchange": exchange,
                        "trading_symbol": trading_symbol,
                        "exchange_token": int(exchange_token),
                        "identifier": identifier,
                    }
                )

            logger.info("[Selection] F&O-capable universe size from derivatives filter: %d", len(results))
            return results

        screen = self.api.run_screener(
            filters={
                "$and": [
                    {"instrumentType": {"$in": ["EQLC", "EQMC", "EQSC", "EQMCC", "EQNCC"]}},
                ]
            },
            sort=[{"hmdVolume": -1}],
            limit=SCREEN_LIMIT,
        ) or []

        results = []
        for item in screen:
            exchange = item.get("exchange") or "NSE"
            trading_symbol = item.get("tradingSymbol")
            exchange_token = item.get("exchangeToken")
            if not trading_symbol or exchange_token is None:
                continue
            identifier = make_identifier(exchange, trading_symbol, int(exchange_token))

            results.append(
                {
                    "exchange": exchange,
                    "trading_symbol": trading_symbol,
                    "exchange_token": int(exchange_token),
                    "identifier": identifier,
                }
            )

        logger.info("[Selection] Raw NSE equity universe size: %d", len(results))
        return results

    def _filter_monthly_optionable(self, equities: List[dict]) -> List[dict]:
        optionable = []
        for stock in equities:
            try:
                monthly_expiry = self._pick_monthly_expiry(stock["identifier"])
            except Exception as exc:
                logger.debug("[Selection] Skipping %s: %s", stock["identifier"], exc)
                continue
            if not monthly_expiry:
                continue
            stock_copy = dict(stock)
            stock_copy["monthly_expiry"] = monthly_expiry
            optionable.append(stock_copy)

        logger.info("[Selection] Monthly optionable universe size: %d", len(optionable))
        return optionable

    def _pick_monthly_expiry(self, identifier: str, prefer_future_only: bool = True) -> Optional[str]:
        today = now_ist().date()
        try:
            result = self.api.get_option_expiry_list(identifier=identifier)
        except Exception as exc:
            message = exc_message(exc)
            if "Expiry list for the provided instrument not found" in message:
                return None
            raise
        expiries = result.get("expiryList", []) if isinstance(result, dict) else result or []
        monthly = []
        for item in expiries:
            if not item.get("isOptionsExpiry"):
                continue
            contract = str(item.get("contract") or "")
            if contract not in {"near_month", "mid_month", "far_month"}:
                continue
            expiry_date = datetime.date.fromisoformat(str(item["date"])[:10])
            if prefer_future_only and expiry_date <= today:
                continue
            monthly.append(expiry_date)

        if not monthly:
            return None
        return str(sorted(monthly)[0])

    # --- HISTORICAL ANALYTICS ------------------------------------------------
    def _classify_candidates(self, optionable_stocks: List[dict]) -> List[dict]:
        if not optionable_stocks:
            return []

        vix_close = self._fetch_vix_close_series()
        if len(vix_close) < 100:
            raise ValueError("VIX history is incomplete; need 100 completed daily values.")

        stock_ids = [item["identifier"] for item in optionable_stocks]
        daily_map = self._fetch_history_map(stock_ids, interval="1d")
        hourly_map = self._fetch_history_map(stock_ids, interval="1h")

        analytics = []
        for stock in optionable_stocks:
            daily = daily_map.get(stock["identifier"], [])
            hourly = hourly_map.get(stock["identifier"], [])
            if len(daily) < 100 or len(hourly) < 100:
                continue

            daily_metrics = self._compute_timeframe_metrics(daily[-100:], DAILY_HORIZONS, DAILY_WEIGHTS, is_hourly=False)
            hourly_metrics = self._compute_timeframe_metrics(hourly[-100:], HOURLY_HORIZONS, HOURLY_WEIGHTS, is_hourly=True)
            if not daily_metrics or not hourly_metrics:
                continue

            analytics.append(
                {
                    **stock,
                    "daily": daily_metrics,
                    "hourly": hourly_metrics,
                    "complete_data": True,
                }
            )

        if not analytics:
            return []

        daily_baseline = {
            horizon: median(item["daily"]["aelr"][horizon] for item in analytics)
            for horizon in DAILY_HORIZONS
        }
        hourly_baseline = {
            horizon: median(item["hourly"]["aelr"][horizon] for item in analytics)
            for horizon in HOURLY_HORIZONS
        }
        vix_metrics = self._compute_vix_metrics(vix_close[-100:])

        qualified = []
        for item in analytics:
            daily_score = self._apply_baseline(item["daily"], daily_baseline, DAILY_WEIGHTS)
            hourly_score = self._apply_baseline(item["hourly"], hourly_baseline, HOURLY_WEIGHTS)

            daily_sideways = (
                daily_score >= 0.05
                and item["daily"]["drift"] <= 0.30
                and item["daily"]["bandwidth"] <= 0.12
                and item["daily"]["upper_touches"] >= 2
                and item["daily"]["lower_touches"] >= 2
                and item["daily"]["vol_exp"] <= 1.15
            )
            hourly_sideways = (
                hourly_score >= 0.04
                and item["hourly"]["drift"] <= 0.35
                and item["hourly"]["bandwidth"] <= 0.05
                and item["hourly"]["upper_touches"] >= 2
                and item["hourly"]["lower_touches"] >= 2
                and item["hourly"]["vol_exp"] <= 1.20
            )

            if daily_sideways and hourly_sideways and vix_metrics["zscore"] < 1.0 and vix_metrics["slope"] <= 0:
                classification = CLASS_SIDEWAYS
            elif (daily_sideways ^ hourly_sideways) and vix_metrics["zscore"] < 1.5:
                classification = CLASS_POTENTIAL
            else:
                classification = CLASS_NOT

            combined_score = daily_score + hourly_score
            drift_rank = item["daily"]["drift"] + item["hourly"]["drift"]

            if classification != CLASS_SIDEWAYS:
                continue

            qualified.append(
                {
                    **item,
                    "classification": classification,
                    "mrscore_1d": daily_score,
                    "mrscore_1h": hourly_score,
                    "combined_score": combined_score,
                    "drift_rank": drift_rank,
                    "vix_z": vix_metrics["zscore"],
                    "vix_slope": vix_metrics["slope"],
                }
            )

        qualified.sort(key=lambda item: (-item["combined_score"], item["drift_rank"], item["identifier"]))
        logger.info("[Selection] SIDEWAYS candidates: %d", len(qualified))
        return qualified[:5]

    def _fetch_vix_close_series(self) -> List[float]:
        end_date = previous_weekday(now_ist().date())
        start_date = end_date - datetime.timedelta(days=200)
        response = self.api.get_historical_ohlcv(
            instruments=[self.vix_identifier],
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            interval="1d",
            ohlcv=["close"],
        ) or []
        if not response:
            return []
        data = response[0].get("data", [])
        return [float(bar["close"]) for bar in data if float_or_none(bar.get("close")) is not None][-100:]

    def _fetch_history_map(self, identifiers: List[str], interval: str) -> Dict[str, List[dict]]:
        if interval == "1d":
            end_dt = previous_weekday(now_ist().date())
            start_dt = end_dt - datetime.timedelta(days=220)
            start_arg = start_dt.isoformat()
            end_arg = end_dt.isoformat()
        else:
            end_candle = latest_completed_hour_candle_start(now_ist())
            start_dt = end_candle - datetime.timedelta(days=45)
            start_arg = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_arg = end_candle.strftime("%Y-%m-%d %H:%M:%S")

        history_map: Dict[str, List[dict]] = {}
        for batch in chunked(identifiers, HISTORICAL_BATCH_SIZE):
            payload = self.api.get_historical_ohlcv(
                instruments=batch,
                start_date=start_arg,
                end_date=end_arg,
                interval=interval,
                ohlcv=["open", "high", "low", "close"],
            ) or []
            for item in payload:
                instrument = item.get("instrument", {})
                identifier = instrument.get("identifier")
                if identifier:
                    history_map[identifier] = item.get("data", [])
        return history_map

    def _compute_timeframe_metrics(
        self,
        candles: List[dict],
        horizons: Tuple[int, ...],
        weights: Dict[int, float],
        is_hourly: bool,
    ) -> Optional[dict]:
        opens = [float_or_none(bar.get("open")) for bar in candles]
        highs = [float_or_none(bar.get("high")) for bar in candles]
        lows = [float_or_none(bar.get("low")) for bar in candles]
        closes = [float_or_none(bar.get("close")) for bar in candles]
        if any(value is None or value <= 0 for value in opens + highs + lows + closes):
            return None

        o = [float(v) for v in opens]
        h = [float(v) for v in highs]
        l = [float(v) for v in lows]
        c = [float(v) for v in closes]

        elr_1 = [log_ratio(high, low) for high, low in zip(h, l)]
        mean_elr_1 = mean(elr_1)
        if mean_elr_1 <= 0:
            return None

        aelr: Dict[int, float] = {}
        for horizon in horizons:
            window_elr = []
            for idx in range(horizon - 1, len(c)):
                high_n = max(h[idx - horizon + 1: idx + 1])
                low_n = min(l[idx - horizon + 1: idx + 1])
                window_elr.append(log_ratio(high_n, low_n))
            if not window_elr:
                return None
            aelr[horizon] = (1.0 / math.sqrt(horizon)) * (mean(window_elr) / mean_elr_1)

        recent_closes = c[80:100]
        recent_elr = elr_1[80:100]
        if len(recent_closes) != 20 or len(recent_elr) != 20:
            return None

        upper = max(recent_closes)
        lower = min(recent_closes)
        band_range = max(upper - lower, 0.0)
        upper_touch_level = upper - (0.25 * band_range)
        lower_touch_level = lower + (0.25 * band_range)
        upper_touches = sum(1 for price in recent_closes if price >= upper_touch_level)
        lower_touches = sum(1 for price in recent_closes if price <= lower_touch_level)

        numerator_slice = elr_1[92:100] if is_hourly else elr_1[95:100]

        return {
            "aelr": aelr,
            "weights": weights,
            "drift": abs(log_ratio(c[99], c[80])) / max(sum(recent_elr), 1e-9),
            "bandwidth": (upper - lower) / c[99],
            "upper_touches": upper_touches,
            "lower_touches": lower_touches,
            "vol_exp": mean(numerator_slice) / max(mean(recent_elr), 1e-9),
        }

    def _compute_vix_metrics(self, vix_close: List[float]) -> dict:
        last_20 = vix_close[80:100]
        last_5 = vix_close[95:100]
        return {
            "zscore": (vix_close[-1] - mean(last_20)) / max(stdev(last_20), 1e-9),
            "slope": mean(last_5) - mean(last_20),
        }

    def _apply_baseline(self, metrics: dict, baseline: Dict[int, float], weights: Dict[int, float]) -> float:
        score = 0.0
        for horizon, aelr_value in metrics["aelr"].items():
            base = baseline.get(horizon, 0.0)
            dev = ((base - aelr_value) / base) if base > 0 else 0.0
            score += weights[horizon] * dev
        return score

    # --- ENTRY ----------------------------------------------------------------
    def _attempt_entry(self):
        trade_date = now_ist().date()
        if self.position["active"]:
            logger.info("[Entry] Position already active. Skipping.")
            return

        if self._full_exit_date == trade_date:
            logger.info("[Entry] No re-entry allowed after full exit on %s.", trade_date.isoformat())
            return

        if self._universe_built_for != trade_date:
            self._build_candidate_universe()

        if not self.selected_stock:
            logger.info("[Entry] No SIDEWAYS stock available at entry time.")
            return

        with self._entry_lock:
            if self.position["active"]:
                return
            self._enter_condor(self.selected_stock, is_recenter=False)

    def _enter_condor(self, stock: dict, is_recenter: bool):
        chain = self.api.get_option_chain(
            identifier=stock["identifier"],
            expiry_date=stock["monthly_expiry"],
            option_type=["CE", "PE"],
            moneyness=["ATM", "ITM", "OTM"],
        )
        strikes = chain.get("strikes", []) if isinstance(chain, dict) else []
        if not strikes:
            raise ValueError(f"No option chain available for {stock['identifier']}")

        spot = float(chain.get("underlyingPrice") or 0.0)
        if spot <= 0:
            raise ValueError(f"Missing underlying price for {stock['identifier']}")

        atm_iv = self._resolve_atm_iv(strikes, spot)
        expiry_date = datetime.date.fromisoformat(stock["monthly_expiry"])
        days_to_expiry = max((expiry_date - now_ist().date()).days, 1)
        time_to_expiry = days_to_expiry / 365.0
        sigma0 = spot * atm_iv * math.sqrt(time_to_expiry)
        if sigma0 <= 0:
            raise ValueError(f"Invalid sigma0 for {stock['identifier']}")

        target_short_call = spot + sigma0
        target_long_call = spot + (1.5 * sigma0)
        target_short_put = spot - sigma0
        target_long_put = spot - (1.5 * sigma0)

        calls = sorted((opt for opt in strikes if opt.get("optionType") == "CE"), key=lambda item: float(item["strikePrice"]))
        puts = sorted((opt for opt in strikes if opt.get("optionType") == "PE"), key=lambda item: float(item["strikePrice"]))

        short_call = self._closest_option(calls, target_short_call)
        long_call = self._closest_farther_option(calls, target_long_call, float(short_call["strikePrice"]), higher=True)
        short_put = self._closest_option(puts, target_short_put)
        long_put = self._closest_farther_option(puts, target_long_put, float(short_put["strikePrice"]), higher=False)

        lot_size = self._resolve_lot_size([short_call, long_call, short_put, long_put])
        entry_credit = self._net_credit(short_call, long_call, short_put, long_put)
        if entry_credit <= 0:
            raise ValueError(f"Net credit is non-positive for {stock['identifier']}")

        width = max(
            float(long_call["strikePrice"]) - float(short_call["strikePrice"]),
            float(short_put["strikePrice"]) - float(long_put["strikePrice"]),
        )
        max_loss_per_share = max(width - entry_credit, 1.0)
        risk_per_lot = max_loss_per_share * lot_size
        sizing_capital = self._get_position_sizing_capital()
        lots = int(sizing_capital // risk_per_lot)
        if lots < 1:
            raise ValueError(
                f"Sizing capital is insufficient for one lot of {stock['identifier']}"
            )

        quantity = lots * lot_size
        logger.info(
            "[Entry] %s | expiry=%s | spot=%.2f | iv=%.4f | sigma0=%.2f | qty=%d | sizingCapital=%.2f",
            stock["identifier"],
            stock["monthly_expiry"],
            spot,
            atm_iv,
            sigma0,
            quantity,
            sizing_capital,
        )

        legs = [
            (short_call, self._tx_sell()),
            (long_call, self._tx_buy()),
            (short_put, self._tx_sell()),
            (long_put, self._tx_buy()),
        ]
        for leg, side in legs:
            self._place_market_order(leg, quantity, side)

        underlying = {
            "exchange": stock["exchange"],
            "trading_symbol": stock["trading_symbol"],
            "exchange_token": stock["exchange_token"],
            "identifier": stock["identifier"],
        }

        self._register_stream_instrument(underlying)
        for leg in (short_call, long_call, short_put, long_put):
            self._register_stream_instrument(
                {
                    "exchange": leg["exchange"],
                    "trading_symbol": leg["tradingSymbol"],
                    "exchange_token": int(leg["exchangeToken"]),
                    "identifier": make_identifier(leg["exchange"], leg["tradingSymbol"], int(leg["exchangeToken"])),
                }
            )

        self.position.update(
            {
                "active": True,
                "underlying": underlying,
                "classification": stock["classification"],
                "expiry_date": stock["monthly_expiry"],
                "spot_entry": spot,
                "entry_iv": atm_iv,
                "sigma0": sigma0,
                "entry_credit": entry_credit,
                "lot_size": lot_size,
                "lots": lots,
                "quantity": quantity,
                "recentered": is_recenter,
                "short_call": self._leg_state(short_call),
                "long_call": self._leg_state(long_call),
                "short_put": self._leg_state(short_put),
                "long_put": self._leg_state(long_put),
            }
        )

    def _resolve_atm_iv(self, strikes: List[dict], spot: float) -> float:
        ce = min((opt for opt in strikes if opt.get("optionType") == "CE"), key=lambda opt: abs(float(opt["strikePrice"]) - spot))
        pe = min((opt for opt in strikes if opt.get("optionType") == "PE"), key=lambda opt: abs(float(opt["strikePrice"]) - spot))
        iv_values = []
        ce_iv = float_or_none(ce.get("metrics", {}).get("iv"))
        pe_iv = float_or_none(pe.get("metrics", {}).get("iv"))
        if ce_iv is not None and ce_iv > 0:
            iv_values.append(ce_iv)
        if pe_iv is not None and pe_iv > 0:
            iv_values.append(pe_iv)
        atm_iv = mean(iv_values)
        if atm_iv <= 0:
            raise ValueError("ATM IV is unavailable in option chain metrics.")
        return atm_iv

    def _closest_option(self, options: List[dict], target_strike: float) -> dict:
        return min(options, key=lambda opt: abs(float(opt["strikePrice"]) - target_strike))

    def _closest_farther_option(self, options: List[dict], target_strike: float, anchor_strike: float, higher: bool) -> dict:
        filtered = [
            opt for opt in options
            if (float(opt["strikePrice"]) > anchor_strike if higher else float(opt["strikePrice"]) < anchor_strike)
        ]
        if not filtered:
            raise ValueError("Unable to build a valid hedge wing from listed strikes.")
        return min(filtered, key=lambda opt: abs(float(opt["strikePrice"]) - target_strike))

    def _resolve_lot_size(self, options: List[dict]) -> int:
        for option in options:
            lot_size = int(option.get("lotSize") or 0)
            if lot_size > 0:
                return lot_size

        identifiers = [
            make_identifier(option["exchange"], option["tradingSymbol"], int(option["exchangeToken"]))
            for option in options
        ]
        metrics = self.api.get_instrument_metrics(identifiers=identifiers) or []
        for item in metrics:
            lot_size = int(item.get("lotSize") or 0)
            if lot_size > 0:
                return lot_size
        raise ValueError("Lot size could not be resolved for the selected options.")

    def _net_credit(self, short_call: dict, long_call: dict, short_put: dict, long_put: dict) -> float:
        return (
            float(short_call.get("ltp") or 0.0)
            + float(short_put.get("ltp") or 0.0)
            - float(long_call.get("ltp") or 0.0)
            - float(long_put.get("ltp") or 0.0)
        )

    # --- MANAGEMENT ----------------------------------------------------------
    def _manage_position(self, ts: datetime.datetime):
        with self._management_lock:
            if not self.position["active"]:
                return

            spot = self._current_spot()
            if spot is None:
                return

            sigma0 = float(self.position["sigma0"] or 0.0)
            if sigma0 <= 0:
                return

            signed_move = (spot - float(self.position["spot_entry"])) / sigma0
            current_close_cost = self._current_close_cost()
            current_iv = self._current_position_iv()
            net_delta = self._net_delta()

            short_call_strike = float(self.position["short_call"]["strike"])
            short_put_strike = float(self.position["short_put"]["strike"])

            if current_close_cost is not None and current_close_cost <= float(self.position["entry_credit"]) * PREMIUM_CAPTURE_EXIT:
                self._exit_all_positions("premium capture target hit")
                return

            if current_close_cost is not None and current_close_cost >= float(self.position["entry_credit"]) * PREMIUM_EXPANSION_EXIT:
                self._exit_all_positions("premium expansion exit")
                return

            if abs(signed_move) >= SIGMA_MOVE_EXIT:
                self._exit_all_positions("sigma move exit")
                return

            if net_delta is not None and abs(net_delta) >= DELTA_EXIT_LIMIT:
                self._exit_all_positions("net delta breach")
                return

            if current_iv is not None and current_iv >= float(self.position["entry_iv"]) * IV_EXPANSION_EXIT:
                self._exit_all_positions("iv expansion exit")
                return

            call_gap = abs(short_call_strike - spot) / sigma0
            put_gap = abs(spot - short_put_strike) / sigma0
            near_short = min(call_gap, put_gap) <= SHORT_STRIKE_PROXIMITY

            if (
                not self.position["recentered"]
                and ts.time() <= RECENTER_CUTOFF_TIME
                and (abs(signed_move) >= SIGMA_MOVE_RECENTER or near_short)
            ):
                self._recenter_position()

    def _recenter_position(self):
        underlying = self.position["underlying"]
        if not underlying:
            return

        logger.info("[Manage] Re-centering %s once.", underlying["identifier"])
        selected_snapshot = self.selected_stock or {}
        stock = {
            **selected_snapshot,
            "exchange": underlying["exchange"],
            "trading_symbol": underlying["trading_symbol"],
            "exchange_token": underlying["exchange_token"],
            "identifier": underlying["identifier"],
            "monthly_expiry": self.position["expiry_date"],
            "classification": CLASS_SIDEWAYS,
        }
        self._exit_all_positions("re-center adjustment", mark_full_exit=False)
        self._enter_condor(stock, is_recenter=True)

    def _current_spot(self) -> Optional[float]:
        underlying = self.position["underlying"]
        if not underlying:
            return None
        return self.ltp_cache.get(underlying["identifier"])

    def _current_close_cost(self) -> Optional[float]:
        legs = [
            self.position["short_call"],
            self.position["long_call"],
            self.position["short_put"],
            self.position["long_put"],
        ]
        identifiers = [leg["identifier"] for leg in legs if leg]
        if not identifiers or any(identifier not in self.ltp_cache for identifier in identifiers):
            return None

        return (
            self.ltp_cache[self.position["short_call"]["identifier"]]
            + self.ltp_cache[self.position["short_put"]["identifier"]]
            - self.ltp_cache[self.position["long_call"]["identifier"]]
            - self.ltp_cache[self.position["long_put"]["identifier"]]
        )

    def _current_position_iv(self) -> Optional[float]:
        ivs = []
        for leg_key in ("short_call", "short_put"):
            leg = self.position.get(leg_key)
            if not leg:
                continue
            greeks = self.greeks_cache.get(leg["identifier"], {})
            iv = greeks.get("iv")
            if iv is not None and iv > 0:
                ivs.append(iv)
        if not ivs:
            return None
        return mean(ivs)

    def _net_delta(self) -> Optional[float]:
        total = 0.0
        has_delta = False
        signed_legs = [
            ("short_call", -1.0),
            ("long_call", 1.0),
            ("short_put", -1.0),
            ("long_put", 1.0),
        ]
        for leg_key, direction in signed_legs:
            leg = self.position.get(leg_key)
            if not leg:
                continue
            delta = self.greeks_cache.get(leg["identifier"], {}).get("delta")
            if delta is None:
                continue
            has_delta = True
            total += direction * float(delta)
        if not has_delta:
            return None
        return total

    # --- ORDER HELPERS -------------------------------------------------------
    def _place_market_order(self, instrument: dict, quantity: int, transaction_type):
        with self._order_lock:
            self.api.place_order(
                exchange=instrument["exchange"],
                trading_symbol=instrument["tradingSymbol"],
                exchange_token=int(instrument["exchangeToken"]),
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=getattr(self.api, "ORDER_TYPE_MARKET", "MARKET"),
                product=getattr(self.api, "PRODUCT_NRML", "NRML"),
                validity=getattr(self.api, "VALIDITY_DAY", "DAY"),
            )

    def _exit_all_positions(self, reason: str, mark_full_exit: bool = True):
        with self._exit_lock:
            if not self.position["active"]:
                return

            logger.info("[Exit] %s", reason)
            quantity = int(self.position["quantity"] or 0)
            if quantity > 0:
                exit_legs = [
                    (self.position["short_call"], self._tx_buy()),
                    (self.position["long_call"], self._tx_sell()),
                    (self.position["short_put"], self._tx_buy()),
                    (self.position["long_put"], self._tx_sell()),
                ]
                for leg, side in exit_legs:
                    self._place_market_order(
                        {
                            "exchange": leg["exchange"],
                            "tradingSymbol": leg["trading_symbol"],
                            "exchangeToken": leg["exchange_token"],
                        },
                        quantity,
                        side,
                    )

            self.position.update(
                {
                    "active": False,
                    "underlying": None,
                    "classification": None,
                    "expiry_date": None,
                    "spot_entry": None,
                    "entry_iv": None,
                    "sigma0": None,
                    "entry_credit": None,
                    "lot_size": None,
                    "lots": 0,
                    "quantity": 0,
                    "recentered": False,
                    "roll_count": 0,
                    "short_call": None,
                    "long_call": None,
                    "short_put": None,
                    "long_put": None,
                }
            )
            self.greeks_cache.clear()
            self.ltp_cache.clear()

            if mark_full_exit:
                self._full_exit_date = now_ist().date()

    def _eod_exit(self):
        while self.running:
            if now_ist().time() >= EOD_EXIT_TIME:
                with self._exit_lock:
                    if not self._exited_once:
                        self._exited_once = True
                        if self.position["active"]:
                            self._exit_all_positions("EOD exit")
                        else:
                            try:
                                self.api.exit_strategy_positions()
                            except Exception as exc:
                                logger.warning("[EOD] SDK-level exit failed: %s", exc)
                        self.running = False
                break
            time.sleep(5)

    def _position_monitor(self):
        while self.running:
            try:
                positions = self.api.get_open_positions()
                logger.info("[Monitor] Open positions: %d", len(positions))
            except Exception as exc:
                logger.warning("[Monitor] Error fetching positions: %s", exc)
            time.sleep(300)

    # --- STREAM REGISTRATION -------------------------------------------------
    def _register_stream_instrument(self, instrument: dict):
        identifier = instrument["identifier"]
        self.instruments[identifier] = {
            "exchange": instrument["exchange"],
            "trading_symbol": instrument["trading_symbol"],
            "exchange_token": instrument["exchange_token"],
            "position": 0,
            "entry_price": None,
            "last_bar": None,
            "candle": {"open": [], "high": [], "low": [], "close": []},
        }
        if self.ws:
            try:
                self.ws.subscribe([identifier])
            except Exception as exc:
                logger.warning("[Subscribe] Failed for %s: %s", identifier, exc)

    # --- STATE HELPERS -------------------------------------------------------
    def _leg_state(self, option: dict) -> dict:
        identifier = make_identifier(option["exchange"], option["tradingSymbol"], int(option["exchangeToken"]))
        return {
            "identifier": identifier,
            "exchange": option["exchange"],
            "trading_symbol": option["tradingSymbol"],
            "exchange_token": int(option["exchangeToken"]),
            "strike": float(option["strikePrice"]),
            "entry_premium": float(option.get("ltp") or 0.0),
        }

    def _tx_buy(self):
        return getattr(self.api, "TRANSACTION_TYPE_BUY", "BUY")

    def _tx_sell(self):
        return getattr(self.api, "TRANSACTION_TYPE_SELL", "SELL")


# --- ENTRYPOINT --------------------------------------------------------------
def main():
    quotes = QuotesClient()
    api = WizzerClient()
    strategy = Strategy(quotes, api, WATCHLIST)
    strategy.run()


if __name__ == "__main__":
    main()
