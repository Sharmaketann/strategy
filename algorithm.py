from AlgorithmImports import *
from QuantConnect.Algorithm.Framework.Selection import WizzerUniverseSelectionModel
import numpy as np
from collections import defaultdict, deque
from datetime import timedelta
import math

class ReversalStrategy(QCAlgorithm):
    
    def Initialize(self):
        # Dates, currency, capital configured in config.yml (v2)
        
        # Set timezone and account settings
        self.SetTimeZone("Asia/Kolkata")
        
        # Strategy parameters - UPDATED
        self.TOTAL_CAPITAL = 1_000_000  # ₹10L total capital
        self.MAX_POSITIONS = 10         # Maximum 10 positions
        self.PER_POSITION_CAPITAL = self.TOTAL_CAPITAL / self.MAX_POSITIONS  # ₹1L per position
        self.STOP_LOSS_PCT = 0.01  # 1% hard stop-loss
        self.MIN_RDSKEW_RETURNS = 20  # Minimum returns for RDSKEW calculation
        self.BOCPD_SEED_PERIODS = 51  # Periods for BOCPD initialization
        self.RSI_CHANGE_THRESHOLD = 20.0  # 20% RSI change threshold for subselection
        
        # Trading session times (IST)
        self.TRADING_START_HOUR = 9
        self.TRADING_START_MINUTE = 20
        self.TRADING_END_HOUR = 14
        self.TRADING_END_MINUTE = 40
        
        # Universe selection with screening criteria - runs at 9:10 AM
        universe_model = WizzerUniverseSelectionModel(
            filters={
                "$and": [
                    {"hmdClose": {"$gt": 50}},      # Daily Close > 50
                    {"hmdClose": {"$lt": 1000}},    # Daily Close < 1000
                    {"hmdVolume": {"$gt": 2000000}},  # Volume > 20L
                    {"marketCap": {"$gte": 5}}   # ≥₹500 crores (50 billion)
                ]
            },
            sort=[{"hmdVolume": -1}],  # Sort by volume descending
            limit=500,  # Reasonable universe size for processing
            hour=9,
            minute=5,  # General universe screening at 9:10 AM
            securityType=SecurityType.Equity,
            market=Market.India
        )
        self.SetUniverseSelection(universe_model)
        
        # Data structures
        self.symbols = set()
        self.subselected_symbols = set()  # Symbols passing RSI filter
        self.historical_data = {}  # 5-min and 1-hour data per symbol
        self.rdskew_data = {}      # RDSKEW calculations
        self.bocpd_states = {}     # BOCPD state per symbol
        self.position_data = {}    # Position tracking
        self.profit_thresholds = {} # Dynamic profit targets
        
        # NEW: Daily trading tracking - prevent re-entry
        self.traded_today = set()  # Symbols that have been traded today
        self.last_reset_date = None  # Track when we last reset the traded set
        
        # Universe classification
        self.long_candidates = set()
        self.short_candidates = set()
        
        # Classification state tracking
        self.universe_subselected = False
        self.universe_classified = False
        self.symbols_ready_for_subselection = set()
        self.symbols_ready_for_classification = set()
        
        # 5-minute bar tracking
        self.current_5min_bars = {}
        self.last_5min_period = {}
        
        # Order management
        self.DefaultOrderProperties = OrderProperties()
        self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
        
        # UPDATED: Schedule universe subselection at 9:17 AM (changed from 9:12 AM)
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(9, 17),  # Universe subselection at 9:17 AM
            self.PerformUniverseSubselection
        )
        
        # Schedule RDSKEW calculation and classification at 9:19 AM
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(9, 19),  # RDSKEW calculation and classification at 9:19 AM
            self.PerformRDSKEWClassification
        )
        
        # Schedule end-of-day exit
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(self.TRADING_END_HOUR, self.TRADING_END_MINUTE),
            self.ExitAllPositions
        )
        
        # Warmup for indicators
        self.SetWarmUp(100, Resolution.Minute)
        
        self.Log("📊 SkewEdge Intraday Reversal Strategy initialized - ONE TRADE PER STOCK PER DAY")
        self.Log(f"💰 Capital: ₹{self.TOTAL_CAPITAL:,}, Max Positions: {self.MAX_POSITIONS}, Per Position: ₹{self.PER_POSITION_CAPITAL:,.0f}")
        self.Log(f"🔧 RSI Change Threshold: {self.RSI_CHANGE_THRESHOLD}% (using OPEN prices)")
        self.Log(f"🚫 Re-entry Prevention: Each stock can only trade ONCE per day")
        self.Log(f"⏰ Universe Subselection scheduled at 9:17 AM")
    
    def OnSecuritiesChanged(self, changes: SecurityChanges):
        """Handle universe changes and prepare data for subselection"""
        
        # Skip during warmup
        if self.IsWarmingUp:
            return
        
        # Remove securities
        for removed in changes.RemovedSecurities:
            symbol = removed.Symbol
            if symbol in self.symbols:
                self.symbols.remove(symbol)
                self.Log(f"🗑️ Removed from universe: {symbol.Value}")
                # Clean up data structures
                self.CleanupSymbolData(symbol)
                # Exit position if exists
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol, tag="Universe removal")
                    self.Log(f"🚪 Liquidated due to universe removal: {symbol.Value}")
        
        # Add new securities - prepare for subselection
        for added in changes.AddedSecurities:
            symbol = added.Symbol
            self.symbols.add(symbol)
            self.Log(f"➕ Added to universe: {symbol.Value}")
            
            # Initialize data structures
            self.InitializeSymbolData(symbol)
            
            # UPDATED: Fetch historical data for RSI calculation using OPEN prices
            try:
                self.FetchHistoricalDataForRSI(symbol)
                
                # Mark symbol as ready for subselection
                self.symbols_ready_for_subselection.add(symbol)
                
                self.Log(f"✅ Data prepared for RSI subselection (OPEN prices): {symbol.Value}")
            except Exception as e:
                self.Error(f"❌ Failed to prepare RSI data for {symbol.Value}: {e}")
                self.symbols.discard(symbol)
        
        self.Log(f"🌐 Universe updated: {len(self.symbols)} symbols ready for subselection")
    
    def FetchHistoricalDataForRSI(self, symbol):
        """UPDATED: Fetch 20 days of historical data for RSI calculation using OPEN prices"""
        try:
            # Fetch 20 days of daily data for RSI calculation
            history_daily = self.History(symbol, 30, Resolution.Daily)  # Extra buffer
            
            if not history_daily.empty:
                daily_opens = []  # CHANGED: Now using OPEN prices instead of CLOSE
                for idx, row in history_daily.iterrows():
                    daily_opens.append(float(row['open']))  # CHANGED: 'open' instead of 'close'
                
                self.historical_data[symbol]['daily_opens'] = daily_opens  # CHANGED: Store as 'daily_opens'
                self.Log(f"📈 Fetched {len(daily_opens)} daily OPEN prices for RSI calculation: {symbol.Value}")
            else:
                self.Log(f"⚠️ No daily data available for {symbol.Value}")
                
        except Exception as e:
            self.Error(f"❌ Error fetching daily data for {symbol.Value}: {e}")
            raise
    
    def PerformUniverseSubselection(self):
        """UPDATED: Stage 1.1: Universe subselection based on RSI change at 9:17 AM"""
        if self.IsWarmingUp:
            return
        
        self.Log("🔍 Starting Universe Subselection (Stage 1.1) at 9:17 AM using OPEN prices")
        self.Log(f"📊 Evaluating {len(self.symbols_ready_for_subselection)} symbols for RSI change filter")
        
        # Calculate RSI for all symbols and filter based on change
        for symbol in self.symbols_ready_for_subselection.copy():
            try:
                if self.CheckRSIChangeFilter(symbol):
                    self.subselected_symbols.add(symbol)
                    # Now fetch additional data for RDSKEW calculation
                    self.FetchHistoricalDataForRDSKEW(symbol)
                    self.ComputeMaxLogReturn(symbol)
                    self.InitializeBOCPD(symbol)
                    self.symbols_ready_for_classification.add(symbol)
                    self.Log(f"✅ {symbol.Value} passed RSI filter and prepared for RDSKEW")
                else:
                    self.Log(f"❌ {symbol.Value} filtered out by RSI change criteria")
                    
            except Exception as e:
                self.Error(f"❌ Failed RSI subselection for {symbol.Value}: {e}")
                self.symbols_ready_for_subselection.discard(symbol)
        
        self.universe_subselected = True
        self.Log(f"✅ Universe subselection completed: {len(self.subselected_symbols)} symbols selected from {len(self.symbols)} total")
    
    def CheckRSIChangeFilter(self, symbol):
        """UPDATED: Check if symbol passes RSI change filter (>20% change) using OPEN prices"""
        if symbol not in self.historical_data:
            return False
        
        daily_opens = self.historical_data[symbol].get('daily_opens', [])  # CHANGED: Use 'daily_opens'
        
        if len(daily_opens) < 12:  # Need at least 12 days for RSI(10) calculation
            self.Log(f"⚠️ Insufficient data for RSI calculation: {symbol.Value} ({len(daily_opens)} days)")
            return False
        
        # UPDATED: Calculate RSI using OPEN prices for current day (t) and previous day (t-1)
        rsi_current = self.CalculateRSI(daily_opens, 10)  # RSI for day t using OPEN prices
        rsi_previous = self.CalculateRSI(daily_opens[:-1], 10)  # RSI for day t-1 using OPEN prices
        
        if rsi_current is None or rsi_previous is None:
            self.Log(f"⚠️ RSI calculation failed for {symbol.Value}")
            return False
        
        # Calculate RSI change percentage
        if rsi_previous != 0:
            rsi_change_pct = abs((rsi_current - rsi_previous) / rsi_previous) * 100
        else:
            rsi_change_pct = 0
        
        self.Log(f"📊 RSI Analysis (OPEN prices) {symbol.Value}: Current={rsi_current:.2f}, Previous={rsi_previous:.2f}, Change={rsi_change_pct:.2f}%")
        
        # Filter: RSI change > threshold
        passed = rsi_change_pct > self.RSI_CHANGE_THRESHOLD
        if passed:
            self.Log(f"✅ {symbol.Value} RSI change {rsi_change_pct:.2f}% > {self.RSI_CHANGE_THRESHOLD}% threshold (OPEN prices)")
        else:
            self.Log(f"❌ {symbol.Value} RSI change {rsi_change_pct:.2f}% <= {self.RSI_CHANGE_THRESHOLD}% threshold (OPEN prices)")
        
        return passed
    
    def CalculateRSI(self, prices, period=10):
        """Calculate RSI for given price series (now works with OPEN prices)"""
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes (OPEN-to-OPEN changes)
        deltas = []
        for i in range(1, len(prices)):
            deltas.append(prices[i] - prices[i-1])
        
        if len(deltas) < period:
            return None
        
        # Separate gains and losses
        gains = [max(delta, 0) for delta in deltas]
        losses = [abs(min(delta, 0)) for delta in deltas]
        
        # Calculate initial averages (SMA for first calculation)
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        # Calculate RSI
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def FetchHistoricalDataForRDSKEW(self, symbol):
        """Fetch 7 days of historical data for RDSKEW and BOCPD"""
        try:
            # Fetch 5-minute data for RDSKEW and BOCPD
            history_5min = self.History(symbol, 7 * 24 * 12, Resolution.Minute5)  # 7 days of 5-min bars
            
            if not history_5min.empty:
                closes_5min = []
                for idx, row in history_5min.iterrows():
                    if isinstance(idx, tuple):
                        timestamp = idx[1]
                    else:
                        timestamp = idx
                    closes_5min.append(float(row['close']))
                
                self.historical_data[symbol]['five_min_closes'] = closes_5min
                self.Log(f"📈 Fetched {len(closes_5min)} 5-minute closes for RDSKEW: {symbol.Value}")
            
            # Fetch 1-hour data for MAX_LOG_RETURN
            history_1hour = self.History(symbol, 7 * 24, Resolution.Hour)  # 7 days of hourly bars
            
            if not history_1hour.empty:
                closes_1hour = []
                for idx, row in history_1hour.iterrows():
                    closes_1hour.append(float(row['close']))
                
                self.historical_data[symbol]['hourly_closes'] = closes_1hour
                self.Log(f"📈 Fetched {len(closes_1hour)} hourly closes for profit targets: {symbol.Value}")
                
        except Exception as e:
            self.Error(f"❌ Error fetching historical data for {symbol.Value}: {e}")
            raise
    
    def PerformRDSKEWClassification(self):
        """Scheduled function to compute RDSKEW and classify universe at 9:19 AM"""
        if self.IsWarmingUp:
            return
        
        # Only proceed if subselection is completed
        if not self.universe_subselected:
            self.Log("⚠️ Universe subselection not completed, skipping RDSKEW classification")
            return
        
        self.Log("🔍 Starting RDSKEW calculation and universe classification at 9:19 AM")
        self.Log(f"📊 Computing RDSKEW for {len(self.symbols_ready_for_classification)} subselected symbols")
        
        # Compute RDSKEW only for subselected symbols
        for symbol in self.symbols_ready_for_classification.copy():
            try:
                self.ComputeRDSKEW(symbol)
            except Exception as e:
                self.Error(f"❌ Failed to compute RDSKEW for {symbol.Value}: {e}")
                self.symbols_ready_for_classification.discard(symbol)
        
        # Classify universe based on RDSKEW
        if len(self.rdskew_data) > 0:
            self.ClassifyUniverse()
            self.universe_classified = True
            self.Log(f"✅ Universe classification completed at 9:19 AM")
            self.Log(f"📊 Final universe: {len(self.long_candidates)} long candidates, {len(self.short_candidates)} short candidates")
        else:
            self.Log("⚠️ No RDSKEW data available for classification")
    
    def InitializeSymbolData(self, symbol):
        """Initialize data structures for a symbol"""
        self.historical_data[symbol] = {
            'daily_opens': [],         # CHANGED: For RSI calculation using OPEN prices
            'five_min_closes': [],     # For RDSKEW calculation
            'hourly_closes': [],       # For profit targets
            'five_min_returns': deque(maxlen=100),
            'completed_candle_closes': []  # FIXED: Store actual completed candle closes
        }
        
        self.position_data[symbol] = {
            'net_position': 0,
            'entry_price': 0,
            'entry_time': None,
            'completed_candles': 0
        }
        
        self.current_5min_bars[symbol] = {
            'open': 0,
            'high': 0,
            'low': float('inf'),
            'close': 0,
            'period_start': None
        }
        
        self.last_5min_period[symbol] = -1
    
    def CleanupSymbolData(self, symbol):
        """Clean up data structures for removed symbol"""
        data_structures = [
            self.historical_data, self.rdskew_data, self.bocpd_states,
            self.position_data, self.profit_thresholds, self.current_5min_bars,
            self.last_5min_period
        ]
        
        for structure in data_structures:
            structure.pop(symbol, None)
        
        self.subselected_symbols.discard(symbol)
        self.long_candidates.discard(symbol)
        self.short_candidates.discard(symbol)
        self.symbols_ready_for_subselection.discard(symbol)
        self.symbols_ready_for_classification.discard(symbol)
        self.traded_today.discard(symbol)  # NEW: Clean up daily trading tracker
    
    def ComputeRDSKEW(self, symbol):
        """Compute Return Distribution Skewness (RDSKEW)"""
        closes = self.historical_data[symbol]['five_min_closes']
        
        if len(closes) < self.MIN_RDSKEW_RETURNS + 1:
            self.Log(f"⚠️ Insufficient data for RDSKEW calculation: {symbol.Value} ({len(closes)} closes)")
            return
        
        # Compute 5-minute log returns
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                log_return = math.log(closes[i] / closes[i-1])
                log_returns.append(log_return)
        
        if len(log_returns) < self.MIN_RDSKEW_RETURNS:
            self.Log(f"⚠️ Insufficient returns for RDSKEW: {symbol.Value} ({len(log_returns)} returns)")
            return
        
        # Calculate RDSKEW = sqrt(N) * S3 / (S2^1.5)
        N = len(log_returns)
        S2 = sum(r**2 for r in log_returns)
        S3 = sum(r**3 for r in log_returns)
        
        if S2 == 0:
            self.Log(f"⚠️ Zero variance for RDSKEW calculation: {symbol.Value}")
            return
        
        rdskew = np.sqrt(N) * S3 / (S2**1.5)
        rdskew = round(rdskew, 5)
        
        self.rdskew_data[symbol] = rdskew
        self.Log(f"📊 RDSKEW for {symbol.Value}: {rdskew:.5f} (N={N}, S2={S2:.6f}, S3={S3:.6f})")
    
    def ComputeMaxLogReturn(self, symbol):
        """Compute maximum 1-hour log return for profit targets"""
        closes = self.historical_data[symbol]['hourly_closes']
        
        if len(closes) < 2:
            # Fallback to 2% as mentioned in specification
            self.profit_thresholds[symbol] = 0.02
            self.Log(f"📊 Profit threshold for {symbol.Value}: 0.02 (fallback - insufficient hourly data)")
            return
        
        # Compute 1-hour log returns
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                log_return = math.log(closes[i] / closes[i-1])
                log_returns.append(log_return)
        
        if log_returns:
            max_log_return = max(log_returns)
            self.profit_thresholds[symbol] = max(max_log_return, 0.01)  # Minimum 1%
            self.Log(f"📊 Profit threshold for {symbol.Value}: {self.profit_thresholds[symbol]:.4f} (max hourly log return)")
        else:
            self.profit_thresholds[symbol] = 0.02
            self.Log(f"📊 Profit threshold for {symbol.Value}: 0.02 (fallback)")
    
    def InitializeBOCPD(self, symbol):
        """Initialize Bayesian Online Change Point Detection with CORRECT implementation"""
        closes = self.historical_data[symbol]['five_min_closes']
        
        if len(closes) < self.BOCPD_SEED_PERIODS + 1:
            self.Log(f"⚠️ Insufficient data for BOCPD initialization: {symbol.Value} ({len(closes)} closes)")
            return
        
        # Compute initial returns for seeding
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                log_return = math.log(closes[i] / closes[i-1])
                returns.append(log_return)
        
        if len(returns) < self.BOCPD_SEED_PERIODS:
            self.Log(f"⚠️ Insufficient returns for BOCPD: {symbol.Value} ({len(returns)} returns)")
            return
        
        # Initialize CORRECT BOCPD state
        self.bocpd_states[symbol] = BocpdStateCorrect(lambda_param=10.0)  # λ = 10, hazard = 0.1
        
        # Seed with last 51 returns
        seed_returns = returns[-self.BOCPD_SEED_PERIODS:]
        initial_sigma = np.std(seed_returns) if len(seed_returns) > 1 else 0.01
        
        for ret in seed_returns:
            standardized_return = ret / (initial_sigma + 1e-3)
            p_cp, mu_best = self.bocpd_states[symbol].update(standardized_return)
        
        # Store recent returns for pattern analysis
        self.historical_data[symbol]['five_min_returns'].extend(seed_returns[-5:])
        
        # FIXED: Initialize completed candle closes list with last historical close
        self.historical_data[symbol]['completed_candle_closes'] = [closes[-1]]
        
        self.Log(f"🧠 CORRECT BOCPD initialized for {symbol.Value} with {len(seed_returns)} returns, sigma: {initial_sigma:.6f}")
        self.Log(f"📊 Initial completed candle close for {symbol.Value}: {closes[-1]:.2f}")
    
    def ClassifyUniverse(self):
        """Classify symbols into long/short candidates based on RDSKEW percentiles"""
        if not self.rdskew_data:
            self.Log("⚠️ No RDSKEW data available for classification")
            return
        
        rdskew_values = list(self.rdskew_data.values())
        
        if len(rdskew_values) < 3:  # Need minimum symbols for percentiles
            self.Log(f"⚠️ Insufficient symbols for percentile classification: {len(rdskew_values)}")
            return
        
        # Compute 10th and 90th percentiles
        rdskew_sorted = sorted(rdskew_values)
        n = len(rdskew_sorted)
        
        p10_idx = max(0, int(0.1 * n) - 1)
        p90_idx = min(n - 1, int(0.9 * n))
        
        p10 = rdskew_sorted[p10_idx]
        p90 = rdskew_sorted[p90_idx]
        
        self.Log(f"📊 RDSKEW Percentiles: P10={p10:.5f}, P90={p90:.5f} from {n} symbols")
        
        # Classify symbols
        self.long_candidates.clear()
        self.short_candidates.clear()
        
        for symbol, rdskew in self.rdskew_data.items():
            if rdskew < p10:
                self.long_candidates.add(symbol)
                self.Log(f"🟢 Long candidate: {symbol.Value} (RDSKEW={rdskew:.5f} < P10={p10:.5f})")
            elif rdskew > p90:
                self.short_candidates.add(symbol)
                self.Log(f"🔴 Short candidate: {symbol.Value} (RDSKEW={rdskew:.5f} > P90={p90:.5f})")
            else:
                self.Log(f"⚪ Neutral: {symbol.Value} (RDSKEW={rdskew:.5f})")
        
        self.Log(f"✅ Universe classified: {len(self.long_candidates)} long candidates, "
                f"{len(self.short_candidates)} short candidates")
        
        # Log candidate symbols for verification
        if self.long_candidates:
            long_symbols = [s.Value for s in list(self.long_candidates)[:5]]  # Show first 5
            self.Log(f"🟢 Long candidates (sample): {long_symbols}")
        
        if self.short_candidates:
            short_symbols = [s.Value for s in list(self.short_candidates)[:5]]  # Show first 5
            self.Log(f"🔴 Short candidates (sample): {short_symbols}")
    
    def OnData(self, data: Slice):
        """Process market data and manage positions"""
        if self.IsWarmingUp:
            return
        
        # NEW: Reset daily trading tracker at start of new day
        current_date = self.Time.date()
        if self.last_reset_date != current_date:
            self.traded_today.clear()
            self.last_reset_date = current_date
            self.Log(f"🔄 Daily reset: Cleared traded_today set for {current_date}")
        
        # Check if in trading hours
        current_time = self.Time
        if not self.IsInTradingHours(current_time):
            return
        
        # Only process trading logic if universe has been classified
        if not self.universe_classified:
            return
        
        # Process only subselected symbols
        for symbol in self.subselected_symbols:
            if not data.ContainsKey(symbol):
                continue
            
            try:
                self.ProcessSymbolData(symbol, data[symbol])
            except Exception as e:
                self.Error(f"❌ Error processing {symbol.Value}: {e}")
    
    def IsInTradingHours(self, current_time):
        """Check if current time is within trading hours"""
        start_time = current_time.replace(
            hour=self.TRADING_START_HOUR, 
            minute=self.TRADING_START_MINUTE, 
            second=0, 
            microsecond=0
        )
        end_time = current_time.replace(
            hour=self.TRADING_END_HOUR, 
            minute=self.TRADING_END_MINUTE, 
            second=0, 
            microsecond=0
        )
        
        return start_time <= current_time < end_time
    
    def ProcessSymbolData(self, symbol, bar):
        """Process individual symbol data and update 5-minute bars"""
        current_price = float(bar.Close)
        current_minute = self.Time.minute
        
        # Calculate 5-minute period
        period = current_minute // 5
        
        # Initialize or update current 5-minute bar
        if symbol not in self.current_5min_bars:
            return
        
        current_bar = self.current_5min_bars[symbol]
        
        # Check if new 5-minute period started
        if period != self.last_5min_period.get(symbol, -1):
            # Complete previous bar if exists
            if self.last_5min_period.get(symbol, -1) != -1:
                # FIXED: Pass the close price of the completed candle
                completed_close = current_bar['close']
                self.Complete5MinuteBar(symbol, completed_close)
            
            # Start new bar
            current_bar['open'] = current_price
            current_bar['high'] = current_price
            current_bar['low'] = current_price
            current_bar['close'] = current_price
            current_bar['period_start'] = self.Time
            
            self.last_5min_period[symbol] = period
        else:
            # Update current bar
            current_bar['high'] = max(current_bar['high'], current_price)
            current_bar['low'] = min(current_bar['low'], current_price)
            current_bar['close'] = current_price
    
    def Complete5MinuteBar(self, symbol, close_price):
        """Complete a 5-minute bar and update CORRECT BOCPD"""
        if symbol not in self.bocpd_states:
            return
        
        # FIXED: Get previous close from completed candle closes list
        completed_closes = self.historical_data[symbol]['completed_candle_closes']
        
        if len(completed_closes) == 0:
            self.Log(f"⚠️ No previous completed candles for {symbol.Value}")
            return
        
        prev_close = completed_closes[-1]  # Last completed candle close
        
        if prev_close <= 0:
            self.Log(f"⚠️ Invalid previous close for {symbol.Value}: {prev_close}")
            return
        
        # FIXED: Add current completed close to the list
        completed_closes.append(close_price)
        
        # Keep only last 100 completed closes for memory management
        if len(completed_closes) > 100:
            completed_closes.pop(0)
        
        # Compute log return using actual completed candle closes
        log_return = math.log(close_price / prev_close)
        
        # Add to returns deque
        self.historical_data[symbol]['five_min_returns'].append(log_return)
        
        # Update CORRECT BOCPD
        bocpd_state = self.bocpd_states[symbol]
        
        # Standardize return
        recent_returns = list(self.historical_data[symbol]['five_min_returns'])
        if len(recent_returns) > 1:
            sigma = np.std(recent_returns[-20:]) if len(recent_returns) >= 20 else np.std(recent_returns)
            sigma = max(sigma, 1e-3)
        else:
            sigma = 1e-3
        
        standardized_return = log_return / sigma
        
        # Update with CORRECT BOCPD - returns (p_changepoint, mu_best_runlength)
        p_changepoint, mu_best = bocpd_state.update(standardized_return)
        
        # Increment completed candles
        self.position_data[symbol]['completed_candles'] += 1
        
        # Enhanced BOCPD logging with actual candle data
        self.Log(f"🧠 CORRECT BOCPD Update {symbol.Value}: "
                f"Prev_Close={prev_close:.2f}, Current_Close={close_price:.2f}, "
                f"Return={log_return:.6f}, Std_Return={standardized_return:.4f}, "
                f"P_Changepoint={p_changepoint:.4f}, Mu_Best={mu_best:.4f}, "
                f"Candles={self.position_data[symbol]['completed_candles']}")
        
        # Check entry/exit conditions
        self.CheckEntryExit(symbol)
    
    def CheckEntryExit(self, symbol):
        """Check entry and exit conditions for a symbol"""
        if symbol not in self.bocpd_states:
            return
        
        position_info = self.position_data[symbol]
        current_position = self.Portfolio[symbol].Quantity
        
        # Update net position
        position_info['net_position'] = current_position
        
        # Check exit conditions first
        if current_position != 0:
            self.CheckExitConditions(symbol)
        
        # Check entry conditions (with daily trading restriction)
        elif self.CanEnterNewPosition() and symbol not in self.traded_today:  # NEW: Check if not traded today
            self.CheckEntryConditions(symbol)
    
    def CanEnterNewPosition(self):
        """Check if we can enter a new position"""
        open_positions = sum(1 for symbol in self.subselected_symbols 
                           if self.Portfolio[symbol].Invested)
        can_enter = open_positions < self.MAX_POSITIONS
        
        if not can_enter:
            self.Log(f"🚫 Cannot enter new position: {open_positions}/{self.MAX_POSITIONS} positions occupied")
        
        return can_enter
    
    def CheckEntryConditions(self, symbol):
        """Check if entry conditions are met"""
        position_info = self.position_data[symbol]
        
        # NEW: Check if already traded today
        if symbol in self.traded_today:
            self.Log(f"🚫 {symbol.Value}: Already traded today, no re-entry allowed")
            return
        
        # Must have completed at least 2 candles
        if position_info['completed_candles'] < 2:
            self.Log(f"⏳ {symbol.Value}: Waiting for more candles ({position_info['completed_candles']}/2)")
            return
        
        # Must have CORRECT BOCPD state
        if symbol not in self.bocpd_states:
            self.Log(f"⚠️ {symbol.Value}: No BOCPD state available")
            return
        
        bocpd_state = self.bocpd_states[symbol]
        
        # Get CORRECT BOCPD outputs
        p_changepoint = bocpd_state.get_changepoint_probability()
        mu_best = bocpd_state.get_map_mean()
        
        # Enhanced entry condition logging
        self.Log(f"🔍 Entry Check {symbol.Value}: P_CP={p_changepoint:.4f}, Mu_Best={mu_best:.4f}, "
                f"Long_Candidate={symbol in self.long_candidates}, Short_Candidate={symbol in self.short_candidates}, "
                f"Traded_Today={symbol in self.traded_today}")
        
        # Check long entry conditions
        if symbol in self.long_candidates and mu_best > 0:
            self.Log(f"🟢 Long entry signal for {symbol.Value}: BOCPD_mu={mu_best:.4f} > 0, P_CP={p_changepoint:.4f}")
            self.EnterLongPosition(symbol)
        
        # Check short entry conditions
        elif symbol in self.short_candidates and mu_best < 0:
            self.Log(f"🔴 Short entry signal for {symbol.Value}: BOCPD_mu={mu_best:.4f} < 0, P_CP={p_changepoint:.4f}")
            self.EnterShortPosition(symbol)
        
        else:
            # Log why no entry
            if symbol in self.long_candidates:
                self.Log(f"⚪ {symbol.Value} (Long candidate): BOCPD_mu={mu_best:.4f} <= 0, no entry")
            elif symbol in self.short_candidates:
                self.Log(f"⚪ {symbol.Value} (Short candidate): BOCPD_mu={mu_best:.4f} >= 0, no entry")
    
    def EnterLongPosition(self, symbol):
        """Enter a long position"""
        current_price = self.Securities[symbol].Price
        if current_price <= 0:
            self.Log(f"⚠️ Invalid price for long entry {symbol.Value}: {current_price}")
            return
        
        # Calculate position size using equi-weight allocation
        quantity = int(self.PER_POSITION_CAPITAL / current_price)
        
        if quantity <= 0:
            self.Log(f"⚠️ Insufficient capital for long entry {symbol.Value}: ₹{self.PER_POSITION_CAPITAL:,.0f} / ₹{current_price:.2f} = {quantity}")
            return
        
        try:
            ticket = self.MarketOrder(symbol, quantity, tag="Long Entry")
            if ticket:
                self.position_data[symbol]['entry_time'] = self.Time
                self.position_data[symbol]['entry_price'] = current_price
                
                # NEW: Mark as traded today
                self.traded_today.add(symbol)
                
                position_value = quantity * current_price
                self.Log(f"🟢 LONG ENTRY: {symbol.Value} | Qty: {quantity} | Price: ₹{current_price:.2f} | Value: ₹{position_value:,.0f}")
                self.Log(f"🚫 {symbol.Value} marked as traded today - no re-entry allowed")
        except Exception as e:
            self.Error(f"❌ Failed to enter long position for {symbol.Value}: {e}")
    
    def EnterShortPosition(self, symbol):
        """Enter a short position"""
        current_price = self.Securities[symbol].Price
        if current_price <= 0:
            self.Log(f"⚠️ Invalid price for short entry {symbol.Value}: {current_price}")
            return
        
        # Calculate position size using equi-weight allocation
        quantity = -int(self.PER_POSITION_CAPITAL / current_price)
        
        if quantity >= 0:
            self.Log(f"⚠️ Insufficient capital for short entry {symbol.Value}: ₹{self.PER_POSITION_CAPITAL:,.0f} / ₹{current_price:.2f} = {abs(quantity)}")
            return
        
        try:
            ticket = self.MarketOrder(symbol, quantity, tag="Short Entry")
            if ticket:
                self.position_data[symbol]['entry_time'] = self.Time
                self.position_data[symbol]['entry_price'] = current_price
                
                # NEW: Mark as traded today
                self.traded_today.add(symbol)
                
                position_value = abs(quantity) * current_price
                self.Log(f"🔴 SHORT ENTRY: {symbol.Value} | Qty: {quantity} | Price: ₹{current_price:.2f} | Value: ₹{position_value:,.0f}")
                self.Log(f"🚫 {symbol.Value} marked as traded today - no re-entry allowed")
        except Exception as e:
            self.Error(f"❌ Failed to enter short position for {symbol.Value}: {e}")
    
    def CheckExitConditions(self, symbol):
        """Check all exit conditions for a position"""
        position_info = self.position_data[symbol]
        current_position = self.Portfolio[symbol].Quantity
        
        if current_position == 0:
            return
        
        entry_price = position_info['entry_price']
        entry_time = position_info['entry_time']
        current_price = self.Securities[symbol].Price
        
        if entry_price <= 0 or current_price <= 0 or not entry_time:
            self.Log(f"⚠️ Invalid exit data for {symbol.Value}: entry_price={entry_price}, current_price={current_price}")
            return
        
        # Calculate P&L
        if current_position > 0:  # Long position
            pnl_pct = (current_price / entry_price - 1) * 100
        else:  # Short position
            pnl_pct = (entry_price / current_price - 1) * 100
        
        # Log current position status
        elapsed_minutes = (self.Time - entry_time).total_seconds() / 60
        self.Log(f"📊 Position Status {symbol.Value}: P&L={pnl_pct:.2f}%, "
                f"Time={elapsed_minutes:.1f}min, Entry=₹{entry_price:.2f}, Current=₹{current_price:.2f}")
        
        # Check stop-loss (1%) - HIGHEST PRIORITY
        if pnl_pct <= -self.STOP_LOSS_PCT * 100:
            self.Log(f"🛑 Stop-loss triggered for {symbol.Value}: P&L={pnl_pct:.2f}% <= -{self.STOP_LOSS_PCT*100}%")
            self.ExitPosition(symbol, "Stop Loss")
            return
        
        # Check profit target
        if self.CheckProfitTarget(symbol, current_position, entry_price, current_price):
            profit_threshold = self.profit_thresholds.get(symbol, 0.02)
            self.Log(f"🎯 Profit target hit for {symbol.Value}: P&L={pnl_pct:.2f}%, Target={profit_threshold*100:.2f}%")
            self.ExitPosition(symbol, "Profit Target")
            return
    
    def CheckProfitTarget(self, symbol, position, entry_price, current_price):
        """Check if profit target is reached"""
        if symbol not in self.profit_thresholds:
            return False
        
        threshold = math.exp(self.profit_thresholds[symbol]) - 1
        
        if position > 0:  # Long position
            target_reached = current_price >= entry_price * (1 + threshold)
        else:  # Short position
            target_reached = current_price <= entry_price * (1 - threshold)
        
        return target_reached
    
    def ExitPosition(self, symbol, reason):
        """Exit a position"""
        current_position = self.Portfolio[symbol].Quantity
        if current_position == 0:
            return
        
        try:
            ticket = self.MarketOrder(symbol, -current_position, tag=f"Exit: {reason}")
            if ticket:
                current_price = self.Securities[symbol].Price
                entry_price = self.position_data[symbol]['entry_price']
                
                if entry_price > 0:
                    if current_position > 0:
                        pnl_pct = (current_price / entry_price - 1) * 100
                        position_type = "LONG"
                    else:
                        pnl_pct = (entry_price / current_price - 1) * 100
                        position_type = "SHORT"
                    
                    position_value = abs(current_position) * current_price
                    elapsed_time = (self.Time - self.position_data[symbol]['entry_time']).total_seconds() / 60
                    
                    self.Log(f"🚪 {position_type} EXIT: {symbol.Value} | Reason: {reason} | "
                            f"P&L: {pnl_pct:.2f}% | Value: ₹{position_value:,.0f} | Time: {elapsed_time:.1f}min")
                
                # Reset position data (but keep symbol in traded_today set)
                self.position_data[symbol]['entry_price'] = 0
                self.position_data[symbol]['entry_time'] = None
                
        except Exception as e:
            self.Error(f"❌ Failed to exit position for {symbol.Value}: {e}")
    
    def ExitAllPositions(self):
        """Exit all positions at end of day - FIXED to handle all open positions"""
        if self.IsWarmingUp:
            return

        positions_exited = 0
        total_pnl = 0
        
        # NEW: Exit ALL open positions regardless of universe membership
        for kvp in self.Portfolio:
            symbol = kvp.Key
            holding = kvp.Value
            
            if holding.Invested:
                try:
                    # Calculate P&L before exit
                    if symbol in self.position_data and self.position_data[symbol]['entry_price'] > 0:
                        entry_price = self.position_data[symbol]['entry_price']
                        current_price = self.Securities[symbol].Price
                        
                        if current_price > 0:
                            if holding.Quantity > 0:  # Long position
                                pnl_pct = (current_price / entry_price - 1) * 100
                            else:  # Short position
                                pnl_pct = (entry_price / current_price - 1) * 100
                            
                            total_pnl += pnl_pct
                            self.Log(f"🔚 EOD Exit {symbol.Value}: P&L={pnl_pct:.2f}%")
                    
                    # Exit the position
                    self.Liquidate(symbol, tag="EOD Exit")
                    positions_exited += 1
                    
                except Exception as e:
                    self.Error(f"❌ Failed to exit {symbol.Value} at EOD: {e}")

        # Log daily summary
        total_portfolio_value = self.Portfolio.TotalPortfolioValue
        cash = self.Portfolio.Cash
        holdings_value = self.Portfolio.TotalHoldingsValue
        
        self.Log(f"🔚 End-of-day: {positions_exited} positions exited")
        self.Log(f"📊 Daily Summary: Portfolio=₹{total_portfolio_value:,.0f}, Cash=₹{cash:,.0f}, Holdings=₹{holdings_value:,.0f}")
        self.Log(f"📈 Total P&L from exited positions: {total_pnl:.2f}%")
        self.Log(f"🚫 Traded today: {len(self.traded_today)} symbols (will reset tomorrow)")


class BocpdStateCorrect:
    """
    Mathematically Correct Bayesian Online Change Point Detection
    
    Uses Normal-Gamma prior with Student-t predictive distribution
    NumPy-only implementation with manual gamma function approximation
    """
    
    def __init__(self, lambda_param=10.0, mu0=0.0, kappa0=1.0, alpha0=3.0, beta0=1e-3, max_runlength=200):
        """
        Initialize BOCPD with Normal-Gamma prior
        
        Args:
            lambda_param: Expected run length (hazard = 1/lambda_param)
            mu0: Prior mean
            kappa0: Prior precision scaling
            alpha0: Prior shape parameter
            beta0: Prior rate parameter
            max_runlength: Maximum run length to prevent memory explosion
        """
        # Hyperparameters
        self.hazard = 1.0 / lambda_param  # Hazard rate = 1/λ
        self.lambda_param = lambda_param
        
        # Prior parameters (Normal-Gamma)
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        
        # State arrays (vectorized)
        self.R = np.array([1.0])  # Run-length posterior
        self.mu = np.array([mu0])  # Posterior means
        self.kappa = np.array([kappa0])  # Posterior precisions
        self.alpha = np.array([alpha0])  # Posterior shapes
        self.beta = np.array([beta0])  # Posterior rates
        
        # Memory management
        self.max_runlength = max_runlength
        self.min_probability = 1e-10  # Numerical stability
        
        # Tracking
        self.t = 0  # Time step
        
    def update(self, x):
        """
        Update BOCPD with new observation
        
        Args:
            x: New observation (standardized return)
            
        Returns:
            tuple: (p_changepoint, mu_best_runlength)
        """
        self.t += 1
        
        # Compute Student-t predictive probabilities (vectorized)
        pred_probs = self._compute_predictive_probabilities(x)
        
        # Compute growth probabilities (run-length continues)
        growth_probs = self.R * (1 - self.hazard) * pred_probs
        
        # Compute changepoint probability (new run-length = 0)
        cp_prob = np.sum(self.R * self.hazard * pred_probs)
        
        # Normalize probabilities (log-space for stability)
        log_growth = np.log(np.maximum(growth_probs, self.min_probability))
        log_cp = np.log(max(cp_prob, self.min_probability))
        
        # Log-sum-exp trick for numerical stability
        max_log = max(np.max(log_growth), log_cp)
        norm_growth = np.exp(log_growth - max_log)
        norm_cp = np.exp(log_cp - max_log)
        total_prob = np.sum(norm_growth) + norm_cp
        
        # Update run-length posterior
        new_R = np.zeros(len(self.R) + 1)
        new_R[0] = norm_cp / total_prob  # Changepoint
        new_R[1:] = norm_growth / total_prob  # Growth
        
        # Update hyperparameters (vectorized Normal-Gamma updates)
        new_mu, new_kappa, new_alpha, new_beta = self._update_hyperparameters(x)
        
        # Truncate if necessary (memory management)
        if len(new_R) > self.max_runlength:
            # Keep highest probability run-lengths
            keep_indices = np.argsort(new_R)[-self.max_runlength:]
            new_R = new_R[keep_indices]
            new_mu = new_mu[keep_indices]
            new_kappa = new_kappa[keep_indices]
            new_alpha = new_alpha[keep_indices]
            new_beta = new_beta[keep_indices]
            
            # Renormalize after truncation
            new_R = new_R / np.sum(new_R)
        
        # Update state
        self.R = new_R
        self.mu = new_mu
        self.kappa = new_kappa
        self.alpha = new_alpha
        self.beta = new_beta
        
        # Return outputs for strategy
        p_changepoint = self.R[0]
        mu_best = self.get_map_mean()
        
        return p_changepoint, mu_best
    
    def _compute_predictive_probabilities(self, x):
        """
        Compute Student-t predictive probabilities (vectorized)
        
        Uses correct Student-t PDF with proper degrees of freedom and scale
        """
        # Student-t parameters
        nu = 2 * self.alpha  # Degrees of freedom
        scale_sq = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)  # Scale squared
        
        # Clip for numerical stability
        nu = np.maximum(nu, 1e-3)
        scale_sq = np.maximum(scale_sq, 1e-6)
        scale = np.sqrt(scale_sq)
        
        # Standardized values
        z = (x - self.mu) / scale
        
        # Log Student-t PDF using manual gamma function approximation
        log_gamma_half_nu_plus_1 = self._log_gamma((nu + 1) / 2)
        log_gamma_half_nu = self._log_gamma(nu / 2)
        log_pi = np.log(np.pi)
        
        log_pdf = (log_gamma_half_nu_plus_1 - log_gamma_half_nu - 
                  0.5 * np.log(nu) - 0.5 * log_pi - np.log(scale) -
                  ((nu + 1) / 2) * np.log(1 + z**2 / nu))
        
        # Convert to probability (with numerical clipping)
        pred_probs = np.exp(np.clip(log_pdf, -50, 50))  # Prevent overflow/underflow
        pred_probs = np.maximum(pred_probs, self.min_probability)
        
        return pred_probs
    
    def _log_gamma(self, z):
        """
        Stirling's approximation for log(Gamma(z)) - NumPy only implementation
        
        For z > 1: log(Γ(z)) ≈ (z-0.5)*log(z) - z + 0.5*log(2π) + 1/(12z)
        """
        z = np.asarray(z)
        
        # Handle edge cases
        result = np.zeros_like(z, dtype=float)
        
        # For very small values, use simple approximation
        small_mask = z <= 1.0
        if np.any(small_mask):
            # For z ≤ 1, use Γ(z+1) = z*Γ(z), so log(Γ(z)) = log(Γ(z+1)) - log(z)
            z_small = z[small_mask] + 1.0
            stirling_small = ((z_small - 0.5) * np.log(z_small) - z_small + 
                            0.5 * np.log(2 * np.pi) + 1.0 / (12.0 * z_small))
            result[small_mask] = stirling_small - np.log(z[small_mask])
        
        # For larger values, use Stirling's approximation directly
        large_mask = z > 1.0
        if np.any(large_mask):
            z_large = z[large_mask]
            result[large_mask] = ((z_large - 0.5) * np.log(z_large) - z_large + 
                                0.5 * np.log(2 * np.pi) + 1.0 / (12.0 * z_large))
        
        return result
    
    def _update_hyperparameters(self, x):
        """
        Update Normal-Gamma hyperparameters (vectorized)
        
        Correct Bayesian updates for μ, κ, α, β
        """
        # Changepoint: reset to prior
        new_mu = np.zeros(len(self.mu) + 1)
        new_kappa = np.zeros(len(self.kappa) + 1)
        new_alpha = np.zeros(len(self.alpha) + 1)
        new_beta = np.zeros(len(self.beta) + 1)
        
        # Run-length 0 (changepoint): prior parameters
        new_mu[0] = self.mu0
        new_kappa[0] = self.kappa0
        new_alpha[0] = self.alpha0
        new_beta[0] = self.beta0
        
        # Run-length > 0 (growth): Bayesian updates
        kappa_new = self.kappa + 1
        mu_new = (self.kappa * self.mu + x) / kappa_new
        alpha_new = self.alpha + 0.5
        
        # Beta update (with numerical stability)
        diff_sq = (x - self.mu)**2
        beta_update = (self.kappa * diff_sq) / (2 * kappa_new)
        beta_new = self.beta + beta_update
        
        # Ensure beta > 0 for numerical stability
        beta_new = np.maximum(beta_new, 1e-6)
        
        new_mu[1:] = mu_new
        new_kappa[1:] = kappa_new
        new_alpha[1:] = alpha_new
        new_beta[1:] = beta_new
        
        return new_mu, new_kappa, new_alpha, new_beta
    
    def get_map_mean(self):
        """Get posterior mean at MAP (Maximum A Posteriori) run-length"""
        if len(self.R) == 0:
            return 0.0
        
        map_idx = np.argmax(self.R)
        return float(self.mu[map_idx])
    
    def get_changepoint_probability(self):
        """Get probability of changepoint (run-length = 0)"""
        return float(self.R[0]) if len(self.R) > 0 else 0.0
    
    def get_expected_mean(self):
        """Get expectation of posterior mean over all run-lengths"""
        if len(self.R) == 0:
            return 0.0
        
        return float(np.sum(self.R * self.mu))
    
    def get_posterior_variance(self):
        """Get posterior variance at MAP run-length"""
        if len(self.R) == 0:
            return 1.0
        
        map_idx = np.argmax(self.R)
        # Posterior variance = β/(α-1) for α > 1
        if self.alpha[map_idx] > 1:
            return float(self.beta[map_idx] / (self.alpha[map_idx] - 1))
        else:
            return 1.0
    
    def get_run_length_distribution(self):
        """Get current run-length posterior distribution"""
        return self.R.copy()
    
    def get_diagnostics(self):
        """Get diagnostic information for debugging"""
        return {
            'time_step': self.t,
            'num_run_lengths': len(self.R),
            'map_run_length': int(np.argmax(self.R)),
            'changepoint_prob': self.get_changepoint_probability(),
            'map_mean': self.get_map_mean(),
            'expected_mean': self.get_expected_mean(),
            'posterior_variance': self.get_posterior_variance(),
            'hazard_rate': self.hazard,
            'lambda_param': self.lambda_param
        }