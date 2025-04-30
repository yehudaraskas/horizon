import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from collections import defaultdict
import pywt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    filename='fx_trader.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handle_trading_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            # Implement circuit breaker pattern
            if args[0].current_balance < args[0].initial_balance * 0.9:
                logger.critical("Account drawdown exceeding 10% - halting trading")
                raise SystemExit("Emergency shutdown triggered")
            return None
    return wrapper

class RealtimeTrader:
    def __init__(self, initial_balance=100000, max_leverage=30):
        # Core parameters
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_leverage = max_leverage
        self.current_leverage = max_leverage / 2  # Start at middle leverage
        
        # Risk parameters
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.max_total_risk = 0.06     # 6% total risk
        self.position_scaling = 1.0     # Dynamic scaling based on performance
        
        # Performance tracking
        self.trades = []
        self.win_rate = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0
        self.position = None
        self.window_size = 20
        self.min_profit_target = 0.001  # 0.1%
        self.max_loss = 0.002      # 0.2%
        
        # Balance tracking
        self.position_size = 100000  # Standard lot size
        
    @handle_trading_errors
    def analyze_with_wavelets(self, prices, wavelet='db4', level=3):
        """Multi-level wavelet analysis and denoising"""
        # Convert to numpy array
        prices = np.array(prices)
        
        # Decompose the signal
        coeffs = pywt.wavedec(prices, wavelet, level=level)
        
        # Analyze each level for trend
        trends = []
        for i, coef in enumerate(coeffs):
            if len(coef) > 1:
                trend = np.mean(np.diff(coef))
                trends.append(trend)
        
        # Weight the trends (higher weights for lower frequencies)
        weights = [0.5, 0.3, 0.2]  # Adjust based on level importance
        weighted_trend = sum(t * w for t, w in zip(trends, weights))
        
        # Denoise the signal
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(prices)))
        coeffs_clean = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        denoised = pywt.waverec(coeffs_clean, wavelet)
        
        return denoised, weighted_trend
    
    @handle_trading_errors
    def calculate_gaussian_channels(self, prices, window=20, num_std=2):
        """Calculate adaptive Gaussian channels"""
        prices = np.array(prices)
        
        # Use exponential weights for more recent prices
        weights = np.exp(np.linspace(-1, 0, window))
        weights /= weights.sum()
        
        # Get last window of prices
        window_prices = prices[-window:]
        
        # Calculate weighted moving average
        ma = np.sum(window_prices * weights)
        
        # Adaptive standard deviation using recent prices
        dev = np.std(window_prices[-5:])  # Recent volatility
        
        # Calculate channels
        upper = ma + num_std * dev
        lower = ma - num_std * dev
        
        # Calculate channel width for leverage adjustment
        channel_width = (upper - lower) / ma  # Normalized width
        
        return ma, upper, lower, channel_width
    
    @handle_trading_errors
    def calculate_dynamic_leverage(self, confidence, wave_trend, channel_width):
        """Calculate optimal leverage based on market conditions"""
        base_leverage = self.max_leverage / 2  # Start at 50% of max leverage
        
        # 1. Confidence-based scaling (30% impact)
        confidence_scale = np.clip(confidence, 0.3, 1.5)
        leverage_from_confidence = base_leverage * confidence_scale
        
        # 2. Trend strength scaling (20% impact)
        trend_strength = min(abs(wave_trend) * 1000, 1.5)
        leverage_from_trend = base_leverage * trend_strength
        
        # 3. Channel width scaling (30% impact)
        # Narrower channels = higher leverage
        channel_scale = 1.0 / (channel_width * 100)
        channel_scale = np.clip(channel_scale, 0.5, 2.0)
        leverage_from_channel = base_leverage * channel_scale
        
        # 4. Performance scaling (20% impact)
        if len(self.trades) > 10:
            recent_trades = self.trades[-10:]
            win_rate = sum(1 for t in recent_trades if t['profit'] > 0) / len(recent_trades)
            performance_scale = np.clip(win_rate * 2, 0.5, 1.5)
            leverage_from_performance = base_leverage * performance_scale
        else:
            leverage_from_performance = base_leverage
        
        # Weighted average of all factors
        final_leverage = (
            0.3 * leverage_from_confidence +
            0.2 * leverage_from_trend +
            0.3 * leverage_from_channel +
            0.2 * leverage_from_performance
        )
        
        # Smooth transitions (max 20% change)
        max_change = self.current_leverage * 0.2
        if abs(final_leverage - self.current_leverage) > max_change:
            direction = 1 if final_leverage > self.current_leverage else -1
            final_leverage = self.current_leverage + (direction * max_change)
        
        # Apply safety bounds and round
        final_leverage = min(self.max_leverage, max(1.0, final_leverage))
        final_leverage = round(final_leverage, 2)
        
        return final_leverage
    
    @handle_trading_errors
    def calculate_position_size(self, confidence, wave_trend, channel_width):
        """Calculate dynamic position size based on multiple factors"""
        # Calculate base position size (2% risk)
        base_size = self.current_balance * self.max_risk_per_trade
        
        # 1. Confidence scaling (40% weight)
        confidence_scale = confidence / 100.0
        size_from_confidence = base_size * confidence_scale
        
        # 2. Trend strength scaling (30% weight)
        trend_scale = min(abs(wave_trend) / 0.01, 1.0)  # Normalize to typical range
        size_from_trend = base_size * trend_scale
        
        # 3. Volatility scaling (30% weight) - inverse relationship
        vol_scale = max(0.2, 1 - channel_width * 10)  # Smaller positions for high volatility
        size_from_volatility = base_size * vol_scale
        
        # Combine factors with weights
        final_size = (
            0.4 * size_from_confidence +
            0.3 * size_from_trend +
            0.3 * size_from_volatility
        )
        
        # Apply dynamic leverage
        leverage = self.calculate_dynamic_leverage(confidence, wave_trend, channel_width)
        leveraged_size = final_size * leverage
        
        # Ensure within account limits
        max_position = self.current_balance * leverage * 0.95  # 95% safety margin
        
        return min(leveraged_size, max_position)
    
    @handle_trading_errors
    def should_enter_trade(self, confidence, signals, wave_trend, channel_width):
        """Determine if we should enter a trade based on signals and risk"""
        # Initialize return values
        should_enter = True
        position_size = 0
        
        # 1. Check total risk exposure
        total_risk = 0
        for pos in self.positions.values():
            risk = abs(pos['size']) / (self.current_balance * self.current_leverage)
            total_risk += risk
        
        # 2. Calculate minimum confidence threshold
        min_confidence = 75
        if len(self.trades) > 10:
            win_rate = sum(1 for t in self.trades[-10:] if t['profit'] > 0) / 10
            min_confidence = max(60, 75 - (win_rate * 15))  # Lower threshold if winning
        
        # 3. Check entry conditions
        if total_risk >= self.max_total_risk:
            should_enter = False
        elif confidence < min_confidence:
            should_enter = False
        elif not signals or len(signals) < 2:
            should_enter = False
        
        # 4. Calculate position size if conditions are met
        if should_enter:
            position_size = self.calculate_position_size(confidence, wave_trend, channel_width)
        
        return should_enter, position_size
        
        return confidence >= min_confidence, position_size
    
    @handle_trading_errors
    def analyze_price_action(self, current_price, recent_prices):
        """Analyze price action to determine trade confidence"""
        # 1. Wavelet Analysis
        denoised_prices, wave_trend = self.analyze_with_wavelets(recent_prices)
        
        # 2. Channel Analysis
        ma, upper, lower, channel_width = self.calculate_gaussian_channels(recent_prices)
        
        # Initialize confidence and signals
        confidence = 50  # Base confidence
        signals = []
        
        # 3. Trend Analysis (30 points max)
        if abs(wave_trend) > 0.001:  # Strong trend
            confidence += 30
            signals.append("🌊 Strong " + ("upward" if wave_trend > 0 else "downward") + " wave")
        elif abs(wave_trend) > 0.0005:  # Moderate trend
            confidence += 20
            signals.append("🌊 Moderate " + ("upward" if wave_trend > 0 else "downward") + " wave")
        
        # 4. Channel Position Analysis (20 points max)
        channel_position = (current_price - ma) / (upper - lower)
        if abs(channel_position) > 1:  # Outside channels
            confidence += 20
            signals.append("📈 Strong " + ("breakout" if channel_position > 0 else "breakdown"))
        elif abs(channel_position) > 0.8:  # Near channel edges
            confidence += 15
            signals.append("📊 Near " + ("upper" if channel_position > 0 else "lower") + " channel")
        
        # 5. Momentum Analysis (20 points max)
        short_momentum = (current_price - recent_prices[-5]) / recent_prices[-5]
        long_momentum = (current_price - recent_prices[-20]) / recent_prices[-20]
        
        if abs(short_momentum) > 0.001 and abs(long_momentum) > 0.002:
            confidence += 20
            signals.append("🚀 Strong sustained momentum")
        elif abs(short_momentum) > 0.0005:
            confidence += 10
            signals.append("📈 Short-term momentum")
        
        # 6. Channel Width Analysis (15 points max)
        if channel_width < 0.001:  # Very tight channel
            confidence += 15
            signals.append("🎯 Very tight channel")
        elif channel_width < 0.002:  # Normal channel
            confidence += 10
            signals.append("🎯 Normal channel")
        
        # 7. Trend-Momentum Alignment (15 points max)
        if (wave_trend > 0 and short_momentum > 0) or (wave_trend < 0 and short_momentum < 0):
            confidence += 15
            signals.append("✨ Trend-momentum alignment")
        
        # Ensure confidence stays within bounds
        # Channel breakout analysis
        if current_price > current_upper:
            confidence += 20
            signals.append("↑ Above upper channel")
        elif current_price < current_lower:
            confidence += 20
            signals.append("↓ Below lower channel")
            
        # Trend analysis using denoised data
        if current_price > current_ma and avg_velocity > 0:
            confidence += 15
            signals.append("↗ Upward trend")
        
        # Trend analysis using denoised data
        if current_price > current_ma and avg_velocity > 0:
            confidence += 20
            signals.append("↗ Upward trend")
        elif current_price < current_ma and avg_velocity < 0:
            confidence += 20
            signals.append("↘ Downward trend")
            
        # Momentum based on denoised data
        
        confidence = min(100, max(0, confidence))
        return confidence, signals, wave_trend, channel_width
        if abs(momentum) > 0.0003:  # Lower threshold due to denoising
            confidence += 20
            signals.append(f"→ Clean momentum: {momentum:.6f}")
            
        # Channel compression/expansion
        channel_width = current_upper - current_lower
        avg_channel_width = np.mean(upper_channel - lower_channel)
        
        if channel_width < avg_channel_width * 0.8:
            confidence += 20
            signals.append("↔ Channel compression")
        elif channel_width > avg_channel_width * 1.2:
            confidence -= 10
            signals.append("↔ Channel expansion")
            
        # Price velocity confirmation
        if (avg_velocity > 0 and current_price > current_ma) or \
           (avg_velocity < 0 and current_price < current_ma):
            confidence += 20
            signals.append("✓ Price-trend alignment")
            
        # Support/Resistance
        recent_high = max(recent_prices[:-1])
        recent_low = min(recent_prices[:-1])
        
        if current_price > recent_high:
            confidence += 10
            signals.append("⇈ Breakout up")
        elif current_price < recent_low:
            confidence += 10
            signals.append("⇊ Breakout down")
            
        # Calculate relative channel width for position sizing
        rel_channel_width = (current_upper - current_lower) / current_price
        
        return max(0, min(100, confidence)), signals, wave_trend, rel_channel_width
    
    @handle_trading_errors
    def calculate_trade_profit(self, entry_price, current_price, direction, position_size):
        """Calculate actual profit/loss in dollars"""
        # Calculate price difference
        price_diff = current_price - entry_price
        if direction == 'short':
            price_diff = -price_diff
        
        # Calculate profit percentage
        profit_pct = price_diff / entry_price
        
        # Calculate actual profit based on position size and leverage
        profit = position_size * profit_pct
        
        # Round to 2 decimal places
        return round(profit, 2)
    
    @handle_trading_errors
    def adjust_active_trade_leverage(self, position, current_price, recent_prices):
        """Adjust leverage during an active trade based on performance and market conditions"""
        # 1. Analyze current market conditions
        confidence, signals, wave_trend, channel_width = self.analyze_price_action(
            current_price, recent_prices[-self.window_size:]
        )
        
        # 2. Calculate current profit
        current_profit = self.calculate_trade_profit(
            position['entry_price'], 
            current_price, 
            position['direction'],
            position['size']
        )
        profit_pct = current_profit / self.current_balance
        
        # 3. Base leverage adjustment on profit
        if profit_pct <= 0:
            # In loss - reduce leverage
            reduction = max(0.5, 1 + (profit_pct * 50))  # Max 50% reduction
            new_leverage = self.current_leverage * reduction
        else:
            # In profit - check conditions for leverage increase
            new_leverage = self.current_leverage
            
            # Check for perfect conditions (TURBO mode)
            perfect_conditions = (
                confidence >= 90 and
                abs(wave_trend) >= 0.001 and
                channel_width <= 0.001 and
                ((wave_trend > 0 and position['direction'] == 'long') or
                 (wave_trend < 0 and position['direction'] == 'short'))
            )
            
            if perfect_conditions:
                # Calculate acceleration factor (max 1.5x)
                accel = min(1 + (profit_pct * 50), 1.5)
                new_leverage = self.max_leverage * accel
            elif confidence >= 80 and profit_pct > 0.001:
                # Good conditions - moderate increase
                new_leverage = min(
                    self.current_leverage * 1.2,
                    self.max_leverage
                )
        
        # 4. Apply safety limits
        new_leverage = max(1, min(new_leverage, self.max_leverage * 1.5))
        
        # 5. Smooth transition (max 20% change)
        max_change = self.current_leverage * 0.2
        if abs(new_leverage - self.current_leverage) > max_change:
            new_leverage = self.current_leverage + (
                max_change if new_leverage > self.current_leverage else -max_change
            )
        
        return round(new_leverage, 2)
        
        # Ensure smooth transition (max 10% change per check)
        max_change = self.current_leverage * 0.1
        if new_leverage > self.current_leverage + max_change:
            new_leverage = self.current_leverage + max_change
        elif new_leverage < self.current_leverage - max_change:
            new_leverage = self.current_leverage - max_change
        
        # Maintain minimum leverage
        new_leverage = max(1, new_leverage)
        
        # Update position size based on new leverage
        old_exposure = position['size'] * self.current_leverage
        new_size = old_exposure / new_leverage
        
        return new_leverage, new_size
    
    @handle_trading_errors
    def should_exit_trade(self, position, current_price, recent_prices):
        """Determine if we should exit the current trade position"""
        # Initialize return values
        should_exit = False
        message = "No position to exit"
        signals = []
        actual_profit = 0
        
        if position:
            # Calculate profit
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
            if position['direction'] == 'short':
                profit_pct = -profit_pct
            
            actual_profit = self.calculate_trade_profit(
                position['entry_price'],
                current_price,
                position['direction'],
                position['size']
            )
            
            # Check stop loss
            if profit_pct <= -self.max_loss:
                should_exit = True
                message = f"⛔ Stop loss: ${actual_profit:.2f}"
            
            # Check take profit conditions
            elif profit_pct >= self.min_profit_target:
                ma_fast = np.mean(recent_prices[-5:])
                
                if (position['direction'] == 'long' and current_price < ma_fast):
                    should_exit = True
                    signals.append("↓ Trend reversal")
                    message = f"✓ Take profit: ${actual_profit:.2f}"
                elif (position['direction'] == 'short' and current_price > ma_fast):
                    should_exit = True
                    signals.append("↑ Trend reversal")
                    message = f"✓ Take profit: ${actual_profit:.2f}"
                else:
                    message = f"Holding {position['direction']} (${actual_profit:.2f})"
            else:
                message = f"Holding {position['direction']} (${actual_profit:.2f})"
        
        return should_exit, message, signals, actual_profit
    
    @handle_trading_errors
    def run_trading_simulation(self, data):
        print("\033[2J\033[H")  # Clear screen
        print("=== Realtime Trading Simulation ===")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Max Leverage: {self.max_leverage}x")
        print(f"Starting Leverage: {self.current_leverage:.1f}x")
        print("Analyzing price action every 0.5 seconds...")
        
        for i in range(self.window_size, len(data)):
            current_price = data.iloc[i]['price']
            recent_prices = data.iloc[i-self.window_size:i]['price'].values
            current_time = data.iloc[i]['timestamp']
            
            print("\033[H")  # Move to top of screen
            print("=== Realtime Trading Simulation ===")
            print(f"Time: {current_time}")
            print(f"Price: {current_price:.5f}")
            print(f"Balance: ${self.current_balance:,.2f} ({((self.current_balance/self.initial_balance)-1)*100:+.2f}%)")
            print(f"Max Drawdown: {self.max_drawdown*100:.2f}%")
            
            if self.position:
                # Analyze market conditions
                confidence, signals, wave_trend, channel_width = self.analyze_price_action(current_price, recent_prices)
                
                # Adjust leverage for active trade
                new_leverage, new_size = self.adjust_active_trade_leverage(
                    self.position, current_price, recent_prices, wave_trend, channel_width)
                
                if new_leverage != self.current_leverage:
                    self.current_leverage = new_leverage
                    self.position['size'] = new_size
                # Check exit conditions
                should_exit, exit_msg, exit_signals, actual_profit = self.should_exit_trade(
                    self.position, current_price, recent_prices)
                
                print("\nPosition Status:")
                print(f"Type: {self.position['direction'].upper()}")
                print(f"Entry: {self.position['entry_price']:.5f}")
                print(f"Size: {self.position_size:,} units")
                print(f"Max Leverage: {self.max_leverage}x")
                print(f"Current Leverage: {self.current_leverage:.1f}x")
                print(f"Current P/L: ${actual_profit:.2f}")
                print(f"Status: {exit_msg}")
                for signal in exit_signals:
                    print(f"Signal: {signal}")
                
                if should_exit:
                    print(f"\n{exit_msg}")
                    self.current_balance += actual_profit
                    self.peak_balance = max(self.peak_balance, self.current_balance)
                    self.max_drawdown = max(self.max_drawdown, 
                                           (self.peak_balance - self.current_balance) / self.peak_balance)
                    
                    # Record trade
                    self.trades.append({
                        'entry_price': self.position['entry_price'],
                        'exit_price': current_price,
                        'profit': actual_profit,
                        'size': self.position['size'],
                        'direction': self.position['direction'],
                        'duration': (current_time - pd.Timestamp(self.position['entry_time'])).total_seconds() / 60
                    })
                    
                    # Update win rate
                    if len(self.trades) > 0:
                        self.win_rate = sum(1 for t in self.trades if t['profit'] > 0) / len(self.trades)
                    
                    # Clear position
                    self.position = None
                    
                    self.position = None
                    time.sleep(0.2)  # Pause to show exit information
            
            else:
                # Analyze price action for new trade
                confidence, signals, wave_trend, channel_width = self.analyze_price_action(current_price, recent_prices)
                
                # Only show update if confidence is notable
                if confidence > 50:
                    print("\nSignal Analysis:")
                    print(f"Confidence: {confidence:.1f}% | Wave: {wave_trend*10000:+.1f} | Channel: {channel_width*10000:.1f}")
                    print("Signals: " + " | ".join(signals))
                    
                    if self.position:
                        print(f"\nPosition: {self.position['direction'].upper()} @ {self.position['entry_price']:.5f}")
                        print(f"Current Leverage: {self.current_leverage:.1f}x")
                
                # Enter trade if confidence is high enough
                if confidence >= 80:
                    # Determine direction based on moving averages
                    ma_fast = np.mean(recent_prices[-5:])
                    direction = 'long' if current_price > ma_fast else 'short'
                    
                    print(f"\n{'='*40}")
                    print(f"ENTER {direction.upper()} TRADE")
                    print(f"{'='*40}")
                    
                    self.position = {
                        'entry_price': current_price,
                        'entry_time': str(current_time),
                        'direction': direction,
                        'entry_confidence': confidence,
                        'entry_channel_width': channel_width,
                        'size': self.position_size
                    }
                    time.sleep(0.1)  # Pause to show entry information
            
            time.sleep(0.05)

# Load and run
def run_live_trading(self, api_key, instrument="EUR_USD", granularity="M1", backtest_months=0):
        """Run live trading using OANDA"""
        from oanda_fetcher import OANDADataFetcher
        
        print("\033[2J\033[H")  # Clear screen
        print("🤖 AAA TRADING SYSTEM - LIVE MODE")
        print("----------------------------")
        print(f"💰 Starting: ${self.initial_balance:,.2f}")
        print(f"📈 Leverage: {self.current_leverage:.1f}x (max {self.max_leverage}x)")
        print("⚡ System Active - Monitoring Market...")
        
        # Initialize OANDA fetcher
        fetcher = OANDADataFetcher(api_key, instrument, granularity)
        
        while True:
            try:
                # For backtest, use historical time range
                if backtest_months > 0:
                    if not hasattr(self, 'backtest_current_time'):
                        self.backtest_current_time = datetime.now() - timedelta(days=backtest_months * 30)
                        self.last_display_update = datetime.now()
                    current_time = self.backtest_current_time
                    start_time = current_time - timedelta(minutes=self.window_size + 5)
                    self.backtest_current_time += timedelta(minutes=1)  # Process exactly 1 minute at a time to match live trading
                else:
                    current_time = datetime.now()
                    start_time = current_time - timedelta(minutes=self.window_size + 5)
                
                # Fetch latest candles
                candles = fetcher.fetch_candles(start_time, current_time, backtest=bool(backtest_months))
                if not candles:
                    print("Error fetching data, retrying...")
                    time.sleep(5)
                    continue
                    
                # Process candles
                df = fetcher.process_candles(candles)
                if len(df) < self.window_size:
                    print("Not enough data points, waiting...")
                    time.sleep(5)
                    continue
                
                # Get current price and recent prices
                current_price = df.iloc[-1]['close']
                recent_prices = df['close'].values[-self.window_size:]
                
                # Update display every 2 seconds in backtest mode
                should_update = False
                if backtest_months > 0:
                    if (datetime.now() - self.last_display_update).total_seconds() >= 2:
                        should_update = True
                        self.last_display_update = datetime.now()
                else:
                    should_update = True
                
                if should_update:
                    print("\033[H")  # Move to top of screen
                    print("🤖 AAA TRADING SYSTEM - BACKTEST" if backtest_months > 0 else "🤖 AAA TRADING SYSTEM - LIVE MODE")
                    print("----------------------------")
                    print(f"⏰ {current_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f"💵 EUR/USD: {current_price:.5f}")
                    profit_pct = ((self.current_balance/self.initial_balance)-1)*100
                    profit_color = '🟢' if profit_pct >= 0 else '🔴'
                    print(f"{profit_color} P/L: {profit_pct:+.2f}% (${self.current_balance:,.2f})")
                    print(f"📉 Max DD: {self.max_drawdown*100:.1f}%")
                    print(f"📊 Processed: {((current_time - (datetime.now() - timedelta(days=backtest_months * 30))).days / (backtest_months * 30) * 100):.1f}%" if backtest_months > 0 else "")
                
                if self.position:
                    # Analyze market conditions
                    confidence, signals, wave_trend, channel_width = self.analyze_price_action(current_price, recent_prices)
                    
                    # Adjust leverage for active trade
                    new_leverage, new_size = self.adjust_active_trade_leverage(
                        self.position, current_price, recent_prices, wave_trend, channel_width)
                    
                    if new_leverage != self.current_leverage:
                        self.current_leverage = new_leverage
                        self.position['size'] = new_size
                        
                    # Check exit conditions
                    should_exit, exit_msg, exit_signals, actual_profit = self.should_exit_trade(
                        self.position, current_price, recent_prices)
                    
                    print("\n📊 ACTIVE TRADE")
                    direction_arrow = '⬆️' if self.position['direction'] == 'long' else '⬇️'
                    print(f"{direction_arrow} {self.position['direction'].upper()} @ {self.position['entry_price']:.5f}")
                    print(f"💪 Size: {self.position['size']:,}")
                    print(f"📈 Leverage: {self.current_leverage:.1f}x")
                    profit_color = '🟢' if actual_profit >= 0 else '🔴'
                    print(f"{profit_color} P/L: ${actual_profit:+.2f}")
                    
                    # Only show important signals
                    important_signals = [s for s in exit_signals if '⚠️' in s or '🎯' in s or '💰' in s]
                    if important_signals:
                        print("\n🎯 SIGNALS:")
                        for signal in important_signals:
                            print(f"  {signal}")
                    
                    if should_exit:
                        profit_color = '🟢' if actual_profit > 0 else '🔴'
                        print(f"\n{profit_color} TRADE CLOSED: ${actual_profit:+.2f}")
                        self.current_balance += actual_profit
                        self.peak_balance = max(self.peak_balance, self.current_balance)
                        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                        self.max_drawdown = max(self.max_drawdown, drawdown)
                        
                        # Record trade
                        trade = {
                            'entry_time': self.position['entry_time'],
                            'exit_time': current_time,
                            'direction': self.position['direction'],
                            'entry_price': self.position['entry_price'],
                            'exit_price': current_price,
                            'profit': actual_profit
                        }
                        self.trades.append(trade)
                        self.position = None
                        
                else:
                    # Analyze for new trade
                    confidence, signals, wave_trend, channel_width = self.analyze_price_action(current_price, recent_prices)
                    
                    print("\n📊 MARKET STATUS")
                    conf_color = '🟢' if confidence >= 80 else '🟡' if confidence >= 50 else '🔴'
                    print(f"{conf_color} Signal Strength: {confidence}%")
                    trend_arrow = '↗️' if wave_trend > 0 else '↙️' if wave_trend < 0 else '↔️'
                    print(f"{trend_arrow} Trend: {abs(wave_trend):.4f}")
                    
                    # Only show strong signals
                    strong_signals = [s for s in signals if any(x in s for x in ['⚡', '💫', '🎯', '💰'])]
                    if strong_signals:
                        print("\n🎯 SIGNALS:")
                        for signal in strong_signals:
                            print(f"  {signal}")
                    
                    # Enter trade if conditions are right
                    if confidence >= 80:
                        direction = 'long' if wave_trend > 0 else 'short'
                        self.position = {
                            'direction': direction,
                            'entry_price': current_price,
                            'entry_time': current_time,
                            'size': self.position_size,
                            'entry_channel_width': channel_width
                        }
                        print(f"\n{'='*40}\nENTER {direction.upper()} TRADE\n{'='*40}\n")
                
                # Print recent trades
                if self.trades:
                    print("\n📜 RECENT TRADES")
                    for trade in self.trades[-3:]:  # Show only last 3 trades
                        direction_arrow = '⬆️' if trade['direction'] == 'long' else '⬇️'
                        profit_color = '🟢' if trade['profit'] > 0 else '🔴'
                        print(f"{direction_arrow} {profit_color} ${trade['profit']:+.2f}")
                
                # Sleep based on mode
                if backtest_months > 0:
                    time.sleep(0.001)  # Minimal sleep but keep timing consistent
                else:
                    time.sleep(60)  # Normal sleep for live trading
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    print("1. Run 7-Month Backtest")
    print("2. Run Live Trading")
    choice = input("Enter your choice (1/2): ")
    
    api_key = input("Enter your OANDA API key: ")
    trader = RealtimeTrader(initial_balance=100, max_leverage=30)
    
    if choice == "1":
        print("Starting 7-Month Backtest...")
        trader.run_live_trading(api_key, backtest_months=7)
    elif choice == "2":
        print("Starting Live Trading...")
        trader.run_live_trading(api_key)
    
    # Load and prepare data
    data = pd.read_csv('/Users/ericscomputer/BackTrader/EUR_USD_M5_350days.csv')
    data = data.rename(columns={'close': 'price'})
    data['timestamp'] = pd.to_datetime(data['time'])
    
    # Initialize trader
    trader = RealtimeTrader(initial_balance=100, max_leverage=50)
    
    print(f"\nBacktesting on {len(data)} candles...")
    print("Press Ctrl+C to stop")
    
    try:
        trader.run_trading_simulation(data)
    except KeyboardInterrupt:
        print("\n\nBacktest stopped by user")
    finally:
        print(f"\nFinal Results:")
        print(f"Balance: ${trader.current_balance:,.2f} ({((trader.current_balance/trader.initial_balance)-1)*100:+.2f}%)")
        print(f"Total Trades: {len(trader.trades)}")
        if trader.trades:
            print(f"Win Rate: {trader.win_rate*100:.1f}%")
            print(f"Max Drawdown: {trader.max_drawdown*100:.1f}%)")
