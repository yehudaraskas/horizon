# This file will be populated with the ForexTrader CLI core code
# Replace this with the actual implementation from the main artifact
#!/usr/bin/env python3
"""
ForexTrader CLI - Terminal-based Forex Trading System
====================================================

A complete forex trading platform with:
- OANDA API integration for live and historical data
- Advanced technical analysis (Wavelets, LOWESS, Gaussian Channels, Stochastics)
- Interactive backtesting with terminal charting
- Comprehensive trade reporting
- Live trading functionality
"""

# Import required packages
import numpy as np
import pandas as pd
import requests
import pywt
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import argparse
import configparser
from tabulate import tabulate
from colorama import Fore, Back, Style, init as colorama_init
import os
import datetime
import sys
import time
import requests

# Initialize colorama for cross-platform color support
colorama_init()

# ======================================================
# Configuration Management
# ======================================================

class ConfigManager:
    """Manages loading and saving configuration settings"""
    
    def __init__(self, config_path="config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            self.config.read(self.config_path)
        else:
            self.create_default_config()
            
    def create_default_config(self):
        """Create a default configuration file"""
        self.config['API'] = {
            'demo_account_id': '101-001-30759619-001',
            'demo_api_key': '19a9220ae67f6f6afdc1b3386c476bb7-bce51a4f79755a66d3300305fc1b59a6',
            'demo_base_url': 'https://api-fxpractice.oanda.com/v3',
            'live_account_id': '001-001-2344427-001',
            'live_api_key': 'c1dbaa83ee1aef7b3a8adf7d6e6e1e54-c0ce1d694deff90b8860963371b8fe48',
            'live_base_url': 'https://api-fxtrade.oanda.com/v3'
        }
        
        self.config['TRADING'] = {
            'default_lot_size': '1000',
            'max_positions': '5',
            'default_leverage': '20',
            'position_sizing_type': 'risk_based',
            'max_risk_per_trade': '0.02',
            'max_daily_drawdown': '0.05',
            'max_total_drawdown': '0.15',
            'trailing_stop_pips': '50',
            'initial_stop_loss_pips': '30',
            'take_profit_pips': '90'
        }
        
        self.config['TECHNICAL_ANALYSIS'] = {
            'wavelet_type': 'db1',
            'wavelet_level': '1',
            'analysis_timeframe': 'M5',
            'min_data_points': '100',
            'gaussian_period': '20',
            'gaussian_deviations': '2.0',
            'lowess_fraction': '0.25',
            'lowess_iterations': '2',
            'default_window': '20'
        }
        
        self.config['BACKTEST'] = {
            'default_initial_balance': '10000',
            'commission_rate': '0.0001',
            'slippage': '0.0002',
            'data_granularity': 'M1'
        }
        
        self.config['LOGGING'] = {
            'log_directory': './logs',
            'trade_log': 'trades.log',
            'error_log': 'errors.log',
            'debug_log': 'debug.log'
        }
        
        self.config['USER_SETTINGS'] = {
            'selected_option_1': 'default_value_1',
            'selected_option_2': 'default_value_2',
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            self.config.write(f)
    
    def get(self, section, key, fallback=None):
        """Get a configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section, key, fallback=None):
        """Get a configuration value as integer"""
        return self.config.getint(section, key, fallback=fallback)
    
    def getfloat(self, section, key, fallback=None):
        """Get a configuration value as float"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def getboolean(self, section, key, fallback=None):
        """Get a configuration value as boolean"""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def set(self, section, key, value):
        """Set a configuration value"""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
    
    def save(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            self.config.write(f)
    
    def save_user_settings(self, option_1, option_2):
        """Save user-selected options to config"""
        self.config['USER_SETTINGS']['selected_option_1'] = option_1
        self.config['USER_SETTINGS']['selected_option_2'] = option_2
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

    def load_user_settings(self):
        """Load user-selected options from config"""
        option_1 = self.config.get('USER_SETTINGS', 'selected_option_1', fallback='default_value_1')
        option_2 = self.config.get('USER_SETTINGS', 'selected_option_2', fallback='default_value_2')
        return option_1, option_2


# ======================================================
# Terminal Chart Renderer
# ======================================================

class TerminalChartRenderer:
    """Renders charts in the terminal using ASCII/ANSI characters"""
    
    def __init__(self, width=80, height=20):
        self.width = width
        self.height = height
        # In your TerminalChartRenderer class, add position initialization in the __init__ method:
        self.position = None  # Initialize position attribute
        
        self.colors = {
            'price': Fore.WHITE,
            'wavelet': Fore.CYAN,
            'lowess': Fore.YELLOW,
            'gaussian_upper': Fore.GREEN,
            'gaussian_lower': Fore.GREEN,
            'gaussian_middle': Fore.BLUE,
            'buy_signal': Fore.GREEN + Style.BRIGHT,
            'sell_signal': Fore.RED + Style.BRIGHT,
            'background': Back.BLACK,
            'axis': Fore.WHITE + Style.DIM,
            'label': Fore.WHITE + Style.BRIGHT
        }
    
    def _scale_data(self, data, min_val, max_val):
        """Scale data to fit chart height"""
        if max_val == min_val:
            return [self.height // 2] * len(data)
        scale = (self.height - 1) / (max_val - min_val)
        return [(self.height - 1) - int((val - min_val) * scale) for val in data]
    
    def _create_empty_canvas(self):
        """Create an empty chart canvas"""
        return [[' ' for _ in range(self.width)] for _ in range(self.height)]
    
    def _draw_line(self, canvas, points, char='•', color=Fore.WHITE):
        """Draw a line on the canvas with given points"""
        for i in range(1, len(points)):
            x1, y1 = i-1, points[i-1]
            x2, y2 = i, points[i]
            
            # Skip if out of range
            if y1 < 0 or y1 >= self.height or y2 < 0 or y2 >= self.height:
                continue
                
            # Simple line drawing
            if y1 == y2:
                canvas[y1][x1] = color + char + Style.RESET_ALL
            else:
                canvas[y1][x1] = color + char + Style.RESET_ALL
                canvas[y2][x2] = color + char + Style.RESET_ALL
    
    def _draw_dashed_line(self, canvas, points, char, color):
        height = len(canvas)
        width = len(canvas[0]) if height > 0 else 0
        
        for i in range(len(points) - 1):
            x1, y1 = int(i), int(points[i])
            x2, y2 = int(i + 1), int(points[i + 1])
            
            # Clip coordinates to canvas bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Only draw if within bounds
            if 0 <= x1 < width and 0 <= y1 < height:
                canvas[y1][x1] = color + char + Style.RESET_ALL
    
    def draw_points(self, canvas, x_values, y_values, point_char='*', point_color=Fore.WHITE):
        """Draw points on the canvas"""
        for x, y in zip(x_values, y_values):
            if 0 <= y < self.height and 0 <= x < self.width:
                canvas[y][x] = point_color + point_char + Style.RESET_ALL
    
    def _draw_axis_labels(self, canvas, min_val, max_val, timestamps=None):
        """Draw axis labels"""
        # Y-axis labels (price)
        price_range = max_val - min_val
        step = price_range / 4
        
        for i in range(5):
            label_val = max_val - (i * step)
            label_y = int((i / 4) * (self.height - 1))
            label_str = f"{label_val:.5f}"
            
            # Ensure label fits
            if label_y < self.height:
                for j, char in enumerate(label_str):
                    if j < 8:  # Limit label width
                        canvas[label_y][j] = self.colors['label'] + char + Style.RESET_ALL
        
        # X-axis labels (time)
        if timestamps is not None and len(timestamps) > 0:
            time_points = [0, len(timestamps)//2, len(timestamps)-1]
            for x in time_points:
                if x < len(timestamps):
                    time_str = timestamps[x].strftime('%H:%M') if hasattr(timestamps[x], 'strftime') else str(timestamps[x])
                    if len(time_str) <= 5:
                        for j, char in enumerate(time_str):
                            idx = min(x + j, self.width - 1)
                            canvas[self.height - 1][idx] = self.colors['label'] + char + Style.RESET_ALL
    
    def render_chart(self, df: pd.DataFrame, window: int = None, title: str = "Price Chart", active_position=None):
        """
        Render price chart with indicators in terminal and stats area for active trade
        """
        # Update colors to bright neon versions
        self.colors = {
            'price': Fore.WHITE + Style.BRIGHT,
            'wavelet': Fore.CYAN + Style.BRIGHT,
            'lowess': Fore.YELLOW + Style.BRIGHT,
            'gaussian_upper': Fore.GREEN + Style.BRIGHT,
            'gaussian_lower': Fore.GREEN + Style.BRIGHT,
            'gaussian_middle': Fore.BLUE + Style.BRIGHT,
            'buy_signal': Fore.GREEN + Style.BRIGHT,
            'sell_signal': Fore.RED + Style.BRIGHT,
            'background': Back.BLACK,
            'axis': Fore.WHITE + Style.DIM,
            'label': Fore.WHITE + Style.BRIGHT,
            'profit': Fore.GREEN + Style.BRIGHT,
            'loss': Fore.RED + Style.BRIGHT
        }

        # Determine window size
        if window is None or window <= 0 or window > len(df):
            window = min(self.width, len(df))

        # Get data subset
        chart_df = df.iloc[-window:].reset_index(drop=True)

        # Prepare data
        close_prices = chart_df['close'].values
        current_price = close_prices[-1] if len(close_prices) > 0 else None

        # Calculate min/max for prices only (with 5% padding)
        min_price = min(close_prices)
        max_price = max(close_prices)
        price_range = max_price - min_price

        # Add 5% padding
        min_price = min_price - (price_range * 0.05)
        max_price = max_price + (price_range * 0.05)

        # Check if indicator columns exist
        has_wavelet = 'wavelet_denoised' in chart_df.columns and not chart_df['wavelet_denoised'].isna().all()
        has_lowess = 'lowess_trend' in chart_df.columns and not chart_df['lowess_trend'].isna().all()
        has_gaussian = all(col in chart_df.columns for col in ['gaussian_middle', 'gaussian_upper', 'gaussian_lower']) and \
                    not chart_df['gaussian_middle'].isna().all()
        has_signals = 'signal' in chart_df.columns

        # Create canvas
        canvas = self._create_empty_canvas()

        # Scale price data
        scaled_prices = self._scale_data(close_prices, min_price, max_price)

        # Draw indicators first (so price is on top)
        if has_gaussian:
            middle_data = chart_df['gaussian_middle'].fillna(method='ffill').fillna(method='bfill').values
            upper_data = chart_df['gaussian_upper'].fillna(method='ffill').fillna(method='bfill').values
            lower_data = chart_df['gaussian_lower'].fillna(method='ffill').fillna(method='bfill').values

            # Scale all Gaussian channels to the price range
            scaled_middle = self._scale_data(middle_data, min_price, max_price)
            scaled_upper = self._scale_data(upper_data, min_price, max_price)
            scaled_lower = self._scale_data(lower_data, min_price, max_price)

            self._draw_dashed_line(canvas, scaled_upper, '·', self.colors['gaussian_upper'])
            self._draw_dashed_line(canvas, scaled_lower, '·', self.colors['gaussian_lower'])
            self._draw_dashed_line(canvas, scaled_middle, '·', self.colors['gaussian_middle'])

        if has_lowess:
            lowess_data = chart_df['lowess_trend'].fillna(method='ffill').fillna(method='bfill').values
            scaled_lowess = self._scale_data(lowess_data, min_price, max_price)
            self._draw_line(canvas, scaled_lowess, '×', self.colors['lowess'])

        if has_wavelet:
            wavelet_data = chart_df['wavelet_denoised'].fillna(method='ffill').fillna(method='bfill').values
            scaled_wavelet = self._scale_data(wavelet_data, min_price, max_price)
            self._draw_line(canvas, scaled_wavelet, '◦', self.colors['wavelet'])

        # Draw price line last so it's on top
        self._draw_line(canvas, scaled_prices, '•', self.colors['price'])

        # Draw trading signals
        if has_signals:
            buy_x = []
            buy_y = []
            sell_x = []
            sell_y = []

            for i, row in chart_df.iterrows():
                if i < len(scaled_prices):
                    if row['signal'] == 1:
                        buy_x.append(i)
                        buy_y.append(scaled_prices[i])
                    elif row['signal'] == -1:
                        sell_x.append(i)
                        sell_y.append(scaled_prices[i])

            self._draw_points(canvas, buy_x, buy_y, '▲', self.colors['buy_signal'])
            self._draw_points(canvas, sell_x, sell_y, '▼', self.colors['sell_signal'])

        # Draw axis labels
        self._draw_axis_labels(canvas, min_price, max_price, chart_df['timestamp'].values)

        # Create chart string
        chart_str = "\n" + "=" * self.width + "\n"
        chart_str += f" {title} \n"
        chart_str += "=" * self.width + "\n"

        # Add chart body
        for row in canvas:
            chart_str += ''.join(row) + "\n"

        # Add position stats if there is one
        if active_position and current_price:
            chart_str += self._format_position_stats(active_position, current_price, chart_df)

        # Create legend
        legend = f"INDICATORS: "
        legend += f"{self.colors['price']}•••• PRICE{Style.RESET_ALL} | "
        if has_wavelet:
            legend += f"{self.colors['wavelet']}◦◦◦◦ WAVELET{Style.RESET_ALL} | "
        if has_lowess:
            legend += f"{self.colors['lowess']}×××× LOWESS{Style.RESET_ALL} | "
        if has_gaussian:
            legend += f"{self.colors['gaussian_middle']}···· GAUSSIAN{Style.RESET_ALL} | "
        if has_signals:
            legend += f"{self.colors['buy_signal']}▲ BUY{Style.RESET_ALL} {self.colors['sell_signal']}▼ SELL{Style.RESET_ALL}"

        chart_str += "-" * self.width + "\n"
        chart_str += legend + "\n"

        return chart_str
    

    
    def animate_chart(self, df, start_idx=0, speed=0.1):
        """Animate chart in terminal by rendering frames sequentially"""
        if start_idx < 0 or start_idx >= len(df):
            start_idx = 0
        
        # Clear terminal
        print("\033c", end="")
        
        for i in range(start_idx, len(df)):
            # Render chart up to current index
            window_size = min(self.width, i+1)
            chart_slice = df.iloc[max(0, i-window_size+1):i+1]
            
            if len(chart_slice) > 0:
                chart_str = self.render_chart(chart_slice)
                
                # Clear terminal and print chart
                print("\033c", end="")
                print(chart_str)
                
                # Print current indicator values
                row = df.iloc[i]
                print(f"Time: {row['timestamp']}")
                print(f"Price: {row['close']:.5f}")
                
                if 'wavelet_momentum' in row and not pd.isna(row['wavelet_momentum']):
                    print(f"Wavelet Momentum: {row['wavelet_momentum']*10000:+.2f}")
                
                if 'signal' in row and row['signal'] != 0:
                    signal_str = "BUY" if row['signal'] == 1 else "SELL"
                    signal_color = Fore.GREEN if row['signal'] == 1 else Fore.RED
                    print(f"Signal: {signal_color}{signal_str}{Style.RESET_ALL}")
                
                # Pause before next frame
                time.sleep(speed)


# ======================================================
# Interactive Terminal UI
# ======================================================

class TerminalUI:
    """Interactive Terminal User Interface"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.data_fetcher = None
        self.analyzer = TechnicalAnalysis(config_manager)
        self.strategy = WaveletGaussianStrategy(config_manager)
        self.backtester = Backtester(config_manager)
        self.chart_renderer = TerminalChartRenderer()
        self.current_data = None
        self.analyzed_data = None
        self.backtest_results = None
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self):
        """Display application header"""
        self.clear_screen()
        print("="*80)
        print("   FOREX TRADER CLI - Terminal-based Trading System")
        print("="*80)
        print("A complete forex trading platform with advanced technical analysis")
        print("-"*80)
    
    def wait_for_key(self):
        """Wait for user to press a key"""
        print("\nPress Enter to continue...")
        input()
    
    def display_menu(self, options):
        """Display menu with options"""
        for idx, option in enumerate(options, 1):
            print(f"{idx}. {option}")
        
        while True:
            try:
                choice = int(input("\nEnter your choice: "))
                if 1 <= choice <= len(options):
                    return choice
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    def prompt_date_range(self):
        """Prompt for date range"""
        print("\nEnter date range for historical data:")
        
        while True:
            start_date = input("Start date (YYYY-MM-DD): ")
            try:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")
        
        while True:
            end_date = input("End date (YYYY-MM-DD) or 'now' for current date: ")
            if end_date.lower() == 'now':
                end_date = datetime.datetime.now()
                break
            try:
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
                if end_date < start_date:
                    print("End date must be after start date.")
                    continue
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")
        
        return start_date, end_date
    
    def setup_oanda_connection(self, use_live=False):
        """Setup connection to OANDA API"""
        
        # Get list of available instruments
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "USD_CHF"]
        granularities = ["M1", "M5", "M15", "M30", "H1", "H4", "D"]
        
        print("\nSelect an instrument:")
        for idx, instr in enumerate(instruments, 1):
            print(f"{idx}. {instr}")
        
        while True:
            try:
                choice = int(input("\nEnter your choice: "))
                if 1 <= choice <= len(instruments):
                    instrument = instruments[choice-1]
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
        
        print("\nSelect a timeframe:")
        for idx, gran in enumerate(granularities, 1):
            print(f"{idx}. {gran}")
        
        while True:
            try:
                choice = int(input("\nEnter your choice: "))
                if 1 <= choice <= len(granularities):
                    granularity = granularities[choice-1]
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
        
        # Setup data fetcher
        self.data_fetcher = OANDADataFetcher(self.config, instrument, granularity, use_live)
        print(f"\nConnected to OANDA {'Live' if use_live else 'Demo'} API")
        print(f"Instrument: {instrument}, Timeframe: {granularity}")
        
        return instrument, granularity
    
    def fetch_historical_data(self):
        """Fetch and process historical data"""
        if self.data_fetcher is None:
            print("Please setup OANDA connection first.")
            return
        
        # Get date range
        start_date, end_date = self.prompt_date_range()
        
        print(f"\nFetching data for {self.data_fetcher.instrument} from {start_date} to {end_date}...")
        
        # Fetch data
        use_cached = input("Use cached data if available? (y/n): ").lower() == 'y'
        
        self.current_data = self.data_fetcher.get_historical_data(start_date, end_date, use_cached)
        
        if self.current_data is not None:
            print(f"Fetched {len(self.current_data)} candles.")
            
            # Display preview
            print("\nData Preview:")
            print(self.current_data.head().to_string())
            
            return True
        else:
            print("Failed to fetch data.")
            return False
    
    def analyze_data(self):
        """Analyze loaded data with technical indicators"""
        if self.current_data is None:
            print("Please load data first.")
            return
        
        # Get window size for analysis
        while True:
            try:
                window_size = input("\nEnter analysis window size (10-300, default: 20): ")
                if window_size == "":
                    window_size = 20
                else:
                    window_size = int(window_size)
                
                if 10 <= window_size <= 300:
                    break
                print("Window size must be between 10 and 300.")
            except ValueError:
                print("Please enter a valid number.")
        
        print(f"\nAnalyzing data with window size {window_size}...")
        
        # Perform analysis
        self.analyzed_data = self.analyzer.analyze_price_series(self.current_data, window_size)
        
        # Generate trading signals
        self.analyzed_data = self.strategy.generate_signals(self.analyzed_data)
        
        print("Analysis complete. Indicators added:")
        print("- Wavelet denoising and momentum")
        print("- Gaussian channels")
        print("- LOWESS trend")
        print("- Stochastic oscillator")
        print("- Trading signals")
        
        # Display chart
        print("\nGenerating chart...")
        chart_width = min(80, os.get_terminal_size().columns - 5 if hasattr(os, 'get_terminal_size') else 80)
        chart_height = min(30, os.get_terminal_size().lines - 15 if hasattr(os, 'get_terminal_size') else 20)
        
        self.chart_renderer = TerminalChartRenderer(chart_width, chart_height)
        chart = self.chart_renderer.render_chart(self.analyzed_data, min(100, len(self.analyzed_data)))
        print(chart)
        
        return True
    
    def run_backtest(self):
        """Run backtest on analyzed data"""
        if self.analyzed_data is None:
            print("Please analyze data first.")
            return
        
        # In backtesting or live trading
        if self.position:  # If there's an active position
            chart = self.chart_renderer.render_chart(
                self.analyzed_data, 
                min(100, len(self.analyzed_data)),
                active_position=self.position
            )
        else:
            chart = self.chart_renderer.render_chart(
                self.analyzed_data, 
                min(100, len(self.analyzed_data))
    )
        
        
        # Get backtest parameters
        while True:
            try:
                initial_balance = input("\nEnter initial balance ($, default: 10000): ")
                if initial_balance == "":
                    initial_balance = 10000
                else:
                    initial_balance = float(initial_balance)
                
                if initial_balance > 0:
                    break
                print("Initial balance must be positive.")
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                lot_size = input("Enter lot size (units, default: 1000): ")
                if lot_size == "":
                    lot_size = 1000
                else:
                    lot_size = float(lot_size)
                
                if lot_size > 0:
                    break
                print("Lot size must be positive.")
            except ValueError:
                print("Please enter a valid number.")
        
        use_stop_loss = input("Use stop loss? (y/n, default: y): ").lower() != 'n'
        use_take_profit = input("Use take profit? (y/n, default: y): ").lower() != 'n'
        
        print(f"\nRunning backtest with ${initial_balance:,.2f} initial balance and {lot_size:,.0f} units lot size...")
        
        # Reset backtester
        self.backtester.reset(initial_balance)
        
        # Run backtest
        self.backtest_results = self.backtester.run_backtest(
            self.analyzed_data, lot_size, use_stop_loss, use_take_profit)
        
        # Display results
        include_trades = input("Include detailed trade list in summary? (y/n, default: n): ").lower() == 'y'
        self.backtester.print_summary(include_trades)
        
        # Ask if user wants to see an animated chart
        if input("\nView animated backtest chart? (y/n): ").lower() == 'y':
            # Find a good starting point with some trades
            start_idx = 0
            if len(self.backtester.trades) > 0:
                first_trade_time = self.backtester.trades[0]['entry_time']
                for i, row in self.backtest_results.iterrows():
                    if row['timestamp'] >= first_trade_time:
                        start_idx = max(0, i - 20)  # Start a bit before the first trade
                        break
            
            # Adjust speed based on data size
            speed = 0.05
            if len(self.backtest_results) > 1000:
                speed = 0.01
            
            print("\nPress Ctrl+C to stop the animation.")
            try:
                self.chart_renderer.animate_chart(self.backtest_results, start_idx, speed)
            except KeyboardInterrupt:
                print("\nAnimation stopped.")
        
        return True
    
    def live_trading_menu(self):
        """Live trading menu and functionality"""
        if not self.data_fetcher or self.data_fetcher.use_live == False:
            print("Setting up live trading account connection...")
            self.setup_oanda_connection(use_live=True)
        
        # Get account information
        account_info = self.data_fetcher.get_account_summary()
        if account_info:
            account = account_info.get('account', {})
            balance = float(account.get('balance', 0))
            currency = account.get('currency', 'USD')
            
            print("\nAccount Summary:")
            print(f"Balance: {balance:,.2f} {currency}")
            print(f"Margin Available: {float(account.get('marginAvailable', 0)):,.2f} {currency}")
            print(f"Open Trades: {account.get('openTradeCount', 0)}")
            print(f"Pending Orders: {account.get('pendingOrderCount', 0)}")
            
            # Live trading options
            live_options = [
                "Check market prices",
                "Run live analysis",
                "Place market order",
                "View open positions",
                "Return to main menu"
            ]
            
            while True:
                self.display_header()
                print("\nLIVE TRADING MENU")
                print("-"*80)
                
                choice = self.display_menu(live_options)
                
                if choice == 1:  # Check market prices
                    print("\nFetching latest market prices...")
                    latest_prices = self.data_fetcher.get_latest_prices(100)
                    
                    if latest_prices is not None:
                        print(f"\nLatest {self.data_fetcher.instrument} price: {latest_prices.iloc[-1]['close']:.5f}")
                        current_price = latest_prices.iloc[-1]['close']
                        prev_price = latest_prices.iloc[-2]['close'] if len(latest_prices) > 1 else current_price
                        
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100
                        
                        direction = "up" if change >= 0 else "down"
                        color = Fore.GREEN if change >= 0 else Fore.RED
                        
                        print(f"Change: {color}{change:+.5f} ({change_pct:+.3f}%){Style.RESET_ALL}")
                        print(f"Bid/Ask Spread: ~{0.0001:.5f} (~{0.0001/current_price*100:.3f}%)")
                        
                        # Show chart
                        chart = self.chart_renderer.render_chart(latest_prices)
                        print(chart)
                    else:
                        print("Failed to fetch latest prices.")
                
                elif choice == 2:  # Run live analysis
                    print("\nFetching data for analysis...")
                    latest_data = self.data_fetcher.get_latest_prices(250)
                    
                    if latest_data is not None:
                        print(f"Analyzing {len(latest_data)} candles...")
                        
                        # Get window size
                        window_size = 20
                        try:
                            user_window = input("Enter analysis window size (10-100, default: 20): ")
                            if user_window:
                                window_size = int(user_window)
                                window_size = max(10, min(100, window_size))
                        except ValueError:
                            pass
                        
                        # Analyze data
                        analyzed_data = self.analyzer.analyze_price_series(latest_data, window_size)
                        analyzed_data = self.strategy.generate_signals(analyzed_data)
                        
                        # Display latest signals and indicators
                        latest_row = analyzed_data.iloc[-1]
                        print("\nLatest Analysis Results:")
                        print(f"Price: {latest_row['close']:.5f}")
                        
                        if not pd.isna(latest_row['wavelet_momentum']):
                            momentum = latest_row['wavelet_momentum'] * 10000
                            momentum_str = f"{momentum:+.2f}"
                            momentum_color = Fore.GREEN if momentum > 0 else Fore.RED if momentum < 0 else Fore.WHITE
                            print(f"Wavelet Momentum: {momentum_color}{momentum_str}{Style.RESET_ALL}")
                        
                        if 'confidence' in latest_row and not pd.isna(latest_row['confidence']):
                            confidence = latest_row['confidence']
                            conf_color = Fore.GREEN if confidence > 75 else Fore.YELLOW if confidence > 50 else Fore.RED
                            print(f"Signal Confidence: {conf_color}{confidence:.1f}%{Style.RESET_ALL}")
                        
                        if latest_row['signal'] != 0:
                            signal_str = "BUY" if latest_row['signal'] == 1 else "SELL"
                            signal_color = Fore.GREEN if latest_row['signal'] == 1 else Fore.RED
                            print(f"Trading Signal: {signal_color}{signal_str}{Style.RESET_ALL}")
                        else:
                            print("Trading Signal: NONE")
                        
                        # Show chart
                        chart = self.chart_renderer.render_chart(analyzed_data, min(80, len(analyzed_data)))
                        print(chart)
                        
                        # Ask if want to place a trade based on signal
                        if latest_row['signal'] != 0 and input("\nPlace trade based on this signal? (y/n): ").lower() == 'y':
                            direction = "BUY" if latest_row['signal'] == 1 else "SELL"
                            
                            # Get trade details
                            while True:
                                try:
                                    lot_size = input("Enter lot size (units, default: 1000): ")
                                    if lot_size == "":
                                        lot_size = 1000
                                    else:
                                        lot_size = float(lot_size)
                                    
                                    if lot_size > 0:
                                        break
                                    print("Lot size must be positive.")
                                except ValueError:
                                    print("Please enter a valid number.")
                            
                            use_sl_tp = input("Use stop loss and take profit? (y/n, default: y): ").lower() != 'n'
                            
                            if use_sl_tp:
                                stop_loss_pips = self.config.getfloat('TRADING', 'initial_stop_loss_pips', fallback=30)
                                take_profit_pips = self.config.getfloat('TRADING', 'take_profit_pips', fallback=90)
                            else:
                                stop_loss_pips = None
                                take_profit_pips = None
                            
                            # Confirm trade
                            print(f"\nTrade Summary:")
                            print(f"Direction: {direction}")
                            print(f"Instrument: {self.data_fetcher.instrument}")
                            print(f"Lot Size: {lot_size} units")
                            
                            if use_sl_tp:
                                print(f"Stop Loss: {stop_loss_pips} pips")
                                print(f"Take Profit: {take_profit_pips} pips")
                            
                            if input("\nConfirm trade execution? (y/n): ").lower() == 'y':
                                # Execute trade
                                result = self.data_fetcher.execute_trade(
                                    direction, lot_size, stop_loss_pips, take_profit_pips)
                                
                                if result:
                                    print(f"\n{Fore.GREEN}Trade executed successfully!{Style.RESET_ALL}")
                                    print(f"Order ID: {result.get('orderCreateTransaction', {}).get('id', 'N/A')}")
                                else:
                                    print(f"\n{Fore.RED}Failed to execute trade.{Style.RESET_ALL}")
                    else:
                        print("Failed to fetch data for analysis.")
                
                elif choice == 3:  # Place market order
                    print("\nPlace Market Order:")
                    
                    # Get latest price
                    current_price = self.data_fetcher.get_latest_price()
                    
                    if current_price:
                        print(f"Current {self.data_fetcher.instrument} price: {current_price:.5f}")
                        
                        # Get trade details
                        direction = None
                        while direction not in ["BUY", "SELL"]:
                            direction = input("Enter direction (BUY/SELL): ").upper()
                        
                        while True:
                            try:
                                lot_size = input("Enter lot size (units, default: 1000): ")
                                if lot_size == "":
                                    lot_size = 1000
                                else:
                                    lot_size = float(lot_size)
                                
                                if lot_size > 0:
                                    break
                                print("Lot size must be positive.")
                            except ValueError:
                                print("Please enter a valid number.")
                        
                        use_sl_tp = input("Use stop loss and take profit? (y/n, default: y): ").lower() != 'n'
                        
                        if use_sl_tp:
                            while True:
                                try:
                                    stop_loss_pips = input("Enter stop loss in pips (default: 30): ")
                                    if stop_loss_pips == "":
                                        stop_loss_pips = 30
                                    else:
                                        stop_loss_pips = float(stop_loss_pips)
                                    
                                    if stop_loss_pips > 0:
                                        break
                                    print("Stop loss must be positive.")
                                except ValueError:
                                    print("Please enter a valid number.")
                            
                            while True:
                                try:
                                    take_profit_pips = input("Enter take profit in pips (default: 90): ")
                                    if take_profit_pips == "":
                                        take_profit_pips = 90
                                    else:
                                        take_profit_pips = float(take_profit_pips)
                                    
                                    if take_profit_pips > 0:
                                        break
                                    print("Take profit must be positive.")
                                except ValueError:
                                    print("Please enter a valid number.")
                        else:
                            stop_loss_pips = None
                            take_profit_pips = None
                        
                        # Confirm trade
                        print(f"\nTrade Summary:")
                        print(f"Direction: {direction}")
                        print(f"Instrument: {self.data_fetcher.instrument}")
                        print(f"Current Price: {current_price:.5f}")
                        print(f"Lot Size: {lot_size} units")
                        
                        if use_sl_tp:
                            if direction == "BUY":
                                stop_loss_price = current_price - (stop_loss_pips * 0.0001)
                                take_profit_price = current_price + (take_profit_pips * 0.0001)
                            else:
                                stop_loss_price = current_price + (stop_loss_pips * 0.0001)
                                take_profit_price = current_price - (take_profit_pips * 0.0001)
                            
                            print(f"Stop Loss: {stop_loss_price:.5f} ({stop_loss_pips} pips)")
                            print(f"Take Profit: {take_profit_price:.5f} ({take_profit_pips} pips)")
                        
                        if input("\nConfirm trade execution? (y/n): ").lower() == 'y':
                            # Execute trade
                            result = self.data_fetcher.execute_trade(
                                direction, lot_size, stop_loss_pips, take_profit_pips)
                            
                            if result:
                                print(f"\n{Fore.GREEN}Trade executed successfully!{Style.RESET_ALL}")
                                print(f"Order ID: {result.get('orderCreateTransaction', {}).get('id', 'N/A')}")
                            else:
                                print(f"\n{Fore.RED}Failed to execute trade.{Style.RESET_ALL}")
                    else:
                        print("Failed to fetch current price.")
                
                elif choice == 4:  # View open positions
                    print("\nFetching open positions...")
                    
                    # This would need to be implemented in OANDADataFetcher
                    # For now, just display account summary again
                    account_info = self.data_fetcher.get_account_summary()
                    if account_info:
                        account = account_info.get('account', {})
                        balance = float(account.get('balance', 0))
                        currency = account.get('currency', 'USD')
                        
                        print("\nAccount Summary:")
                        print(f"Balance: {balance:,.2f} {currency}")
                        print(f"Margin Available: {float(account.get('marginAvailable', 0)):,.2f} {currency}")
                        print(f"Open Trades: {account.get('openTradeCount', 0)}")
                        print(f"Pending Orders: {account.get('pendingOrderCount', 0)}")
                    else:
                        print("Failed to fetch account information.")
                
                elif choice == 5:  # Return to main menu
                    break
                
                self.wait_for_key()
        else:
            print("Failed to fetch account information.")
        
        return True
    
    def export_results(self):
        """Export backtest results to file"""
        if self.backtest_results is None:
            print("No backtest results to export.")
            return
        
        filename = input("\nEnter filename for export (default: backtest_results.csv): ")
        if not filename:
            filename = "backtest_results.csv"
        
        # Create exports directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        filepath = os.path.join("exports", filename)
        
        # Export results
        self.backtest_results.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
        
        # Export trade details if available
        if self.backtester.trade_details:
            trades_filename = filename.replace(".csv", "_trades.csv")
            trades_filepath = os.path.join("exports", trades_filename)
            
            # Convert trade details to DataFrame
            trades_df = pd.DataFrame(self.backtester.trade_details)
            trades_df.to_csv(trades_filepath, index=False)
            print(f"Trade details exported to {trades_filepath}")
        
        # Export summary statistics
        stats = self.backtester.get_summary_statistics()
        stats_filename = filename.replace(".csv", "_summary.txt")
        stats_filepath = os.path.join("exports", stats_filename)
        
        with open(stats_filepath, 'w') as f:
            f.write("BACKTEST SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Initial Balance: ${self.backtester.initial_balance:,.2f}\n")
            f.write(f"Final Balance: ${(self.backtester.initial_balance + stats['net_profit']):,.2f}\n")
            f.write(f"Net Profit: ${stats['net_profit']:,.2f} ({stats['return_pct']:.2f}%)\n")
            f.write(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%\n")
            f.write(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n\n")
            
            f.write("Trade Statistics:\n")
            f.write(f"Total Trades: {stats['total_trades']}\n")
            f.write(f"Win Rate: {stats['win_rate']*100:.2f}%\n")
            f.write(f"Profit Factor: {stats['profit_factor']:.2f}\n")
            f.write(f"Average Trade: ${stats['avg_trade']:,.2f}\n")
            f.write(f"Average Winner: ${stats['avg_winning_trade']:,.2f}\n")
            f.write(f"Average Loser: ${stats['avg_losing_trade']:,.2f}\n")
            f.write(f"Largest Winner: ${stats['largest_winner']:,.2f}\n")
            f.write(f"Largest Loser: ${stats['largest_loser']:,.2f}\n")
        
        print(f"Summary statistics exported to {stats_filepath}")
        return True
    
    def run(self):
        """Main application loop"""
        while True:
            self.display_header()
            
            main_options = [
                "Setup OANDA Connection",
                "Fetch Historical Data",
                "Analyze Data with Technical Indicators",
                "Run Backtest",
                "Live Trading",
                "Export Results",
                "Settings",
                "Exit Application"
            ]
            
            choice = self.display_menu(main_options)
            
            if choice == 1:  # Setup OANDA Connection
                self.setup_oanda_connection()
            elif choice == 2:  # Fetch Historical Data
                self.fetch_historical_data()
            elif choice == 3:  # Analyze Data
                self.analyze_data()
            elif choice == 4:  # Run Backtest
                self.run_backtest()
            elif choice == 5:  # Live Trading
                self.live_trading_menu()
            elif choice == 6:  # Export Results
                self.export_results()
            elif choice == 7:  # Settings
                # TODO: Implement settings menu
                print("\nSettings functionality not implemented yet.")
                self.wait_for_key()
            elif choice == 8:  # Exit
                self.clear_screen()
                print("Thank you for using Forex Trader CLI!")
                print("Exiting application...")
                break
            
            # Wait for key press before returning to main menu
            if choice != 8:
                self.wait_for_key()


# ======================================================
# Main Function
# ======================================================

def main():
    """Main entry point for application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Forex Trader CLI - Terminal-based Trading System")
    parser.add_argument("--config", help="Path to configuration file", default="config.ini")
    parser.add_argument("--live", action="store_true", help="Use live trading mode")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on startup")
    parser.add_argument("--file", help="CSV file to use for backtest")
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Initialize UI
    ui = TerminalUI(config_manager)
    
    # Start application
    ui.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

class OANDADataFetcher:
    """Fetches forex data from OANDA API"""
    
    def __init__(self, config_manager, instrument="EUR_USD", granularity="M5", use_live=False):
        self.config = config_manager
        self.instrument = instrument
        self.granularity = granularity
        self.use_live = use_live
        
        # Setup API based on environment
        if use_live:
            self.account_id = self.config.get('API', 'live_account_id')
            self.api_key = self.config.get('API', 'live_api_key')
            self.base_url = self.config.get('API', 'live_base_url')
        else:
            self.account_id = self.config.get('API', 'demo_account_id')
            self.api_key = self.config.get('API', 'demo_api_key')
            self.base_url = self.config.get('API', 'demo_base_url')
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def generate_date_range(self, start_date, end_date):
        """Convert date strings to datetime objects"""
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        return start_date, end_date
    
    def fetch_candles(self, start_date, end_date, chunk_size=5000, backtest=False):
        """Fetch candles in chunks to handle API limits"""
        start_date, end_date = self.generate_date_range(start_date, end_date)
        all_candles = []
        current_start = start_date
        
        if backtest:
            # Generate synthetic data (for testing only)
            base_price = 1.0800
            volatility = 0.0002
            trend = 0.0001
            
            current_time = start_date
            while current_time < end_date:
                random_walk = random.gauss(0, volatility)
                trend_component = trend * (random.random() - 0.4)
                price_change = random_walk + trend_component
                
                base_price *= (1 + price_change)
                
                spread = 0.0002 * random.random()
                high = base_price * (1 + random.random() * 0.0005)
                low = base_price * (1 - random.random() * 0.0005)
                
                candle = {
                    'time': current_time.strftime('%Y-%m-%dT%H:%M:%S.000000Z'),
                    'volume': int(random.uniform(50, 200)),
                    'mid': {
                        'o': f"{base_price:.5f}",
                        'h': f"{high:.5f}",
                        'l': f"{low:.5f}",
                        'c': f"{base_price:.5f}"
                    }
                }
                
                all_candles.append(candle)
                # Increment based on granularity
                if self.granularity == 'M1':
                    current_time += datetime.timedelta(minutes=1)
                elif self.granularity == 'M5':
                    current_time += datetime.timedelta(minutes=5)
                elif self.granularity == 'H1':
                    current_time += datetime.timedelta(hours=1)
                elif self.granularity == 'D':
                    current_time += datetime.timedelta(days=1)
                else:
                    current_time += datetime.timedelta(minutes=5)  # Default
            
            return all_candles
        
        else:
            # Real OANDA API calls
            sys.stdout.write(f"Fetching data from {start_date} to {end_date}...\n")
            progress_step = max(1, int((end_date - start_date).total_seconds() / (60 * 20)))
            progress = 0
            
            while current_start < end_date:
                chunk_end = min(current_start + datetime.timedelta(minutes=chunk_size), end_date)
                start_str = current_start.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
                end_str = chunk_end.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
                
                url = f'{self.base_url}/instruments/{self.instrument}/candles'
                params = {
                    'price': 'M',
                    'granularity': self.granularity,
                    'from': start_str,
                    'to': end_str
                }
                
                try:
                    response = requests.get(url, headers=self.headers, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        all_candles.extend(data['candles'])
                        
                        # Update progress
                        progress += 1
                        if progress % progress_step == 0:
                            percent_complete = min(100, int((current_start - start_date).total_seconds() / (end_date - start_date).total_seconds() * 100))
                            sys.stdout.write(f"\rProgress: {percent_complete}% - Fetched {len(all_candles)} candles...")
                            sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\nError {response.status_code}: {response.text}\n")
                        return None
                    
                    time.sleep(0.5)  # Respect API rate limits
                    
                except Exception as e:
                    sys.stdout.write(f"\nException occurred: {e}\n")
                    return None
                
                current_start = chunk_end
            
            sys.stdout.write(f"\rProgress: 100% - Fetched {len(all_candles)} candles\n")
            return all_candles
    
    def process_candles(self, candles):
        """Convert candles to pandas DataFrame"""
        if not candles:
            return None
            
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle['time'],
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': int(candle['volume'])
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def save_data(self, df, filename=None):
        """Save data to CSV"""
        if df is not None:
            if filename is None:
                # Create a default filename
                start_date = df['timestamp'].min().strftime('%Y%m%d')
                end_date = df['timestamp'].max().strftime('%Y%m%d')
                filename = f"{self.instrument}_{self.granularity}_{start_date}_{end_date}.csv"
            
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            full_path = os.path.join("data", filename)
            
            df.to_csv(full_path, index=False)
            print(f"Data saved to {full_path}")
            print(f"Total rows: {len(df)}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return full_path
        return None

    def get_historical_data(self, start_date, end_date, use_cached=True):
        """Get historical data with optional caching"""
        # Generate cache filename
        start_str = start_date.strftime('%Y%m%d') if isinstance(start_date, datetime.datetime) else start_date.replace('-', '')
        end_str = end_date.strftime('%Y%m%d') if isinstance(end_date, datetime.datetime) else end_date.replace('-', '')
        cache_filename = f"data/{self.instrument}_{self.granularity}_{start_str}_{end_str}.csv"
        
        # Check if cache exists and is requested
        if use_cached and os.path.exists(cache_filename):
            print(f"Loading cached data from {cache_filename}")
            return pd.read_csv(cache_filename, parse_dates=['timestamp'])
        
        # Fetch new data
        candles = self.fetch_candles(start_date, end_date)
        if candles:
            df = self.process_candles(candles)
            self.save_data(df, os.path.basename(cache_filename))
            return df
        
        return None

    def get_latest_prices(self, count=100):
        """Get the latest prices for an instrument"""
        url = f'{self.base_url}/instruments/{self.instrument}/candles'
        params = {
            'price': 'M',
            'granularity': self.granularity,
            'count': count
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return self.process_candles(data['candles'])
            else:
                print(f"Error fetching latest prices: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching latest prices: {e}")
            return None

    def execute_trade(self, direction, units, stop_loss_pips=None, take_profit_pips=None):
        """Execute a trade on the OANDA platform"""
        if not self.use_live:
            print("Warning: Trading on demo account")
        
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        
        # Calculate stop loss and take profit prices if provided
        current_price = self.get_latest_price(self.instrument)
        if current_price is None:
            print("Error: Could not get current price for trade execution")
            return None
        
        # Calculate stop loss and take profit levels
        if stop_loss_pips is not None:
            if direction == "BUY":
                stop_loss_price = current_price - (stop_loss_pips * 0.0001)
            else:
                stop_loss_price = current_price + (stop_loss_pips * 0.0001)
        else:
            stop_loss_price = None
            
        if take_profit_pips is not None:
            if direction == "BUY":
                take_profit_price = current_price + (take_profit_pips * 0.0001)
            else:
                take_profit_price = current_price - (take_profit_pips * 0.0001)
        else:
            take_profit_price = None
        
        # Prepare the order data
        order_data = {
            "order": {
                "instrument": self.instrument,
                "units": units if direction == "BUY" else -units,
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        
        # Add stop loss if specified
        if stop_loss_price is not None:
            order_data["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss_price:.5f}"
            }
            
        # Add take profit if specified
        if take_profit_price is not None:
            order_data["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit_price:.5f}"
            }
        
        try:
            response = requests.post(url, headers=self.headers, json=order_data)
            if response.status_code == 201:
                result = response.json()
                print(f"Trade executed: {direction} {units} {self.instrument}")
                return result
            else:
                print(f"Error executing trade: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Exception executing trade: {e}")
            return None
    
    def get_latest_price(self, instrument=None):
        """Get the latest price for an instrument"""
        if instrument is None:
            instrument = self.instrument
            
        url = f'{self.base_url}/instruments/{instrument}/candles'
        params = {
            'price': 'M',
            'granularity': 'S5',  # Use 5-second candles for latest price
            'count': 1
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['candles'] and len(data['candles']) > 0:
                    return float(data['candles'][0]['mid']['c'])
                else:
                    return None
            else:
                print(f"Error fetching latest price: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching latest price: {e}")
            return None
            
    def get_account_summary(self):
        """Get account summary from OANDA"""
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting account summary: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception getting account summary: {e}")
            return None

# Imports at the top of the file

# other imports...

# Other classes (like ConfigManager, Logger, etc.)



# Insert the TechnicalAnalysis class here
class TechnicalAnalysis:
    """Technical analysis engine with multiple indicators based on realtime_trader_v2.py"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.wavelet_type = self.config.get('TECHNICAL_ANALYSIS', 'wavelet_type', fallback='db4')
        self.wavelet_level = self.config.getint('TECHNICAL_ANALYSIS', 'wavelet_level', fallback=3)
        self.gaussian_period = self.config.getint('TECHNICAL_ANALYSIS', 'gaussian_period', fallback=20)
        self.gaussian_deviations = self.config.getfloat('TECHNICAL_ANALYSIS', 'gaussian_deviations', fallback=2.0)
        self.lowess_fraction = self.config.getfloat('TECHNICAL_ANALYSIS', 'lowess_fraction', fallback=0.25)
        self.lowess_iterations = self.config.getint('TECHNICAL_ANALYSIS', 'lowess_iterations', fallback=2)
    
    def analyze_with_wavelets(self, prices, wavelet='db4', level=2):
        """Multi-level wavelet analysis and denoising with apply"""
        # Convert to numpy array
        prices = np.array(prices)
        
        # Decompose the signal
        coeffs = pywt.wavedec(prices, wavelet, level=level)
        
        # Use apply to calculate trends for each coefficient level
        def calculate_trend(coef):
            return np.mean(np.diff(coef)) if len(coef) > 1 else 0
        
        trends = pd.Series(coeffs).apply(calculate_trend).tolist()
        
        # Weight the trends (higher weights for lower frequencies)
        weights = [0.6, 0.4]  # Adjusted weights for 2 levels
        weighted_trend = sum(t * w for t, w in zip(trends, weights[:len(trends)]))
        
        # Denoise the signal using apply
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(prices)))
        
        def apply_threshold(i, coef):
            return coef if i == 0 else pywt.threshold(coef, threshold, mode='soft')
            
        coeffs_clean = [apply_threshold(i, c) for i, c in enumerate(coeffs)]
        denoised = pywt.waverec(coeffs_clean, wavelet)
        
        return denoised, weighted_trend
    
    def calculate_gaussian_channels(self, prices):
        original_length = len(prices)
        window_prices = np.array(prices)
        
        # Perform wavelet decomposition
        coeffs = pywt.dwt(window_prices, 'db4')  # Example wavelet
        approx, detail = coeffs
        
        # Now we need to upsample back to original size
        # Option 1: Simple interpolation
        upsampled_approx = np.interp(
            np.linspace(0, 1, original_length),
            np.linspace(0, 1, len(approx)),
            approx
        )
        
        # Option 2: Wavelet reconstruction (more accurate for wavelets)
        # This recreates a full-size array from the coefficients
        reconstructed = pywt.idwt(approx, detail, 'db4')
        
        # Make sure reconstructed array is exactly the original length
        # (wavelet reconstruction sometimes returns slightly different lengths)
        if len(reconstructed) != original_length:
            reconstructed = np.interp(
                np.linspace(0, 1, original_length),
                np.linspace(0, 1, len(reconstructed)),
                reconstructed
            )
        
        # Now generate weights for the full-size array
        weights = self.calculate_gaussian_weights(original_length)
        
        # Both arrays should now be the same size
        ma = np.sum(reconstructed * weights)
        
        # Calculate bands using the normalized data
        std_dev = np.std(reconstructed)
        upper = ma + self.gaussian_deviations * std_dev
        lower = ma - self.gaussian_deviations * std_dev
        channel_width = upper - lower
        
        return ma, upper, lower, channel_width
    
    def calculate_gaussian_weights(self, length):
        """Calculate Gaussian weights for a given length"""
        x = np.linspace(-1, 1, length)
        weights = np.exp(-0.5 * (x / 0.4) ** 2)
        weights /= weights.sum()
        return weights
    
    def analyze_price_action(self, current_price, recent_prices):
        """Analyze price action to determine trade confidence, exactly as in realtime_trader_v2.py"""
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
        confidence = min(100, max(0, confidence))
        
        return confidence, signals, wave_trend, channel_width
    
    def analyze_price_series(self, df, window=None):
        """
        Comprehensive analysis of price series, returning all indicators
        
        Args:
            df (DataFrame): Price dataframe with OHLCV columns
            window (int): Analysis window size
            
        Returns:
            DataFrame: Original dataframe with added indicator columns
        """
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Default window if not specified
        if window is None:
            window = self.config.getint('TECHNICAL_ANALYSIS', 'default_window', fallback=20)
            window = min(window, len(df))
        
        # Get price series
        close_prices = df['close'].values
        
        # Process each candle with the exact methods from realtime_trader_v2.py
        wavelet_denoised = []
        wavelet_momentum = []
        gaussian_ma = []
        gaussian_upper = []
        gaussian_lower = []
        confidence_values = []
        signal_list = []
        channel_width_values = []
        
        # Initialize with NaN values
        for _ in range(window):
            wavelet_denoised.append(np.nan)
            wavelet_momentum.append(np.nan)
            gaussian_ma.append(np.nan)
            gaussian_upper.append(np.nan)
            gaussian_lower.append(np.nan)
            confidence_values.append(np.nan)
            signal_list.append([])
            channel_width_values.append(np.nan)
        
        # Process with sliding window
        for i in range(window, len(close_prices)):
            recent_prices = close_prices[i-window:i]
            current_price = close_prices[i-1]  # Use previous price to avoid lookahead bias
            
            # Wavelet analysis
            denoised, wave_trend = self.analyze_with_wavelets(recent_prices)
            wavelet_denoised.append(denoised[-1])
            wavelet_momentum.append(wave_trend)
            
            # Gaussian channels
            ma, upper, lower, channel_width = self.calculate_gaussian_channels(recent_prices)
            gaussian_ma.append(ma)
            gaussian_upper.append(upper)
            gaussian_lower.append(lower)
            channel_width_values.append(channel_width)
            
            # Price action analysis
            confidence, signals, _, _ = self.analyze_price_action(current_price, recent_prices)
            confidence_values.append(confidence)
            signal_list.append(signals)
        
        # Add indicators to dataframe
        result_df['wavelet_denoised'] = wavelet_denoised
        result_df['wavelet_momentum'] = wavelet_momentum
        result_df['gaussian_middle'] = gaussian_ma
        result_df['gaussian_upper'] = gaussian_upper
        result_df['gaussian_lower'] = gaussian_lower
        result_df['confidence'] = confidence_values
        result_df['channel_width'] = channel_width_values
        
        # Calculate LOWESS trend (not in original, but useful)
        result_df['lowess_trend'] = self.calculate_lowess_trend(close_prices, window)
        
        # Calculate Stochastic Oscillator (not in original, but useful)
        k, d = self.calculate_stochastic(df['high'].values, df['low'].values, close_prices)
        result_df['stoch_k'] = k
        result_df['stoch_d'] = d
        
        return result_df
    
    def calculate_lowess_trend(self, prices, window=None):
        """Calculate LOWESS trend as additional indicator"""
        if window is None:
            window = min(len(prices), 100)
        
        # Adjust frac based on window size
        frac = max(min(window / len(prices), 0.9), 0.1)
        
        # Create x values (index)
        x = np.arange(len(prices))
        
        # Apply LOWESS smoothing
        smoothed = lowess(
            prices, 
            x, 
            frac=frac,
            it=self.lowess_iterations,
            return_sorted=False
        )
        
        return smoothed
    
    def calculate_stochastic(self, high_prices, low_prices, close_prices, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator as additional indicator"""
        # Calculate %K
        k_values = []
        for i in range(len(close_prices)):
            if i < k_period - 1:
                k_values.append(50)  # Default value for initial periods
                continue
                
            window_high = max(high_prices[i-(k_period-1):i+1])
            window_low = min(low_prices[i-(k_period-1):i+1])
            
            if window_high == window_low:
                k_values.append(50)  # Avoid division by zero
            else:
                # Calculate %K: (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
                k_value = ((close_prices[i] - window_low) / (window_high - window_low)) * 100
                k_values.append(k_value)
        
        # Calculate %D (simple moving average of %K)
        d_values = []
        for i in range(len(k_values)):
            if i < d_period - 1:
                d_values.append(50)  # Default value for initial periods
                continue
                
            d_value = sum(k_values[i-(d_period-1):i+1]) / d_period
            d_values.append(d_value)
        
        return np.array(k_values), np.array(d_values)



# Rest of the file...



# ======================================================
# Trading Strategy and Backtester
# ======================================================

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.initial_balance = self.config.getfloat('BACKTEST', 'default_initial_balance', fallback=10000)
        self.commission_rate = self.config.getfloat('BACKTEST', 'commission_rate', fallback=0.0001)
        self.slippage = self.config.getfloat('BACKTEST', 'slippage', fallback=0.0002)
        self.position = None
        self.trades = []
        self.trade_details = []
        self.balance_history = []
    
    def reset(self, initial_balance=None):
        """Reset backtester state"""
        if initial_balance is not None:
            self.initial_balance = initial_balance
        self.position = None
        self.trades = []
        self.trade_details = []
        self.balance_history = []
    
    def run_backtest(self, signals_df, lot_size=None, use_stop_loss=True, use_take_profit=True):
        """
        Run backtest on signals dataframe
        
        Args:
            signals_df (DataFrame): Dataframe with price data and signals
            lot_size (float): Size of each position in units
            use_stop_loss (bool): Whether to use stop loss
            use_take_profit (bool): Whether to use take profit
            
        Returns:
            DataFrame: Dataframe with backtest results
        """
        # Reset state
        self.reset()
        
        # Set default lot size if not specified
        if lot_size is None:
            lot_size = self.config.getfloat('TRADING', 'default_lot_size', fallback=1000)
        
        # Get stop loss and take profit in pips
        stop_loss_pips = self.config.getfloat('TRADING', 'initial_stop_loss_pips', fallback=30)
        take_profit_pips = self.config.getfloat('TRADING', 'take_profit_pips', fallback=90)
        
        # For tracking equity
        current_balance = self.initial_balance
        self.balance_history.append((signals_df.iloc[0]['timestamp'], current_balance))
        
        # Create result dataframe
        results_df = signals_df.copy()
        results_df['position'] = 0  # 0: no position, 1: long, -1: short
        results_df['equity'] = self.initial_balance
        
        # Iterate through each row (excluding the first)
        for i in range(1, len(results_df)):
            # Skip first few rows until indicators are available
            if pd.isna(results_df.iloc[i]['wavelet_denoised']):
                results_df.iloc[i, results_df.columns.get_loc('equity')] = current_balance
                continue
            
            current_timestamp = results_df.iloc[i]['timestamp']
            current_price = results_df.iloc[i]['close']
            signal = results_df.iloc[i]['signal']
            
            # Update current position
            if self.position is not None:
                results_df.iloc[i, results_df.columns.get_loc('position')] = 1 if self.position['direction'] == 'long' else -1
            
            # Check for stop loss or take profit
            if self.position is not None:
                entry_price = self.position['entry_price']
                direction = self.position['direction']
                
                # Calculate current profit/loss
                price_diff = current_price - entry_price
                if direction == 'short':
                    price_diff = -price_diff
                
                pip_value = 0.0001  # For most forex pairs
                pips_gained = price_diff / pip_value
                
                # Check stop loss
                if use_stop_loss and pips_gained <= -stop_loss_pips:
                    # Stop loss hit
                    profit_loss = self._calculate_profit_loss(entry_price, current_price, 
                                                             direction, lot_size)
                    current_balance += profit_loss
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': self.position['entry_time'],
                        'exit_time': current_timestamp,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_loss': profit_loss,
                        'exit_reason': 'stop_loss'
                    })
                    
                    # Add detailed trade info
                    self.trade_details.append({
                        **self.trades[-1],
                        'lot_size': lot_size,
                        'pips': pips_gained,
                        'balance_after': current_balance
                    })
                    
                    self.position = None
                    self.balance_history.append((current_timestamp, current_balance))
                
                # Check take profit
                elif use_take_profit and pips_gained >= take_profit_pips:
                    # Take profit hit
                    profit_loss = self._calculate_profit_loss(entry_price, current_price, 
                                                             direction, lot_size)
                    current_balance += profit_loss
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': self.position['entry_time'],
                        'exit_time': current_timestamp,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_loss': profit_loss,
                        'exit_reason': 'take_profit'
                    })
                    
                    # Add detailed trade info
                    self.trade_details.append({
                        **self.trades[-1],
                        'lot_size': lot_size,
                        'pips': pips_gained,
                        'balance_after': current_balance
                    })
                    
                    self.position = None
                    self.balance_history.append((current_timestamp, current_balance))
            
            # Handle signals
            if signal != 0:
                # Close existing position if it's in the opposite direction
                if self.position is not None:
                    if (signal == 1 and self.position['direction'] == 'short') or \
                       (signal == -1 and self.position['direction'] == 'long'):
                        
                        # Close position
                        entry_price = self.position['entry_price']
                        direction = self.position['direction']
                        
                        profit_loss = self._calculate_profit_loss(entry_price, current_price, 
                                                                direction, lot_size)
                        current_balance += profit_loss
                        
                        # Record trade
                        self.trades.append({
                            'entry_time': self.position['entry_time'],
                            'exit_time': current_timestamp,
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit_loss': profit_loss,
                            'exit_reason': 'signal_reversal'
                        })
                        
                        # Add detailed trade info
                        pip_value = 0.0001  # For most forex pairs
                        price_diff = current_price - entry_price
                        if direction == 'short':
                            price_diff = -price_diff
                        pips_gained = price_diff / pip_value
                        
                        self.trade_details.append({
                            **self.trades[-1],
                            'lot_size': lot_size,
                            'pips': pips_gained,
                            'balance_after': current_balance
                        })
                        
                        self.position = None
                        self.balance_history.append((current_timestamp, current_balance))
                
                # Open new position if we don't have one
                if self.position is None:
                    direction = 'long' if signal == 1 else 'short'
                    
                    # Apply slippage
                    adjusted_price = current_price
                    if direction == 'long':
                        adjusted_price *= (1 + self.slippage)
                    else:
                        adjusted_price *= (1 - self.slippage)
                    
                    self.position = {
                        'direction': direction,
                        'entry_price': adjusted_price,
                        'entry_time': current_timestamp,
                        'lot_size': lot_size
                    }
                    
                    # Update position in results
                    results_df.iloc[i, results_df.columns.get_loc('position')] = 1 if direction == 'long' else -1
            
            # Update equity for this row
            results_df.iloc[i, results_df.columns.get_loc('equity')] = current_balance
            
            # Calculate unrealized P&L if position is open
            if self.position is not None:
                entry_price = self.position['entry_price']
                direction = self.position['direction']
                
                unrealized_pl = self._calculate_profit_loss(entry_price, current_price, 
                                                          direction, lot_size, include_fees=False)
                results_df.iloc[i, results_df.columns.get_loc('equity')] = current_balance + unrealized_pl
        
        # Close any open position at the end
        if self.position is not None:
            last_price = results_df.iloc[-1]['close']
            entry_price = self.position['entry_price']
            direction = self.position['direction']
            
            profit_loss = self._calculate_profit_loss(entry_price, last_price, 
                                                    direction, lot_size)
            current_balance += profit_loss
            
            # Record trade
            self.trades.append({
                'entry_time': self.position['entry_time'],
                'exit_time': results_df.iloc[-1]['timestamp'],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': last_price,
                'profit_loss': profit_loss,
                'exit_reason': 'end_of_backtest'
            })
            
            # Add detailed trade info
            pip_value = 0.0001  # For most forex pairs
            price_diff = last_price - entry_price
            if direction == 'short':
                price_diff = -price_diff
            pips_gained = price_diff / pip_value
            
            self.trade_details.append({
                **self.trades[-1],
                'lot_size': lot_size,
                'pips': pips_gained,
                'balance_after': current_balance
            })
            
            self.position = None
            self.balance_history.append((results_df.iloc[-1]['timestamp'], current_balance))
        
        return results_df
    
    def _calculate_profit_loss(self, entry_price, exit_price, direction, lot_size, include_fees=True):
        """Calculate profit/loss for a trade"""
        # Calculate raw price difference
        price_diff = exit_price - entry_price
        if direction == 'short':
            price_diff = -price_diff
        
        # Calculate profit in account currency
        profit = price_diff * lot_size
        
        # Deduct commission and slippage if requested
        if include_fees:
            commission = (entry_price + exit_price) * lot_size * self.commission_rate
            profit -= commission
        
        return profit
    
    def get_summary_statistics(self):
        """Calculate and return summary statistics for the backtest"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'net_profit': 0,
                'return_pct': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'avg_trade': 0,
                'avg_winning_trade': 0,
                'avg_losing_trade': 0,
                'largest_winner': 0,
                'largest_loser': 0
            }
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        losing_trades = [t for t in self.trades if t['profit_loss'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        gross_profit = sum(t['profit_loss'] for t in winning_trades)
        gross_loss = sum(t['profit_loss'] for t in losing_trades)
        net_profit = gross_profit + gross_loss
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Return metrics
        return_pct = (net_profit / self.initial_balance) * 100
        
        # Calculate drawdown
        max_drawdown = 0
        peak_balance = self.initial_balance
        
        for timestamp, balance in self.balance_history:
            peak_balance = max(peak_balance, balance)
            drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        # Trade metrics
        avg_trade = net_profit / total_trades if total_trades > 0 else 0
        avg_winning_trade = gross_profit / win_count if win_count > 0 else 0
        avg_losing_trade = gross_loss / loss_count if loss_count > 0 else 0
        
        largest_winner = max([t['profit_loss'] for t in self.trades]) if self.trades else 0
        largest_loser = min([t['profit_loss'] for t in self.trades]) if self.trades else 0
        
        # Risk metrics
        daily_returns = []
        if len(self.balance_history) > 1:
            for i in range(1, len(self.balance_history)):
                prev_balance = self.balance_history[i-1][1]
                curr_balance = self.balance_history[i][1]
                daily_return = (curr_balance - prev_balance) / prev_balance
                daily_returns.append(daily_return)
        
        sharpe_ratio = 0
        if daily_returns:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
            sharpe_ratio = (avg_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'return_pct': return_pct,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade': avg_trade,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'largest_winner': largest_winner,
            'largest_loser': largest_loser
        }
    
    def print_summary(self, include_trades=False):
        """Print backtest summary to console"""
        stats = self.get_summary_statistics()
        
        print("\n" + "="*50)
        print("BACKTEST SUMMARY")
        print("="*50)
        
        # Print general statistics
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${(self.initial_balance + stats['net_profit']):,.2f}")
        print(f"Net Profit: ${stats['net_profit']:,.2f} ({stats['return_pct']:.2f}%)")
        print(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        
        print("\nTrade Statistics:")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']*100:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Average Trade: ${stats['avg_trade']:,.2f}")
        print(f"Average Winner: ${stats['avg_winning_trade']:,.2f}")
        print(f"Average Loser: ${stats['avg_losing_trade']:,.2f}")
        print(f"Largest Winner: ${stats['largest_winner']:,.2f}")
        print(f"Largest Loser: ${stats['largest_loser']:,.2f}")
        
        # Print trade breakdown by exit reason
        exit_reasons = {}
        for trade in self.trades:
            reason = trade['exit_reason']
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'profit': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['profit'] += trade['profit_loss']
        
        print("\nTrade Exit Breakdown:")
        for reason, stats in exit_reasons.items():
            print(f"{reason}: {stats['count']} trades, ${stats['profit']:,.2f} profit")
        
        # Print individual trades if requested
        if include_trades and self.trades:
            print("\n" + "="*80)
            print("TRADE DETAILS")
            print("="*80)
            
            trade_table = []
            for i, trade in enumerate(self.trade_details, 1):
                profit_color = Fore.GREEN if trade['profit_loss'] > 0 else Fore.RED
                direction_str = Fore.BLUE + "LONG" + Style.RESET_ALL if trade['direction'] == 'long' else Fore.MAGENTA + "SHORT" + Style.RESET_ALL
                
                trade_table.append([
                    i,
                    trade['entry_time'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['entry_time'], 'strftime') else trade['entry_time'],
                    trade['exit_time'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['exit_time'], 'strftime') else trade['exit_time'],
                    direction_str,
                    f"{trade['entry_price']:.5f}",
                    f"{trade['exit_price']:.5f}",
                    f"{profit_color}{trade['pips']:.1f}{Style.RESET_ALL}",
                    f"{profit_color}${trade['profit_loss']:,.2f}{Style.RESET_ALL}",
                    trade['exit_reason']
                ])
            
            headers = ["#", "Entry Time", "Exit Time", "Direction", "Entry", "Exit", "Pips", "P/L", "Reason"]
            print(tabulate(trade_table, headers=headers, tablefmt="pipe"))
        
        return stats

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.name = "Base Strategy"
        
    def generate_signals(self, df):
        """
        Generate trading signals from analyzed data
        
        Args:
            df (DataFrame): Analyzed price dataframe with indicators
            
        Returns:
            DataFrame: Original dataframe with added signal columns
        """
        signals_df = df.copy()
        signals_df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        return signals_df


class WaveletGaussianStrategy(TradingStrategy):
    """Trading strategy using Wavelet Analysis and Gaussian Channels"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.name = "Wavelet-Gaussian Strategy"
        
    def generate_signals(self, df):
        """Generate signals based on Wavelet and Gaussian Channels"""
        signals_df = df.copy()
        signals_df['signal'] = 0
        signals_df['confidence'] = 0
        
        # Calculate price-channel relationship
        signals_df['channel_position'] = (signals_df['close'] - signals_df['gaussian_middle']) / \
                                        (signals_df['gaussian_upper'] - signals_df['gaussian_lower'])
        
        # Generate signals (with 1-period lag to avoid lookahead bias)
        for i in range(1, len(signals_df)):
            # Skip if not enough data
            if pd.isna(signals_df.iloc[i-1]['wavelet_denoised']) or \
               pd.isna(signals_df.iloc[i-1]['gaussian_middle']):
                continue
            
            momentum = 0
            if i > 1:
                # Calculate short-term momentum
                momentum = signals_df.iloc[i-1]['wavelet_denoised'] - signals_df.iloc[i-2]['wavelet_denoised']
            
            # Position relative to channels
            channel_pos = signals_df.iloc[i-1]['channel_position']
            
            # Confidence calculation (0-100)
            confidence = 50  # Base confidence
            
            # Channel breakout signals
            if channel_pos > 1:  # Price above upper channel
                if momentum > 0:
                    # Strong upward breakout
                    signals_df.iloc[i, signals_df.columns.get_loc('signal')] = 1  # Buy signal
                    confidence += 30
                elif momentum < 0:
                    # Potential reversal at resistance
                    signals_df.iloc[i, signals_df.columns.get_loc('signal')] = -1  # Sell signal
                    confidence += 20
            elif channel_pos < -1:  # Price below lower channel
                if momentum < 0:
                    # Strong downward breakout
                    signals_df.iloc[i, signals_df.columns.get_loc('signal')] = -1  # Sell signal
                    confidence += 30
                elif momentum > 0:
                    # Potential reversal at support
                    signals_df.iloc[i, signals_df.columns.get_loc('signal')] = 1  # Buy signal
                    confidence += 20
            
            # Trend-following signals
            elif abs(channel_pos) < 0.5:  # Price near middle band
                # Direction aligned with LOWESS trend
                curr_price = signals_df.iloc[i-1]['close']
                prev_price = signals_df.iloc[i-2]['close'] if i > 1 else curr_price
                lowess_slope = signals_df.iloc[i-1]['lowess_trend'] - signals_df.iloc[i-2]['lowess_trend'] if i > 1 else 0
                
                if lowess_slope > 0 and momentum > 0:
                    signals_df.iloc[i, signals_df.columns.get_loc('signal')] = 1  # Buy signal
                    confidence += 15
                elif lowess_slope < 0 and momentum < 0:
                    signals_df.iloc[i, signals_df.columns.get_loc('signal')] = -1  # Sell signal
                    confidence += 15
            
            # Stochastic confirmation
            if not pd.isna(signals_df.iloc[i-1]['stoch_k']) and not pd.isna(signals_df.iloc[i-1]['stoch_d']):
                k_value = signals_df.iloc[i-1]['stoch_k']
                d_value = signals_df.iloc[i-1]['stoch_d']
                
                # Stochastic crossover
                prev_k = signals_df.iloc[i-2]['stoch_k'] if i > 1 else k_value
                prev_d = signals_df.iloc[i-2]['stoch_d'] if i > 1 else d_value
                
                k_cross_above_d = prev_k <= prev_d and k_value > d_value
                k_cross_below_d = prev_k >= prev_d and k_value < d_value
                
                # Overbought/oversold conditions
                if k_value < 20 and d_value < 20:
                    if signals_df.iloc[i, signals_df.columns.get_loc('signal')] == 1:
                        confidence += 20  # Strong buy confirmation
                    elif k_cross_above_d:
                        signals_df.iloc[i, signals_df.columns.get_loc('signal')] = 1
                        confidence += 15
                elif k_value > 80 and d_value > 80:
                    if signals_df.iloc[i, signals_df.columns.get_loc('signal')] == -1:
                        confidence += 20  # Strong sell confirmation
                    elif k_cross_below_d:
                        signals_df.iloc[i, signals_df.columns.get_loc('signal')] = -1
                        confidence += 15
            
            # Store confidence value (capped at 100)
            signals_df.iloc[i, signals_df.columns.get_loc('confidence')] = min(100, confidence)
            
        return signals_df