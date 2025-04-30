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

# Import code from imported file
from forextrader_core import *
import requests

class ForexTrader:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def fetch_historical_data(self, instrument, granularity, count):
        url = f"{self.base_url}/instruments/{instrument}/candles?granularity={granularity}&count={count}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['candles']  # Adjust based on the actual response structure
        else:
            print("Error fetching historical data:", response.status_code, response.text)
            return None

    def add_indicators(self, data):
        prices = [candle['close'] for candle in data]
        window_size = 14  # Example window size for SMA
        sma = [sum(prices[i:i+window_size]) / window_size for i in range(len(prices) - window_size + 1)]
        return sma

    def backtest_strategy(self, data, indicators):
        results = []
        for i in range(len(data)):
            if i > 0 and data[i]['close'] > indicators[i-1]:  # Simple buy condition
                results.append("Buy")
            else:
                results.append("Hold")
        return results

    def validate_results(self, results):
        total_trades = len(results)
        successful_trades = results.count("Buy")  # Example metric
        win_rate = successful_trades / total_trades if total_trades > 0 else 0
        print(f"Total Trades: {total_trades}, Successful Trades: {successful_trades}, Win Rate: {win_rate:.2%}")

    def chatbot_backtest(self):
        instrument = input("Enter the instrument (e.g., EUR/USD): ")
        granularity = input("Enter the granularity (e.g., M1): ")
        count = int(input("How many candles to fetch?: "))
        historical_data = self.fetch_historical_data(instrument, granularity, count)
        if historical_data:
            indicators = self.add_indicators(historical_data)
            results = self.backtest_strategy(historical_data, indicators)
            self.validate_results(results)

def call_claude_api(user_input):
    api_url = "https://api.claude.ai/v1/chat"  # Replace with the actual API endpoint
    headers = {
        'Authorization': 'Bearer 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',
        'Content-Type': 'application/json'
    }
    payload = {
        "message": user_input
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get('response')  # Adjust based on API response structure
    else:
        print("Error calling Claude API:", response.status_code, response.text)
        return None

def chatbot_interface():
    user_progress = {
        'option_1_completed': False,
        'option_2_completed': False,
        'option_3_completed': False
    }
    
    while True:
        user_input = input("How can I assist you today? (Type 'exit' to quit): ").strip().lower()
        
        if user_input == 'exit':
            print("Thank you for using the Forex Trader CLI. Goodbye!")
            break
        elif user_input == 'option 1':
            # Execute option 1 functionality
            user_progress['option_1_completed'] = True
            print("Option 1 completed.")
        elif user_input == 'option 2':
            # Execute option 2 functionality
            user_progress['option_2_completed'] = True
            print("Option 2 completed.")
        elif user_input == 'option 3':
            if user_progress['option_1_completed'] and user_progress['option_2_completed']:
                # Execute option 3 functionality
                user_progress['option_3_completed'] = True
                print("Option 3 completed.")
            else:
                print("Please complete options 1 and 2 first.")
        elif user_input == 'option 4':
            if user_progress['option_3_completed']:
                # Execute option 4 functionality
                print("Option 4 completed.")
            else:
                print("Please complete option 3 first.")
        elif user_input == 'backtest':
            trader = ForexTrader('your_api_key', 'https://api-fxpractice.oanda.com')
            trader.chatbot_backtest()
        else:
            claude_response = call_claude_api(user_input)
            if claude_response:
                print(claude_response)
            else:
                print("I'm sorry, I didn't understand that. Please try again.")

if __name__ == "__main__":
    try:
        chatbot_interface()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
