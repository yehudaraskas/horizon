import requests
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import json
import logging

# Configure logger
logger = logging.getLogger(__name__)

class OANDADataFetcher:
    def __init__(self, api_key, instrument="EUR_USD", granularity="M1"):
        self.api_key = api_key
        self.instrument = instrument
        self.granularity = granularity
        self.base_url = "https://api-fxtrade.oanda.com/v3"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def generate_random_period(self, years_back=3):
        """Generate a random 3-month period within the specified years back"""
        now = datetime.now()
        max_days_back = years_back * 365
        end_date = now - timedelta(days=random.randint(0, max_days_back))
        start_date = end_date - timedelta(days=90)
        return start_date, end_date
    
    def fetch_candles(self, start_date, end_date, chunk_size=5000, progress_callback=None):
        """Fetch candles in chunks to handle API limits
        
        Args:
            start_date (datetime): Start date for data fetch
            end_date (datetime): End date for data fetch
            chunk_size (int): Number of candles per request
            progress_callback (callable): Function to call with progress updates
            
        Returns:
            list: List of candle dictionaries with OHLCV data
        """
        logger.info(f"Fetching {self.instrument} {self.granularity} data from {start_date} to {end_date}")
        
        all_candles = []
        total_minutes = int((end_date - start_date).total_seconds() / 60)
        processed_minutes = 0
        current_start = start_date
        
        # Validate dates
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        if (end_date - start_date).days > 180:
            logger.warning("Fetching more than 180 days of data. This might take a while.")
        
            # Real OANDA API calls
            retry_count = 0
            max_retries = 3
            
            while current_start < end_date:
                chunk_end = min(current_start + timedelta(minutes=chunk_size), end_date)
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
                    else:
                        print(f"Error {response.status_code}: {response.text}")
                        return None
                    
                    time.sleep(0.1)
                    
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        return None
                    
                    logger.warning(f"Request failed (attempt {retry_count}/{max_retries}): {e}")
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                    
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    return None
                
                # Reset retry count on successful request
                retry_count = 0
                
                current_start = chunk_end
                processed_minutes += chunk_size
                
                if progress_callback:
                    progress = (processed_minutes / total_minutes) * 100
                    progress_callback({
                        'stage': 'Fetching historical data...',
                        'progress': min(95, progress)
                    })
            
            return all_candles
    
    def process_candles(self, candles):
        """Convert candles to pandas DataFrame with proper formatting
        
        Args:
            candles (list): List of candle dictionaries from Oanda API
            
        Returns:
            pandas.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
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
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Check for missing data
        expected_periods = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.granularity)
        missing_periods = expected_periods.difference(df.index)
        
        if len(missing_periods) > 0:
            logger.warning(f"Found {len(missing_periods)} missing periods in the data")
            logger.debug(f"First few missing periods: {missing_periods[:5]}")
        
        return df
    
    def save_data(self, df, filename):
        """Save data to CSV"""
        if df is not None:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            print(f"Total rows: {len(df)}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
    def fetch_and_save_random_period(self):
        """Main function to fetch and save data for a random period"""
        start_date, end_date = self.generate_random_period()
        print(f"Fetching data for period: {start_date} to {end_date}")
        
        candles = self.fetch_candles(start_date, end_date)
        if candles:
            df = self.process_candles(candles)
            filename = f"EURUSD_M1_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            self.save_data(df, filename)
            return filename
        return None

# Usage example
if __name__ == "__main__":
    API_KEY = "19a9220ae67f6f6afdc1b3386c476bb7-bce51a4f79755a66d3300305fc1b59a6"   
    fetcher = OANDADataFetcher(API_KEY)
    filename = fetcher.fetch_and_save_random_period()
