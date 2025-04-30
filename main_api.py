# main_api.py (Updated with run command)

from fastapi import FastAPI, HTTPException, Query, Body, Path
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import datetime
import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid
import logging
import uvicorn # Import uvicorn

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import your existing core classes ---
try:
    from forextrader_core import ConfigManager, OANDADataFetcher, TechnicalAnalysis, WaveletGaussianStrategy, Backtester #
except ImportError as e:
    logging.error(f"Fatal Error: Could not import classes from forextrader_core.py: {e}")
    exit("API cannot start without core logic. Please ensure forextrader_core.py is accessible.")
except Exception as e:
    logging.error(f"Fatal Error: An unexpected error occurred during core class import: {e}")
    exit("API cannot start due to an unexpected error during import.")


# --- Constants ---
CONFIG_SAVE_DIR = Path("./backtest_configs")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Horizon Financial API", # Updated project name
    version="1.0",
    description="API to interact with the Horizon Financial Forex Trading System backend logic."
)

# --- Allow Cross-Origin Requests (CORS) ---
origins = ["*"] # Adjust for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Base Configuration ---
try:
    base_config_manager = ConfigManager(config_path="config.ini") #
    logging.info("Base configuration loaded successfully.")
except Exception as e:
    logging.error(f"Fatal Error: Could not load base configuration from config.ini: {e}")
    base_config_manager = None

# --- Instantiate Core Components (Global Instances) ---
data_fetcher = None
analyzer = None
strategy = None
backtester = None

if base_config_manager:
    try:
        # Using Demo OANDA account by default from config.ini
        data_fetcher = OANDADataFetcher(base_config_manager, use_live=False) #
        analyzer = TechnicalAnalysis(base_config_manager) #
        strategy = WaveletGaussianStrategy(base_config_manager) #
        backtester = Backtester(base_config_manager) #
        logging.info("Core trading components initialized successfully.")
    except Exception as e:
        logging.error(f"Fatal Error: Failed to initialize core components: {e}")
else:
    logging.warning("API starting without base configuration. Some features might be unavailable.")


# --- Pydantic Model for Saving Backtest Config ---
class BacktestConfigSaveRequest(BaseModel):
    config_name: Optional[str] = Field(None, description="Optional user-friendly name for the configuration")
    instrument: str = Field("EUR_USD", description="Trading instrument (e.g., EUR_USD)")
    granularity: str = Field("M5", description="Candlestick granularity (e.g., M1, M5, H1)")
    start_date: str = Field(..., description="Start date for backtest (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for backtest (YYYY-MM-DD)")
    initial_balance: float = Field(10000.0, gt=0, description="Initial account balance")
    lot_size: float = Field(1000.0, gt=0, description="Trade size in units")
    analysis_window: int = Field(20, ge=10, le=300, description="Window size for technical analysis")
    use_stop_loss: bool = Field(True, description="Enable stop loss")
    use_take_profit: bool = Field(True, description="Enable take profit")
    use_cached_data: bool = Field(True, description="Use cached OANDA data if available")
    # Add other parameters as needed...


# --- Ensure Config Directory Exists ---
try:
    CONFIG_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Configuration save directory checked/created: {CONFIG_SAVE_DIR.resolve()}")
except Exception as e:
    logging.error(f"Fatal Error: Could not create configuration directory {CONFIG_SAVE_DIR}: {e}")


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    if not all([base_config_manager, data_fetcher, analyzer, strategy, backtester]):
        logging.warning("API started with one or more core components uninitialized due to errors.")
    else:
        logging.info("API startup complete. All core components seem initialized.")

@app.get("/")
def read_root():
    """ Basic endpoint to check if API is running """
    if not all([base_config_manager, data_fetcher, analyzer, strategy, backtester]):
         raise HTTPException(status_code=503, detail="API core components failed to initialize properly.")
    return {"message": "Horizon Financial API is running"}

@app.post("/backtest/config", status_code=201)
async def save_backtest_config(config_data: BacktestConfigSaveRequest):
    """ Saves the received backtest configuration to a JSON file. """
    # ...(logic as before)...
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if config_data.config_name:
            safe_name = "".join(c for c in config_data.config_name if c.isalnum() or c in ('-', '_')).rstrip()
            filename_base = f"{safe_name}_{timestamp}"
        else:
            filename_base = f"config_{timestamp}"

        filename = f"{filename_base}.json"
        filepath = CONFIG_SAVE_DIR / filename

        with open(filepath, 'w') as f:
            f.write(config_data.json(indent=4))

        logging.info(f"Saved backtest config to {filepath}")
        return {"message": "Configuration saved successfully", "config_filename": filename}
    except Exception as e:
        logging.error(f"API Error saving config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {e}")


@app.post("/backtest/run/{config_filename}")
async def run_backtest_from_config(
    config_filename: str = Path(..., description="The filename of the saved configuration (e.g., config_YYYYMMDD_HHMMSS.json)")
    ):
    """ Loads a saved configuration and runs the backtest simulation. """
    # ...(logic as before)...
    if not all([base_config_manager, data_fetcher, analyzer, strategy, backtester]):
        raise HTTPException(status_code=503, detail="API core components are not initialized.")

    filepath = CONFIG_SAVE_DIR / config_filename

    if not filepath.is_file():
        logging.warning(f"Config file not found: {filepath}")
        raise HTTPException(status_code=404, detail=f"Configuration file '{config_filename}' not found in {CONFIG_SAVE_DIR}.")

    try:
        logging.info(f"Loading config file: {filepath}")
        with open(filepath, 'r') as f:
            saved_config_data = json.load(f)
            request_data = BacktestConfigSaveRequest(**saved_config_data)

        # Reconfigure components or use data directly from request_data
        current_data_fetcher = data_fetcher
        current_data_fetcher.instrument = request_data.instrument.upper().replace("/", "_")
        current_data_fetcher.granularity = request_data.granularity.upper()

        start_dt = datetime.datetime.strptime(request_data.start_date, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(request_data.end_date, "%Y-%m-%d")

        logging.info(f"Fetching data: {current_data_fetcher.instrument} ({current_data_fetcher.granularity}) from {request_data.start_date} to {request_data.end_date}")
        historical_data_df = current_data_fetcher.get_historical_data(start_dt, end_dt, use_cached=request_data.use_cached_data) #

        if historical_data_df is None or historical_data_df.empty:
             logging.error("Failed to fetch historical data.")
             raise HTTPException(status_code=404, detail="Failed to fetch historical data for the specified range.")
        logging.info(f"Fetched {len(historical_data_df)} candles.")

        logging.info("Analyzing data...")
        analysis_window = request_data.analysis_window
        analyzed_df = analyzer.analyze_price_series(historical_data_df, window=analysis_window) #
        logging.info("Analysis complete.")

        logging.info("Generating signals...")
        signaled_df = strategy.generate_signals(analyzed_df) #
        logging.info("Signal generation complete.")

        logging.info("Running backtest...")
        initial_balance = request_data.initial_balance
        lot_size = request_data.lot_size

        backtester.reset(initial_balance=initial_balance) #
        backtest_results_df = backtester.run_backtest( #
            signaled_df,
            lot_size=lot_size,
            use_stop_loss=request_data.use_stop_loss,
            use_take_profit=request_data.use_take_profit
        )
        logging.info("Backtest complete.")

        summary_stats = backtester.get_summary_statistics() #

        return {
            "message": f"Backtest completed successfully using config '{config_filename}'",
            "parameters_used": request_data.dict(),
            "summary_statistics": summary_stats
        }
    # ...(error handling as before)...
    except FileNotFoundError:
        logging.error(f"Configuration file not found during run: {filepath}")
        raise HTTPException(status_code=404, detail=f"Configuration file '{config_filename}' not found.")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in config file: {filepath}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in configuration file '{config_filename}'.")
    except Exception as e:
        logging.error(f"API Error during backtest run for {config_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during the backtest: {e}")

# --- Add other endpoints as needed ---
# Example:
# @app.get("/status")
# def get_status():
#    # Logic to get current system status (e.g., leverage, confidence)
#    return {"status": "OK", ...}


# --- Add block to run Uvicorn when script is executed directly ---
if __name__ == "__main__":
    print("Starting Horizon Financial API server...")
    # You can configure host and port here
    # Use host="0.0.0.0" to make it accessible on your network
    uvicorn.run(
        "main_api:app", # Points to the 'app' instance in the 'main_api.py' file
        host="127.0.0.1", # Only accessible from the local machine
        port=8000,        # Standard port for local development
        reload=True       # Enables auto-reload on code changes (good for development)
        # workers=4       # Optional: Specify number of worker processes for production
        # log_level="info" # Control logging level
    )
    # Note: Uvicorn handles port shutdown when you stop the script (e.g., Ctrl+C)