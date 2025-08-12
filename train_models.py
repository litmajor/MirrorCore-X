
import pandas as pd
import ccxt.async_support as ccxt
import asyncio
import multiprocessing
from typing import Dict, Optional
from dataclasses import dataclass
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
import logging
from datetime import datetime, timezone

# Assuming scanner.py defines MomentumScanner and TradingConfig
from scanner import MomentumScanner, TradingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define dynamic config
def get_dynamic_config() -> TradingConfig:
    cpu_count = multiprocessing.cpu_count()
    max_concurrent = min(max(20, cpu_count * 5), 100)
    return TradingConfig(
        timeframes={
            "scalping": "1m",
            "short": "5m",
            "medium": "1h",
            "daily": "1d",
            "weekly": "1w"
        },
        backtest_periods={
            "scalping": 100,
            "short": 50,
            "medium": 24,
            "daily": 7,
            "weekly": 4
        },
        momentum_periods={
            "crypto": {
                "scalping": {"short": 10, "long": 60},
                "short": {"short": 5, "long": 20},
                "medium": {"short": 4, "long": 12},
                "daily": {"short": 7, "long": 30},
                "weekly": {"short": 4, "long": 12}
            },
            "forex": {
                "scalping": {"short": 20, "long": 120},
                "short": {"short": 10, "long": 50},
                "medium": {"short": 6, "long": 24},
                "daily": {"short": 5, "long": 20},
                "weekly": {"short": 3, "long": 10}
            }
        },
        signal_thresholds={
            "crypto": {
                "scalping": {"momentum_short": 0.01, "rsi_min": 55, "rsi_max": 70, "macd_min": 0},
                "short": {"momentum_short": 0.03, "rsi_min": 52, "rsi_max": 68, "macd_min": 0},
                "medium": {"momentum_short": 0.05, "rsi_min": 50, "rsi_max": 65, "macd_min": 0},
                "daily": {"momentum_short": 0.06, "rsi_min": 50, "rsi_max": 65, "macd_min": 0},
                "weekly": {"momentum_short": 0.15, "rsi_min": 45, "rsi_max": 70, "macd_min": 0}
            },
            "forex": {
                "scalping": {"momentum_short": 0.002, "rsi_min": 50, "rsi_max": 70, "macd_min": 0},
                "short": {"momentum_short": 0.005, "rsi_min": 48, "rsi_max": 68, "macd_min": 0},
                "medium": {"momentum_short": 0.008, "rsi_min": 47, "rsi_max": 67, "macd_min": 0},
                "daily": {"momentum_short": 0.01, "rsi_min": 45, "rsi_max": 65, "macd_min": 0},
                "weekly": {"momentum_short": 0.03, "rsi_min": 40, "rsi_max": 70, "macd_min": 0}
            }
        },
        trade_durations={
            "scalping": 1800,
            "short": 14400,
            "medium": 86400,
            "daily": 604800,
            "weekly": 2592000
        },
        max_concurrent_requests=max_concurrent
    )

async def fetch_future_price(exchange, symbol: str, timeframe: str, periods: int) -> float:
    """Fetch closing price after specified periods for given timeframe."""
    timeframe_seconds = {
        '1m': 60,
        '5m': 300,
        '1h': 3600,
        '1d': 86400,
        '1w': 604800
    }
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=periods+1)
        return ohlcv[-1][4]  # Closing price
    except Exception as e:
        logger.error(f"Error fetching future price for {symbol} on {timeframe}: {e}")
        return np.nan
    finally:
        await exchange.close()

async def run_scanner(timeframe: str, config: TradingConfig, output_path: str):
    """Run MomentumScanner for a given timeframe and save results."""
    exchange = ccxt.kucoinfutures({'enableRateLimit': True})
    scanner = MomentumScanner(
        exchange=exchange,
        quote_currency='USDT',
        min_volume_usd=500000,
        top_n=50,  # Increased to ensure more signal diversity
        config=config
    )
    try:
        await scanner.scan_market(timeframe)
        scanner.scan_results.to_csv(output_path)
        logger.info(f"Scan results for {timeframe} saved to {output_path}")
    except Exception as e:
        logger.error(f"Error running scanner for {timeframe}: {e}")
    finally:
        await exchange.close()

import os

async def prepare_data(config: TradingConfig):
    """Prepare data for ML by generating multi-timeframe scans and computing targets."""
    import shutil
    timeframes = config.timeframes
    dfs = {}

    # Step 1: Run scans for all timeframes with per-timeframe caching
    for strategy, tf in timeframes.items():
        cached_file = f'scan_results_{tf}_latest.csv'
        if os.path.exists(cached_file):
            logger.info(f"Using cached scan file for {tf}: {cached_file}")
            dfs[tf] = pd.read_csv(cached_file)
        else:
            output_path = f'scan_results_{tf}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            logger.info(f"No cached scan found for {tf}, running fresh scan...")
            await run_scanner(tf, config, output_path)
            dfs[tf] = pd.read_csv(output_path)
            # Save/copy to latest for reuse
            shutil.copy(output_path, cached_file)

    # Step 2: Compute return targets
    exchange = ccxt.kucoinfutures({'enableRateLimit': True})
    for tf, df in dfs.items():
        periods = config.backtest_periods[list(timeframes.keys())[list(timeframes.values()).index(tf)]]
        future_prices = []
        for symbol in df['symbol']:
            price = await fetch_future_price(exchange, symbol, tf, periods)
            future_prices.append(price)
        df[f'{tf}_return'] = (pd.Series(future_prices) - df['price']) / df['price'] * 100
        dfs[tf] = df

    # Step 3: Compute signal consistency
    for tf, df in dfs.items():
        df['consistent'] = 0
        for symbol in df['symbol']:
            signal = df[df['symbol'] == symbol]['signal'].iloc[0]
            matches = sum(1 for other_tf, other_df in dfs.items() if symbol in other_df['symbol'].values and other_df[other_df['symbol'] == symbol]['signal'].iloc[0] == signal)
            df.loc[df['symbol'] == symbol, 'consistent'] = 1 if matches >= 3 else 0
        dfs[tf].to_csv(f'processed_scan_results_{tf}.csv')
    return dfs

def train_models(df: pd.DataFrame, timeframe: str):
    """Train XGBoost models for return prediction and signal consistency."""
    features = [
        'momentum_short', 'momentum_long', 'rsi', 'macd', 'volume_ratio',
        'composite_score', 'volume_composite_score', 'trend_score',
        'confidence_score', 'bb_position', 'trend_strength', 'stoch_k',
        'stoch_d', 'poc_distance', 'ichimoku_bullish', 'vwap_bullish',
        'rsi_bearish_div', 'ema_5_13_bullish', 'ema_9_21_bullish',
        'ema_50_200_bullish'
    ]
    df = df.dropna(subset=features + [f'{timeframe}_return'])

    # Return prediction (regression)
    X_return = df[features]
    y_return = df[f'{timeframe}_return']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_return, y_return, test_size=0.2, random_state=42)
    model_return = xgb.XGBRegressor(random_state=42)
    model_return.fit(X_train_r, y_train_r)
    y_pred_r = model_return.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred_r)
    logger.info(f"{timeframe} Return Prediction MSE: {mse:.2f}")

    # Signal consistency (classification)
    model_consistent = None
    X_consistent = None
    if 'consistent' in df.columns:
        y_consistent = df['consistent']
        if len(np.unique(y_consistent)) > 1:  # Check for multiple classes
            X_consistent = df[features]
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_consistent, y_consistent, test_size=0.2, random_state=42)
            # Apply SMOTE to balance classes if needed
            smote = SMOTE(random_state=42)
            smote_result = smote.fit_resample(X_train_c, y_train_c)
            if len(smote_result) == 2:
                X_train_c, y_train_c = smote_result
            else:
                X_train_c, y_train_c, _ = smote_result
            model_consistent = xgb.XGBClassifier(random_state=42)
            model_consistent.fit(X_train_c, y_train_c)
            y_pred_c = model_consistent.predict(X_test_c)
            accuracy = accuracy_score(y_test_c, y_pred_c)
            logger.info(f"{timeframe} Signal Consistency Accuracy: {accuracy:.2f}")
        else:
            logger.warning(f"Skipping signal consistency training for {timeframe}: Only one class found in 'consistent' column")
    else:
        logger.warning(f"Skipping signal consistency training for {timeframe}: 'consistent' column missing")

    # Save models and predictions
    model_return.save_model(f'model_return_{timeframe}.json')
    if model_consistent and X_consistent is not None:
        model_consistent.save_model(f'model_consistent_{timeframe}.json')
        df[f'predicted_consistent'] = model_consistent.predict(X_consistent)
    else:
        df[f'predicted_consistent'] = 0  # Default to 0 if no model trained
    df.to_csv(f'predictions_{timeframe}.csv')
    return model_return, model_consistent
    return model_return, model_consistent

async def main():
    config = get_dynamic_config()
    dfs = await prepare_data(config)
    for tf, df in dfs.items():
        logger.info(f"Training models for timeframe: {tf}")
        model_return, model_consistent = train_models(df, tf)

if __name__ == "__main__":
    asyncio.run(main())
