import yfinance as yf
import pandas as pd
from typing import Optional

class YFinanceForexAdapter:
    def close(self):
        """No-op close method for compatibility with CCXT exchanges."""
        pass
    """
    Adapter to provide a CCXT-like interface for fetching forex data from yfinance.
    """
    def __init__(self):
        pass

    async def fetch_ticker(self, symbol: str) -> dict:
        # yfinance expects forex symbols like 'EURUSD=X'
        yf_symbol = symbol if symbol.endswith('=X') else symbol + '=X'
        data = yf.Ticker(yf_symbol).history(period='1d')
        if data.empty:
            raise ValueError(f"No data for symbol {yf_symbol}")
        last_row = data.iloc[-1]
        return {
            'symbol': symbol,
            'bid': last_row['Close'],
            'ask': last_row['Close'],
            'last': last_row['Close'],
            'datetime': pd.Timestamp(str(last_row.name)).to_pydatetime(),
            'info': last_row.to_dict(),
        }

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1d', limit: int = 100) -> Optional[pd.DataFrame]:
        # Map timeframe to yfinance interval
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h',
            '1d': '1d', '1wk': '1wk', '1w': '1wk', '1mo': '1mo', '1M': '1mo'
        }
        yf_symbol = symbol if symbol.endswith('=X') else symbol + '=X'
        interval = interval_map.get(timeframe, '1d')
        data = yf.Ticker(yf_symbol).history(period=f'{limit}d', interval=interval)
        if data.empty:
            return None
        data = data.tail(limit)
        data = data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        data = data[['open', 'high', 'low', 'close', 'volume']]
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'timestamp'}, inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
