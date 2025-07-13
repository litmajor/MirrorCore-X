import ccxt
import pandas as pd
import numpy as np
import time
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz

class MomentumScanner:
    def __init__(self, exchange, quote_currency='USDT', min_volume_usd=250_000, top_n=100, backtest_periods=7):
        self.exchange = exchange
        self.quote_currency = quote_currency
        self.min_volume_usd = min_volume_usd
        self.top_n = top_n
        self.backtest_periods = backtest_periods

        self.momentum_df = pd.DataFrame()
        self.fear_greed_history = []
        self.avg_momentum_history = []
        self.time_history = []
    def get_strong_signals(self):
        """Return only coins with strong momentum, RSI, MACD, and signal type."""
        if self.momentum_df.empty:
            return pd.DataFrame()
        df = self.momentum_df.copy()
        filtered = df[
            (df['momentum_7d'] > 0.06) &
            (df['rsi'] >= 50) & (df['rsi'] <= 65) &
            (df['macd'] > 0) &
            (df['signal'].isin(["Consistent Uptrend", "New Spike", "MACD Bullish"]))
        ]
        return filtered

    def fetch_markets(self):
        markets = self.exchange.load_markets()
        futures = [symbol for symbol, market in markets.items()
                   if market['quote'] == self.quote_currency
                   and market.get('contract', False)
                   and market['active']]
        return futures

    def fetch_ohlcv_data(self, symbol, timeframe='1d', limit=35):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching OHLCV data for {symbol}: {e}")
            return None

    def calculate_average_volume(self, df):
        return (df['volume'] * df['close']).mean()

    def calculate_momentum(self, df, period):
        return df['close'].pct_change(periods=period).iloc[-1]

    def calculate_rsi(self, df, period=14):
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period).mean().iloc[-1]
        avg_loss = pd.Series(loss).rolling(window=period).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line.iloc[-1] - signal_line.iloc[-1]

    def get_current_price(self, df):
        return df['close'].iloc[-1]

    def get_current_time(self):
        return datetime.now(pytz.timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S %Z')

    def fetch_fear_greed_index(self):
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url)
            data = response.json()
            return int(data['data'][0]['value'])
        except Exception as e:
            print(f"Error fetching Fear and Greed Index: {e}")
            return None

    def volume_trend_score(self, volume_today, volume_yesterday, threshold_up=0.05, threshold_down=-0.05):
        if volume_yesterday == 0:
            return 0
        change_pct = (volume_today - volume_yesterday) / volume_yesterday
        if change_pct > threshold_up:
            return 1
        elif change_pct < threshold_down:
            return -1
        else:
            return 0

    def classify_signal(self, mom7d, mom30d, rsi, macd):
        high_mom7d = 0.07
        high_mom30d = 0.28
        med_mom7d = 0.05
        med_mom30d = 0.20
        rsi_overbought = 80
        rsi_oversold = 20

        if mom7d > high_mom7d and mom30d > high_mom30d:
            return "Consistent Uptrend"
        elif mom7d > high_mom7d and mom30d < med_mom30d:
            return "New Spike"
        elif mom7d < med_mom7d and mom30d > high_mom30d:
            return "Topping Out"
        elif mom7d < med_mom7d and mom30d < med_mom30d:
            return "Lagging"
        elif mom7d > med_mom7d and mom30d > high_mom30d:
            return "Moderate Uptrend"
        elif mom7d < med_mom7d and mom30d > med_mom30d:
            return "Potential Reversal"
        elif mom7d > med_mom7d and mom30d > med_mom30d:
            return "Consolidation"
        elif mom7d > med_mom7d and mom30d < med_mom30d:
            return "Weak Uptrend"
        elif rsi > rsi_overbought and mom7d > high_mom7d:
            return "Overbought"
        elif rsi < rsi_oversold and mom7d < med_mom7d:
            return "Oversold"
        elif macd > 0 and mom7d > high_mom7d:
            return "MACD Bullish"
        elif macd < 0 and mom7d < med_mom7d:
            return "MACD Bearish"
        else:
            return "Neutral"

    def scan_market(self):
        futures = self.fetch_markets()
        results = []

        for symbol in futures:
            df = self.fetch_ohlcv_data(symbol, limit=35)
            if df is None or len(df) < 31:
                continue

            avg_volume = self.calculate_average_volume(df)
            if avg_volume < self.min_volume_usd:
                continue

            mom7d = self.calculate_momentum(df, 7)
            mom30d = self.calculate_momentum(df, 30)
            rsi = self.calculate_rsi(df)
            macd = self.calculate_macd(df)
            price = self.get_current_price(df)
            timestamp = self.get_current_time()
            signal = self.classify_signal(mom7d, mom30d, rsi, macd)

            results.append({
                'symbol': symbol,
                'momentum_7d': mom7d,
                'momentum_30d': mom30d,
                'rsi': rsi,
                'macd': macd,
                'average_volume_usd': avg_volume,
                'price': price,
                'timestamp': timestamp,
                'signal': signal
            })

        df_results = pd.DataFrame(results)
        df_results.dropna(subset=['momentum_7d', 'momentum_30d'], inplace=True)
        df_results.sort_values(by='momentum_7d', ascending=False, inplace=True)

        self.momentum_df = df_results.head(self.top_n)

        index_value = self.fetch_fear_greed_index()
        self.fear_greed_history.append(index_value)
        self.time_history.append(datetime.utcnow())
        avg_momentum = self.momentum_df['momentum_7d'].mean() if not self.momentum_df.empty else 0
        self.avg_momentum_history.append(avg_momentum)

        return self.momentum_df

    def execute_trades(self):
        positions = self.exchange.fetch_open_orders()
        current_positions = {order['symbol']: order for order in positions}

        for _, row in self.momentum_df.iterrows():
            symbol = row['symbol']
            signal = row['signal']
            price = row['price']

            if signal in ["Consistent Uptrend", "New Spike", "Moderate Uptrend"]:
                if symbol not in current_positions:
                    # Place a buy order
                    self.exchange.create_limit_buy_order(symbol, 1, price)
                    print(f"Buy order placed for {symbol} at {price}")
            elif signal in ["Topping Out", "Lagging", "Potential Reversal"]:
                if symbol in current_positions:
                    # Close the position
                    self.exchange.cancel_order(current_positions[symbol]['id'], symbol)
                    print(f"Position closed for {symbol}")

    def backtest(self, backtest_periods=None):
        if backtest_periods is None:
            backtest_periods = self.backtest_periods
        if self.momentum_df.empty:
            return pd.DataFrame()

        results = []

        for _, row in self.momentum_df.iterrows():
            symbol = row['symbol']
            df = self.fetch_ohlcv_data(symbol, limit=backtest_periods + 1)
            if df is None or len(df) < backtest_periods + 1:
                continue

            entry_price = df['close'].iloc[-backtest_periods - 1]
            exit_price = df['close'].iloc[-1]
            pnl = (exit_price - entry_price) / entry_price

            results.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': round(pnl * 100, 2)
            })

        backtest_df = pd.DataFrame(results)
        backtest_df['cumulative_return'] = (1 + backtest_df['pnl_pct'] / 100).cumprod()
        print(backtest_df[['symbol', 'entry_price', 'exit_price', 'pnl_pct', 'cumulative_return']])
        return backtest_df

    def plot_sentiment_vs_momentum(self):
        if not self.fear_greed_history:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.time_history, self.fear_greed_history, label="Fear & Greed Index", color='orange', marker='o')
        plt.plot(self.time_history, self.avg_momentum_history, label="Average 7d Momentum", color='blue', marker='x')
        plt.axhspan(15, 25, color='green', alpha=0.1, label='Buy Zone')
        plt.axhspan(65, 75, color='red', alpha=0.1, label='Sell Zone')
        plt.title("Fear & Greed Index vs Avg Momentum Over Time")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

if __name__ == "__main__":
    # Initialize the exchange
    exchange = ccxt.kucoinfutures({'enableRateLimit': True})

    # Initialize the MomentumScanner
    scanner = MomentumScanner(exchange)