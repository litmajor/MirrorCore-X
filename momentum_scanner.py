
"""
Enhanced Momentum Scanner with Multi-timeframe Analysis and Advanced Features
Addresses continuity, multi-timeframe analysis, and mean reversion detection
"""

import asyncio
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ContinuousMarketScanner:
    """
    Enhanced scanner that provides continuous data streams to cognitive agents
    Supports multiple timeframes and mean reversion detection
    """
    
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': '',
            'secret': '',
            'sandbox': True,
            'enableRateLimit': True,
        })
        
        # Multi-timeframe configuration
        self.timeframes = {
            'scalp': '5m',      # 60-100 min style (assuming ~20 candles)
            'day_trade': '4h',  # Day trading optimal timeframe
            'position': '1d'    # 7-day style
        }
        
        # Data persistence for ML/RL training
        self.data_storage = DataPersistenceManager()
        
        # Candle clustering configuration
        self.volume_threshold_multiplier = 2.0  # High volume = 2x average
        self.cluster_lookback = 10  # Look back 10 candles for clustering
        
        # Data storage for continuity
        self.market_data = {}
        self.signal_history = {}
        self.market_state = {}
        
        # Scanner state
        self.is_running = False
        self.scan_interval = 90  # 90 seconds - based on your 1-2min scan time
        self.momentum_bias = 0.6  # 60% momentum, 40% mean reversion balance

    async def start_continuous_scanning(self):
        """Start continuous market monitoring"""
        self.is_running = True
        
        # Initialize data structures
        await self._initialize_markets()
        
        # Start parallel scanning tasks
        tasks = [
            self._continuous_price_updates(),
            self._periodic_signal_generation(),
            self._market_state_analysis()
        ]
        
        await asyncio.gather(*tasks)
    
    async def _initialize_markets(self):
        """Initialize market data storage"""
        markets = await self.exchange.load_markets()
        usdt_markets = [symbol for symbol in markets if '/USDT' in symbol]
        
        for symbol in usdt_markets[:50]:  # Top 50 for performance
            self.market_data[symbol] = {
                timeframe: pd.DataFrame() 
                for timeframe in self.timeframes.values()
            }
            self.signal_history[symbol] = []
    
    async def _continuous_price_updates(self):
        """Continuous price monitoring (market ticks)"""
        while self.is_running:
            try:
                # Get ticker updates for all monitored markets
                tickers = await self.exchange.fetch_tickers()
                
                tick_data = {
                    'timestamp': datetime.now(),
                    'tickers': tickers
                }
                
                # Process tick data
                await self._process_tick(tick_data)
                
                # Short interval for real-time updates
                await asyncio.sleep(5)  # 5-second ticks
                
            except Exception as e:
                logger.error(f"Error in price updates: {e}")
                await asyncio.sleep(10)
    
    async def _periodic_signal_generation(self):
        """Generate signals periodically"""
        while self.is_running:
            try:
                # Generate signals for all timeframes
                for style, timeframe in self.timeframes.items():
                    signals = await self._generate_timeframe_signals(style, timeframe)
                    await self._store_signals(signals)
                
                # Wait for scan interval
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in signal generation: {e}")
                await asyncio.sleep(30)
    
    async def _market_state_analysis(self):
        """Analyze overall market state"""
        while self.is_running:
            try:
                market_state = await self._analyze_market_regime()
                self.market_state = market_state
                
                # Update every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in market state analysis: {e}")
                await asyncio.sleep(60)

    async def _generate_timeframe_signals(self, style: str, timeframe: str) -> List[Dict]:
        """Generate signals for a specific timeframe with enhanced analysis"""
        signals = []
        
        for symbol in self.market_data:
            try:
                # Fetch fresh OHLCV data
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                if len(df) < 20:
                    continue
                
                # Enhanced signal generation
                signal_data = await self._analyze_symbol_enhanced(symbol, df, style)
                signals.append(signal_data)
                
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        return signals

    async def _analyze_symbol_enhanced(self, symbol: str, df: pd.DataFrame, style: str) -> Dict:
        """Enhanced analysis with new features"""
        signals = {}
        
        # Basic momentum from existing scanner
        signals.update(self._detect_momentum(df))
        
        # NEW: Mean reversion detection with reverse momentum logic
        reversion_signals = self._detect_smart_mean_reversion(df, style)
        
        # NEW: Candle clustering analysis
        cluster_signals = self._detect_candle_clustering(df)
        
        # NEW: Momentum with clustering validation
        enhanced_momentum = self._detect_enhanced_momentum(df, cluster_signals)
        
        # NEW: Market regime detection
        regime_signals = self._detect_market_regime(df)
        
        # Style-specific logic with day trading addition
        if style == 'scalp':
            signals.update(self._scalp_specific_analysis(df))
        elif style == 'day_trade':
            signals.update(self._day_trade_specific_analysis(df))
        elif style == 'position':
            signals.update(self._position_specific_analysis(df))
        
        return {
            'momentum': enhanced_momentum,
            'reversion': reversion_signals,
            'regime': regime_signals,
            'clustering': cluster_signals,
            'style_specific': signals,
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe': style
        }
    
    def _detect_momentum(self, df: pd.DataFrame) -> Dict:
        """Basic momentum detection from existing scanner"""
        if len(df) < 20:
            return {}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Calculate momentum indicators
        momentum_7d = (latest['close'] - df.iloc[-7]['close']) / df.iloc[-7]['close'] if len(df) >= 7 else 0
        momentum_30d = (latest['close'] - df.iloc[-30]['close']) / df.iloc[-30]['close'] if len(df) >= 30 else 0
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD calculation
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        
        return {
            'momentum_7d': momentum_7d,
            'momentum_30d': momentum_30d,
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'macd': macd.iloc[-1] if not macd.empty else 0,
            'price': latest['close'],
            'volume': latest['volume']
        }
    
    def _detect_smart_mean_reversion(self, df: pd.DataFrame, style: str) -> Dict:
        """Detect mean reversion opportunities with reverse momentum logic"""
        if len(df) < 20:
            return {'reversion_probability': 0, 'reversion_candidate': False}
        
        # Calculate indicators
        close = df['close']
        
        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # RSI for overbought/oversold
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Momentum divergence
        momentum_short = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        momentum_long = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
        
        # Reversion probability calculation
        reversion_score = 0
        
        # Extreme BB position
        if bb_position > 0.8:  # Near upper band
            reversion_score += 0.3
        elif bb_position < 0.2:  # Near lower band
            reversion_score += 0.3
        
        # RSI extremes
        if current_rsi > 70:
            reversion_score += 0.3
        elif current_rsi < 30:
            reversion_score += 0.3
        
        # Momentum divergence (sign of exhaustion)
        if abs(momentum_short) > abs(momentum_long) and momentum_short * momentum_long < 0:
            reversion_score += 0.4
        
        # Volume confirmation (high volume at extremes often signals reversal)
        volume_ma = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1]
        if volume_ratio > 1.5 and reversion_score > 0.3:
            reversion_score += 0.2
        
        return {
            'reversion_probability': min(reversion_score, 1.0),
            'reversion_candidate': reversion_score > 0.6,
            'bb_position': bb_position,
            'rsi_extreme': current_rsi > 70 or current_rsi < 30,
            'momentum_divergence': momentum_short * momentum_long < 0,
            'volume_confirmation': volume_ratio > 1.5
        }
    
    def _detect_candle_clustering(self, df: pd.DataFrame) -> Dict:
        """Detect candle clustering patterns for validation"""
        if len(df) < self.cluster_lookback:
            return {'cluster_detected': False, 'cluster_strength': 0}
        
        recent_candles = df.tail(self.cluster_lookback)
        
        # Volume clustering
        avg_volume = recent_candles['volume'].mean()
        volume_threshold = avg_volume * self.volume_threshold_multiplier
        high_volume_candles = (recent_candles['volume'] > volume_threshold).sum()
        
        # Price clustering (tight ranges or breakouts)
        price_range = recent_candles['high'].max() - recent_candles['low'].min()
        avg_range = (recent_candles['high'] - recent_candles['low']).mean()
        range_ratio = price_range / (avg_range * len(recent_candles))
        
        # Directional clustering
        green_candles = (recent_candles['close'] > recent_candles['open']).sum()
        red_candles = (recent_candles['close'] < recent_candles['open']).sum()
        directional_bias = abs(green_candles - red_candles) / len(recent_candles)
        
        cluster_strength = (
            (high_volume_candles / len(recent_candles)) * 0.4 +
            min(range_ratio, 1.0) * 0.3 +
            directional_bias * 0.3
        )
        
        return {
            'cluster_detected': cluster_strength > 0.6,
            'cluster_strength': cluster_strength,
            'high_volume_ratio': high_volume_candles / len(recent_candles),
            'directional_bias': directional_bias,
            'range_compression': range_ratio < 0.5,  # Tight range indicates potential breakout
            'volume_expansion': high_volume_candles > len(recent_candles) * 0.3
        }
    
    def _detect_enhanced_momentum(self, df: pd.DataFrame, cluster_signals: Dict) -> Dict:
        """Enhanced momentum detection validated by clustering"""
        basic_momentum = self._detect_momentum(df)
        
        if not cluster_signals.get('cluster_detected', False):
            return {**basic_momentum, 'cluster_validated': False}
        
        # Momentum is enhanced when supported by clustering
        cluster_strength = cluster_signals.get('cluster_strength', 0)
        momentum_strength = abs(basic_momentum.get('momentum_7d', 0))
        
        enhanced_score = momentum_strength * (1 + cluster_strength)
        
        return {
            **basic_momentum,
            'cluster_validated': True,
            'enhanced_momentum_score': enhanced_score,
            'momentum_reliability': cluster_strength
        }
    
    def _detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """Detect market regime (trending vs ranging)"""
        if len(df) < 50:
            return {'regime': 'unknown'}
        
        close = df['close']
        
        # ADX for trend strength
        high, low = df['high'], df['low']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        
        atr = true_range.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(14).mean()
        
        current_adx = adx.iloc[-1] if not adx.empty else 25
        
        # Volatility regime
        returns = close.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        vol_percentile = (volatility.iloc[-1] - volatility.quantile(0.2)) / (volatility.quantile(0.8) - volatility.quantile(0.2))
        
        # Determine regime
        if current_adx > 30:
            trend_regime = 'trending'
        elif current_adx < 20:
            trend_regime = 'ranging'
        else:
            trend_regime = 'transitional'
        
        if vol_percentile > 0.7:
            vol_regime = 'high_volatility'
        elif vol_percentile < 0.3:
            vol_regime = 'low_volatility'
        else:
            vol_regime = 'normal_volatility'
        
        return {
            'trend_regime': trend_regime,
            'volatility_regime': vol_regime,
            'adx': current_adx,
            'volatility_percentile': vol_percentile
        }
    
    def _scalp_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis specific to scalping timeframe"""
        if len(df) < 10:
            return {}
        
        # Short-term momentum
        momentum_1m = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        momentum_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else momentum_1m
        
        # Micro support/resistance
        recent_highs = df['high'].tail(20)
        recent_lows = df['low'].tail(20)
        current_price = df['close'].iloc[-1]
        
        resistance_distance = (recent_highs.max() - current_price) / current_price
        support_distance = (current_price - recent_lows.min()) / current_price
        
        return {
            'micro_momentum_1m': momentum_1m,
            'micro_momentum_5m': momentum_5m,
            'resistance_distance': resistance_distance,
            'support_distance': support_distance,
            'scalp_signal': 'buy' if momentum_5m > 0.001 and support_distance > 0.002 else 'sell' if momentum_5m < -0.001 and resistance_distance > 0.002 else 'hold'
        }
    
    def _day_trade_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis specific to day trading timeframe (NEW)"""
        if len(df) < 24:  # Need at least 24 hours of 4h data
            return {}
        
        # Session analysis (assuming 4h timeframes)
        session_data = df.tail(6)  # Last 24 hours
        session_high = session_data['high'].max()
        session_low = session_data['low'].min()
        current_price = df['close'].iloc[-1]
        
        # Position within session range
        session_position = (current_price - session_low) / (session_high - session_low) if session_high != session_low else 0.5
        
        # Momentum persistence
        momentum_4h = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        momentum_12h = (df['close'].iloc[-1] - df['close'].iloc[-3]) / df['close'].iloc[-3] if len(df) >= 3 else momentum_4h
        momentum_persistence = 1 if momentum_4h * momentum_12h > 0 else 0
        
        # Volume profile within session
        session_volume = session_data['volume'].sum()
        avg_session_volume = df['volume'].rolling(6).sum().mean()
        volume_ratio = session_volume / avg_session_volume if avg_session_volume > 0 else 1
        
        # Day trade signal
        day_trade_signal = 'buy' if (session_position < 0.3 and momentum_persistence and volume_ratio > 1.2) else \
                          'sell' if (session_position > 0.7 and momentum_persistence and volume_ratio > 1.2) else 'hold'
        
        return {
            'session_position': session_position,
            'momentum_persistence': momentum_persistence,
            'session_volume_ratio': volume_ratio,
            'day_trade_signal': day_trade_signal,
            'session_range_pct': (session_high - session_low) / session_low if session_low > 0 else 0
        }
    
    def _position_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis specific to position trading timeframe"""
        if len(df) < 30:
            return {}
        
        # Weekly/monthly trends
        momentum_7d = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] if len(df) >= 7 else 0
        momentum_30d = (df['close'].iloc[-1] - df['close'].iloc[-30]) / df['close'].iloc[-30] if len(df) >= 30 else momentum_7d
        
        # Moving average trends
        ma_20 = df['close'].rolling(20).mean()
        ma_50 = df['close'].rolling(50).mean() if len(df) >= 50 else ma_20
        
        current_price = df['close'].iloc[-1]
        ma_20_current = ma_20.iloc[-1]
        ma_50_current = ma_50.iloc[-1]
        
        # Trend alignment
        trend_alignment = (
            (current_price > ma_20_current > ma_50_current) or
            (current_price < ma_20_current < ma_50_current)
        )
        
        # Position size suggestion based on conviction
        conviction_score = abs(momentum_30d) * (1.5 if trend_alignment else 1.0)
        position_size = min(conviction_score * 10, 1.0)  # Max 100% position
        
        return {
            'momentum_7d': momentum_7d,
            'momentum_30d': momentum_30d,
            'trend_alignment': trend_alignment,
            'conviction_score': conviction_score,
            'suggested_position_size': position_size,
            'position_signal': 'buy' if momentum_30d > 0.05 and trend_alignment else 'sell' if momentum_30d < -0.05 and trend_alignment else 'hold'
        }

    async def get_live_signals(self, style: str = 'all') -> Dict:
        """Get latest signals for specified trading style"""
        if style == 'all':
            return {
                style: await self._get_style_signals(style)
                for style in self.timeframes.keys()
            }
        else:
            return await self._get_style_signals(style)
    
    async def _get_style_signals(self, style: str) -> List[Dict]:
        """Get signals for specific style"""
        signals = []
        for symbol in self.signal_history:
            if self.signal_history[symbol]:
                latest_signal = self.signal_history[symbol][-1]
                if latest_signal.get('timeframe') == style:
                    signals.append(latest_signal)
        return sorted(signals, key=lambda x: x.get('momentum', {}).get('enhanced_momentum_score', 0), reverse=True)
    
    async def _store_signals(self, signals: List[Dict]):
        """Store signals in history and persist to storage"""
        for signal in signals:
            symbol = signal.get('symbol')
            if symbol:
                if symbol not in self.signal_history:
                    self.signal_history[symbol] = []
                
                self.signal_history[symbol].append(signal)
                
                # Keep only last 100 signals per symbol
                if len(self.signal_history[symbol]) > 100:
                    self.signal_history[symbol] = self.signal_history[symbol][-100:]
        
        # Persist to storage
        await self.data_storage.store_signals(signals)
    
    async def _analyze_market_regime(self) -> Dict:
        """Analyze overall market conditions"""
        if not self.signal_history:
            return {'regime': 'unknown'}
        
        all_latest_signals = []
        for symbol_signals in self.signal_history.values():
            if symbol_signals:
                all_latest_signals.append(symbol_signals[-1])
        
        if not all_latest_signals:
            return {'regime': 'unknown'}
        
        # Aggregate momentum
        momentum_scores = [s.get('momentum', {}).get('momentum_7d', 0) for s in all_latest_signals]
        avg_momentum = np.mean(momentum_scores)
        momentum_std = np.std(momentum_scores)
        
        # Aggregate reversion probabilities
        reversion_scores = [s.get('reversion', {}).get('reversion_probability', 0) for s in all_latest_signals]
        avg_reversion = np.mean(reversion_scores)
        
        # Market regime classification
        if avg_momentum > 0.02 and momentum_std < 0.05:
            market_regime = 'bull_trending'
        elif avg_momentum < -0.02 and momentum_std < 0.05:
            market_regime = 'bear_trending'
        elif momentum_std > 0.1:
            market_regime = 'volatile_mixed'
        elif avg_reversion > 0.6:
            market_regime = 'mean_reverting'
        else:
            market_regime = 'consolidating'
        
        return {
            'regime': market_regime,
            'avg_momentum': avg_momentum,
            'momentum_dispersion': momentum_std,
            'reversion_bias': avg_reversion,
            'momentum_vs_reversion_ratio': self.momentum_bias
        }
    
    def stop_scanning(self):
        """Stop continuous scanning"""
        self.is_running = False


class DataPersistenceManager:
    """Manages data persistence for ML/RL training"""
    
    def __init__(self, storage_path: str = "scanner_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    async def store_signals(self, signals: List[Dict]):
        """Store signals to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.storage_path / f"signals_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(signals, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to store signals: {e}")
    
    async def load_historical_signals(self, days: int = 30) -> List[Dict]:
        """Load historical signals for training"""
        cutoff_date = datetime.now() - timedelta(days=days)
        signals = []
        
        for file_path in self.storage_path.glob("signals_*.json"):
            try:
                file_date = datetime.strptime(file_path.stem.split('_')[1], '%Y%m%d')
                if file_date >= cutoff_date:
                    with open(file_path, 'r') as f:
                        file_signals = json.load(f)
                        signals.extend(file_signals)
            except Exception as e:
                logger.debug(f"Error loading {file_path}: {e}")
        
        return signals


class MomentumScanner:
    """Enhanced Momentum Scanner with continuous scanning capabilities"""
    
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
        
        # Enhanced scanner integration
        self.continuous_scanner = ContinuousMarketScanner()
        self.enhanced_features = True
    
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
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

    def fetch_fear_greed_index(self):
        try:
            import requests
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

    async def scan_market(self, timeframe='daily'):
        """Enhanced market scanning with continuous features"""
        if self.enhanced_features:
            # Use continuous scanner for enhanced analysis
            signals = await self.continuous_scanner.get_live_signals(timeframe)
            if signals:
                # Convert enhanced signals to compatible format
                return self._convert_enhanced_signals(signals)
        
        # Fallback to original scanning logic
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
        self.time_history.append(datetime.now())
        avg_momentum = self.momentum_df['momentum_7d'].mean() if not self.momentum_df.empty else 0
        self.avg_momentum_history.append(avg_momentum)

        return self.momentum_df
    
    def _convert_enhanced_signals(self, enhanced_signals):
        """Convert enhanced signals to compatible DataFrame format"""
        results = []
        
        if isinstance(enhanced_signals, dict):
            for style, signals in enhanced_signals.items():
                for signal in signals:
                    momentum_data = signal.get('momentum', {})
                    reversion_data = signal.get('reversion', {})
                    
                    result = {
                        'symbol': signal.get('symbol', ''),
                        'momentum_7d': momentum_data.get('momentum_7d', 0),
                        'momentum_30d': momentum_data.get('momentum_30d', 0),
                        'rsi': momentum_data.get('rsi', 50),
                        'macd': momentum_data.get('macd', 0),
                        'price': momentum_data.get('price', 0),
                        'volume': momentum_data.get('volume', 0),
                        'average_volume_usd': momentum_data.get('volume', 0) * momentum_data.get('price', 0),
                        'signal': self._enhanced_to_classic_signal(signal),
                        'timestamp': signal.get('timestamp', datetime.now()),
                        'enhanced_momentum_score': momentum_data.get('enhanced_momentum_score', 0),
                        'reversion_probability': reversion_data.get('reversion_probability', 0),
                        'cluster_validated': momentum_data.get('cluster_validated', False)
                    }
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def _enhanced_to_classic_signal(self, enhanced_signal):
        """Convert enhanced signal to classic signal format"""
        momentum = enhanced_signal.get('momentum', {})
        reversion = enhanced_signal.get('reversion', {})
        
        mom_7d = momentum.get('momentum_7d', 0)
        mom_30d = momentum.get('momentum_30d', 0)
        rsi = momentum.get('rsi', 50)
        enhanced_score = momentum.get('enhanced_momentum_score', 0)
        reversion_prob = reversion.get('reversion_probability', 0)
        
        # Enhanced signal classification
        if reversion_prob > 0.7:
            return "Mean Reversion Candidate"
        elif enhanced_score > 0.05 and momentum.get('cluster_validated', False):
            return "Enhanced Momentum"
        else:
            return self.classify_signal(mom_7d, mom_30d, rsi, momentum.get('macd', 0))

    def execute_trades(self):
        positions = self.exchange.fetch_open_orders()
        current_positions = {order['symbol']: order for order in positions}

        for _, row in self.momentum_df.iterrows():
            symbol = row['symbol']
            signal = row['signal']
            price = row['price']

            if signal in ["Consistent Uptrend", "New Spike", "Moderate Uptrend", "Enhanced Momentum"]:
                if symbol not in current_positions:
                    # Place a buy order
                    self.exchange.create_limit_buy_order(symbol, 1, price)
                    print(f"Buy order placed for {symbol} at {price}")
            elif signal in ["Topping Out", "Lagging", "Potential Reversal", "Mean Reversion Candidate"]:
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

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
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
    import ccxt
    exchange = ccxt.kucoinfutures({'enableRateLimit': True})

    # Initialize the Enhanced MomentumScanner
    scanner = MomentumScanner(exchange)

    # Example usage of continuous scanning
    async def main():
        # Start continuous scanning in background
        scanner_task = asyncio.create_task(scanner.continuous_scanner.start_continuous_scanning())
        
        # Regular scanning every few minutes
        while True:
            results = await scanner.scan_market()
            print(f"Scanned {len(results)} symbols")
            
            # Get enhanced signals
            live_signals = await scanner.continuous_scanner.get_live_signals('day_trade')
            print(f"Live day trading signals: {len(live_signals)}")
            
            await asyncio.sleep(120)  # Wait 2 minutes
    
    # Run the scanner
    # asyncio.run(main())
