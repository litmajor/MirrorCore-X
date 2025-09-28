
"""
Enhanced Scanner Architecture for MirrorCore-X
Addresses continuity, multi-timeframe analysis, and mean reversion detection
"""

import asyncio
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import json
import logging
from pathlib import Path

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
                print(f"Error in price updates: {e}")
                await asyncio.sleep(10)
    
    async def _periodic_signal_generation(self):
        """Generate trading signals periodically"""
        while self.is_running:
            try:
                # Update OHLCV data for all timeframes
                await self._update_ohlcv_data()
                
                # Generate signals for each market and timeframe
                signals = await self._generate_multi_timeframe_signals()
                
                # Store signals with timestamp
                self._store_signals(signals)
                
                # Longer interval for signal generation
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                print(f"Error in signal generation: {e}")
                await asyncio.sleep(30)
    
    async def _update_ohlcv_data(self):
        """Update OHLCV data for all timeframes"""
        for symbol in self.market_data.keys():
            for style, timeframe in self.timeframes.items():
                try:
                    # Fetch recent candles
                    ohlcv = await self.exchange.fetch_ohlcv(
                        symbol, timeframe, limit=100
                    )
                    
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # Update stored data (keep last 500 candles)
                        self.market_data[symbol][timeframe] = df.tail(500)
                        
                except Exception as e:
                    print(f"Error updating {symbol} {timeframe}: {e}")
    
    async def _generate_multi_timeframe_signals(self):
        """Generate signals across multiple timeframes"""
        all_signals = {}
        
        for symbol in self.market_data.keys():
            symbol_signals = {}
            
            for style, timeframe in self.timeframes.items():
                df = self.market_data[symbol][timeframe]
                
                if len(df) < 50:  # Need enough data
                    continue
                
                # Generate signals for this timeframe
                signals = await self._analyze_timeframe(symbol, df, style)
                symbol_signals[style] = signals
            
            # Combine multi-timeframe signals
            all_signals[symbol] = self._combine_timeframe_signals(symbol_signals)
        
        return all_signals
    
    async def _analyze_timeframe(self, symbol: str, df: pd.DataFrame, style: str):
        """Analyze single timeframe for signals"""
        signals = {}
        
        # Current scanner logic (momentum)
        momentum_signals = self._detect_momentum(df)
        
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
        
        price_change = (latest['close'] - prev['close']) / prev['close']
        
        return {
            'price_change_pct': price_change,
            'direction': 'bullish' if price_change > 0 else 'bearish',
            'strength': 'strong' if abs(price_change) > 0.03 else 'moderate' if abs(price_change) > 0.01 else 'weak'
        }
    
    def _detect_enhanced_momentum(self, df: pd.DataFrame, cluster_signals: Dict) -> Dict:
        """Enhanced momentum detection with candle clustering validation"""
        if len(df) < 20:
            return {}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Basic momentum
        price_change = (latest['close'] - prev['close']) / prev['close']
        
        # Volume-weighted momentum
        volume_sma = df['volume'].rolling(20).mean()
        volume_ratio = latest['volume'] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
        
        # Cluster-validated momentum
        cluster_validation = cluster_signals.get('directional_cluster', False)
        cluster_strength = cluster_signals.get('cluster_strength', 0)
        
        # Enhanced momentum score
        momentum_score = abs(price_change) * volume_ratio
        if cluster_validation:
            momentum_score *= (1 + cluster_strength)  # Boost momentum if clusters support
        
        return {
            'price_change_pct': price_change,
            'volume_ratio': volume_ratio,
            'momentum_score': momentum_score,
            'is_momentum': momentum_score > 0.02,  # Dynamic threshold
            'cluster_validated': cluster_validation,
            'direction': 'bullish' if price_change > 0 else 'bearish',
            'strength': 'strong' if momentum_score > 0.05 else 'moderate' if momentum_score > 0.02 else 'weak'
        }
    
    def _detect_smart_mean_reversion(self, df: pd.DataFrame, style: str) -> Dict:
        """Smart mean reversion - reverse logic from momentum"""
        if len(df) < 20:
            return {}
        
        # Calculate indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['bb_upper'] = df['sma_20'] + (df['close'].rolling(20).std() * 2)
        df['bb_lower'] = df['sma_20'] - (df['close'].rolling(20).std() * 2)
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Volume analysis for reversion
        volume_sma = df['volume'].rolling(20).mean()
        recent_volume = df['volume'].tail(5).mean()
        
        latest = df.iloc[-1]
        
        # Reverse momentum logic - look for momentum exhaustion
        recent_moves = df['close'].pct_change().tail(5)
        consecutive_moves = self._count_consecutive_moves(recent_moves)
        momentum_exhaustion = consecutive_moves >= 4  # 4+ moves in same direction
        
        # Volume exhaustion (high volume but decreasing)
        volume_trend = df['volume'].tail(3).pct_change().mean()
        volume_exhaustion = recent_volume > volume_sma.iloc[-1] * 1.5 and volume_trend < -0.1
        
        # Style-specific reversion thresholds
        rsi_overbought = 75 if style == 'scalp' else 70
        rsi_oversold = 25 if style == 'scalp' else 30
        
        # Mean reversion signals
        signals = {
            'is_overbought': latest['rsi'] > rsi_overbought and latest['close'] > latest['bb_upper'],
            'is_oversold': latest['rsi'] < rsi_oversold and latest['close'] < latest['bb_lower'],
            'momentum_exhaustion': momentum_exhaustion,
            'volume_exhaustion': volume_exhaustion,
            'distance_from_mean': (latest['close'] - latest['sma_20']) / latest['sma_20'],
            'bb_squeeze': (latest['bb_upper'] - latest['bb_lower']) / latest['sma_20'] < 0.1
        }
        
        # Enhanced reversion detection - your insight about top movers
        recent_gain = (latest['close'] - df.iloc[-10]['close']) / df.iloc[-10]['close'] if len(df) >= 10 else 0
        signals['excessive_gain'] = recent_gain > 0.20  # 20% gain in 10 periods
        signals['excessive_loss'] = recent_gain < -0.15  # 15% loss in 10 periods
        
        # Combine factors for reversion probability
        reversion_score = 0
        if signals['is_overbought']: reversion_score += 0.3
        if signals['is_oversold']: reversion_score += 0.3
        if signals['momentum_exhaustion']: reversion_score += 0.2
        if signals['volume_exhaustion']: reversion_score += 0.2
        if signals['excessive_gain']: reversion_score += 0.3
        if signals['excessive_loss']: reversion_score += 0.3
        
        signals['reversion_probability'] = min(reversion_score, 1.0)
        signals['reversion_candidate'] = reversion_score > 0.6
        signals['reversion_direction'] = 'down' if signals['is_overbought'] or signals['excessive_gain'] else 'up'
        
        return signals
    
    def _detect_candle_clustering(self, df: pd.DataFrame) -> Dict:
        """Detect candle clustering for trend formation and momentum validation"""
        if len(df) < self.cluster_lookback:
            return {}
        
        # Analyze recent candles
        recent_candles = df.tail(self.cluster_lookback)
        
        # Calculate volume statistics
        volume_sma = df['volume'].rolling(50).mean().iloc[-1]
        high_volume_threshold = volume_sma * self.volume_threshold_multiplier
        
        # Identify high-volume candles
        high_volume_candles = recent_candles[recent_candles['volume'] > high_volume_threshold]
        
        if len(high_volume_candles) == 0:
            return {'cluster_detected': False}
        
        # Analyze clustering patterns
        cluster_analysis = self._analyze_candle_clusters(high_volume_candles, recent_candles)
        
        return cluster_analysis
    
    def _analyze_candle_clusters(self, high_vol_candles: pd.DataFrame, all_candles: pd.DataFrame) -> Dict:
        """Analyze candle clustering patterns"""
        
        # Directional consistency analysis
        bullish_clusters = 0
        bearish_clusters = 0
        
        for idx, candle in high_vol_candles.iterrows():
            body_size = abs(candle['close'] - candle['open'])
            is_bullish = candle['close'] > candle['open']
            
            if body_size > (candle['high'] - candle['low']) * 0.6:  # Strong body
                if is_bullish:
                    bullish_clusters += 1
                else:
                    bearish_clusters += 1
        
        total_clusters = len(high_vol_candles)
        
        # Clustering strength
        if total_clusters == 0:
            return {'cluster_detected': False}
        
        directional_ratio = max(bullish_clusters, bearish_clusters) / total_clusters
        
        # Temporal clustering (are high-volume candles close together?)
        cluster_timestamps = high_vol_candles.index.tolist()
        temporal_gaps = [cluster_timestamps[i+1] - cluster_timestamps[i] for i in range(len(cluster_timestamps)-1)]
        avg_gap = np.mean(temporal_gaps) if temporal_gaps else 0
        
        # Follow-through analysis (do subsequent candles follow cluster direction?)
        latest_cluster_direction = 'bullish' if bullish_clusters > bearish_clusters else 'bearish'
        follow_through = self._analyze_follow_through(all_candles, latest_cluster_direction)
        
        return {
            'cluster_detected': True,
            'cluster_count': total_clusters,
            'bullish_clusters': bullish_clusters,
            'bearish_clusters': bearish_clusters,
            'directional_ratio': directional_ratio,
            'dominant_direction': latest_cluster_direction,
            'directional_cluster': directional_ratio > 0.7,  # 70% same direction
            'cluster_strength': directional_ratio,
            'temporal_clustering': avg_gap < 3,  # Within 3 candles
            'follow_through_strength': follow_through,
            'trend_formation_signal': directional_ratio > 0.7 and follow_through > 0.5
        }
    
    def _analyze_follow_through(self, candles: pd.DataFrame, cluster_direction: str) -> float:
        """Analyze if candles after clusters follow the cluster direction"""
        if len(candles) < 5:
            return 0
        
        recent_moves = candles['close'].pct_change().tail(3)
        
        if cluster_direction == 'bullish':
            follow_through = sum(1 for move in recent_moves if move > 0) / len(recent_moves)
        else:
            follow_through = sum(1 for move in recent_moves if move < 0) / len(recent_moves)
        
        return follow_through
    
    def _count_consecutive_moves(self, price_changes: pd.Series) -> int:
        """Count consecutive moves in the same direction"""
        if len(price_changes) == 0:
            return 0
        
        consecutive = 1
        last_direction = 1 if price_changes.iloc[-1] > 0 else -1
        
        for i in range(len(price_changes) - 2, -1, -1):
            current_direction = 1 if price_changes.iloc[i] > 0 else -1
            if current_direction == last_direction:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """Market regime detection"""
        if len(df) < 50:
            return {}
        
        # Volatility regime
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        current_vol = volatility.iloc[-1]
        avg_vol = volatility.mean()
        
        # Trend regime
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_30'] = df['close'].ewm(span=30).mean()
        
        trend_strength = (df['ema_10'].iloc[-1] - df['ema_30'].iloc[-1]) / df['ema_30'].iloc[-1]
        
        return {
            'volatility_regime': 'high' if current_vol > avg_vol * 1.5 else 'normal',
            'trend_regime': 'strong_up' if trend_strength > 0.02 else 'strong_down' if trend_strength < -0.02 else 'sideways',
            'volatility_ratio': current_vol / avg_vol,
            'trend_strength': trend_strength
        }
    
    def _scalp_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis specific to scalping timeframe"""
        # Volume profile analysis
        volume_sma = df['volume'].rolling(20).mean()
        current_volume = df['volume'].iloc[-1]
        
        # Price action patterns
        latest_candles = df.tail(3)
        is_doji = abs(latest_candles['close'].iloc[-1] - latest_candles['open'].iloc[-1]) < (latest_candles['high'].iloc[-1] - latest_candles['low'].iloc[-1]) * 0.1
        
        return {
            'volume_spike': current_volume > volume_sma.iloc[-1] * 2,
            'is_doji': is_doji,
            'quick_scalp_opportunity': current_volume > volume_sma.iloc[-1] * 1.5 and not is_doji
        }
    
    def _day_trade_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis specific to day trading timeframe (4h candles)"""
        if len(df) < 24:  # Need at least 4 days of data
            return {}
        
        # Session-based analysis for day trading
        # Look for session momentum and reversal patterns
        
        # Daily volatility analysis
        daily_ranges = (df['high'] - df['low']) / df['close']
        avg_daily_range = daily_ranges.rolling(10).mean()
        current_range = daily_ranges.iloc[-1]
        
        # Volume pattern for day trading
        volume_profile = df['volume'].rolling(6).mean()  # 24h rolling average
        current_session_volume = df['volume'].tail(2).mean()  # Last 8 hours
        
        # Price action for day trading setups
        body_sizes = abs(df['close'] - df['open']) / df['close']
        avg_body_size = body_sizes.rolling(10).mean().iloc[-1]
        current_body = body_sizes.iloc[-1]
        
        # Day trading specific signals
        return {
            'high_volatility_session': current_range > avg_daily_range.iloc[-1] * 1.3,
            'volume_expansion': current_session_volume > volume_profile.iloc[-1] * 1.2,
            'strong_directional_move': current_body > avg_body_size * 1.5,
            'day_trade_setup': current_range > avg_daily_range.iloc[-1] * 1.2 and current_session_volume > volume_profile.iloc[-1] * 1.1,
            'session_momentum': 'building' if current_body > avg_body_size else 'fading'
        }
    
    def _position_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis for position trading (7d style)"""
        # Weekly trend analysis
        weekly_change = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] if len(df) >= 7 else 0
        
        return {
            'weekly_trend': 'bullish' if weekly_change > 0.1 else 'bearish' if weekly_change < -0.1 else 'neutral',
            'position_worthy': abs(weekly_change) > 0.15
        }
    
    def _combine_timeframe_signals(self, timeframe_signals: Dict) -> Dict:
        """Combine signals from multiple timeframes"""
        combined = {
            'multi_timeframe_consensus': False,
            'conflicting_signals': False,
            'dominant_timeframe': None,
            'signals_by_timeframe': timeframe_signals
        }
        
        # Check for consensus across timeframes
        bullish_count = sum(1 for tf_signals in timeframe_signals.values() 
                          if tf_signals.get('momentum', {}).get('direction') == 'bullish')
        
        bearish_count = sum(1 for tf_signals in timeframe_signals.values() 
                          if tf_signals.get('momentum', {}).get('direction') == 'bearish')
        
        if bullish_count > bearish_count:
            combined['multi_timeframe_consensus'] = True
            combined['dominant_direction'] = 'bullish'
        elif bearish_count > bullish_count:
            combined['multi_timeframe_consensus'] = True
            combined['dominant_direction'] = 'bearish'
        else:
            combined['conflicting_signals'] = True
        
        return combined
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def _process_tick(self, tick_data: Dict):
        """Process real-time tick data"""
        # Update market state with latest prices
        for symbol, ticker in tick_data['tickers'].items():
            if symbol in self.market_state:
                prev_price = self.market_state[symbol].get('price', ticker['last'])
                price_change = ticker['last'] - prev_price
                
                self.market_state[symbol].update({
                    'price': ticker['last'],
                    'volume': ticker['baseVolume'],
                    'price_change': price_change,
                    'tick_timestamp': tick_data['timestamp']
                })
    
    async def _market_state_analysis(self):
        """Analyze overall market state"""
        while self.is_running:
            try:
                # Market-wide analysis
                market_analysis = self._analyze_market_conditions()
                
                # Update global market state
                self.market_state['global'] = market_analysis
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                print(f"Error in market state analysis: {e}")
                await asyncio.sleep(60)
    
    def _analyze_market_conditions(self) -> Dict:
        """Analyze overall market conditions"""
        # Implement market-wide sentiment analysis
        return {
            'market_sentiment': 'neutral',  # bullish/bearish/neutral
            'volatility_level': 'normal',   # high/normal/low
            'correlation_breakdown': False,  # Are correlations breaking down?
            'risk_on_off': 'risk_neutral'   # risk_on/risk_off/risk_neutral
        }
    
    def _store_signals(self, signals: Dict):
        """Store signals with timestamp for historical analysis"""
        timestamp = datetime.now()
        
        for symbol, signal_data in signals.items():
            # Store in memory for immediate access
            self.signal_history[symbol].append({
                'timestamp': timestamp,
                'signals': signal_data
            })
            
            # Keep only last 1000 signals per symbol in memory
            self.signal_history[symbol] = self.signal_history[symbol][-1000:]
            
            # Persist to disk for ML/RL training
            asyncio.create_task(self.data_storage.persist_signals(symbol, signal_data, timestamp))
    
    async def get_training_data(self, symbol: str, days: int = 30) -> Dict:
        """Get historical data for ML/RL training"""
        return await self.data_storage.get_training_dataset(symbol, days)
    
    # Methods to interface with your SyncBus
    async def get_latest_signals(self, symbol: Optional[str] = None) -> Dict:
        """Get latest signals for SyncBus"""
        if symbol:
            return self.signal_history.get(symbol, [])[-1] if self.signal_history.get(symbol) else {}
        
        # Return latest signals for all symbols
        latest_signals = {}
        for sym, history in self.signal_history.items():
            if history:
                latest_signals[sym] = history[-1]
        
        return latest_signals
    
    async def get_market_tick_data(self) -> Dict:
        """Get current market tick data for SyncBus"""
        return self.market_state
    
    async def get_multi_timeframe_view(self, symbol: str) -> Dict:
        """Get comprehensive multi-timeframe view of a symbol"""
        if symbol not in self.signal_history or not self.signal_history[symbol]:
            return {}
        
        latest_signals = self.signal_history[symbol][-1]
        return {
            'symbol': symbol,
            'current_price': self.market_state.get(symbol, {}).get('price'),
            'timeframe_analysis': latest_signals.get('signals', {}),
            'historical_ohlcv': {
                timeframe: self.market_data[symbol][timeframe].tail(20).to_dict('records')
                for timeframe in self.timeframes.values()
                if symbol in self.market_data and timeframe in self.market_data[symbol]
            }
        }
    
    def stop_scanning(self):
        """Stop continuous scanning"""
        self.is_running = False


class DataPersistenceManager:
    """Manages data persistence for ML/RL training"""
    
    def __init__(self, data_dir: str = "./training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Separate folders for different data types
        (self.data_dir / "signals").mkdir(exist_ok=True)
        (self.data_dir / "ohlcv").mkdir(exist_ok=True)
        (self.data_dir / "clustering").mkdir(exist_ok=True)
        
    async def persist_signals(self, symbol: str, signal_data: Dict, timestamp: datetime):
        """Persist signal data for training"""
        try:
            # Create daily files for efficient storage
            date_str = timestamp.strftime("%Y%m%d")
            file_path = self.data_dir / "signals" / f"{symbol}_{date_str}.json"
            
            # Load existing data or create new
            if file_path.exists():
                with open(file_path, 'r') as f:
                    daily_data = json.load(f)
            else:
                daily_data = []
            
            # Append new signal
            daily_data.append({
                'timestamp': timestamp.isoformat(),
                'signals': signal_data
            })
            
            # Save back to file
            with open(file_path, 'w') as f:
                json.dump(daily_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting signals for {symbol}: {e}")
    
    async def persist_ohlcv(self, symbol: str, timeframe: str, ohlcv_data: pd.DataFrame):
        """Persist OHLCV data for training"""
        try:
            file_path = self.data_dir / "ohlcv" / f"{symbol}_{timeframe}.parquet"
            
            if file_path.exists():
                # Append to existing data
                existing_data = pd.read_parquet(file_path)
                combined_data = pd.concat([existing_data, ohlcv_data]).drop_duplicates(subset=['timestamp'])
            else:
                combined_data = ohlcv_data
            
            # Keep only last 10000 candles per file
            combined_data = combined_data.tail(10000)
            combined_data.to_parquet(file_path)
            
        except Exception as e:
            logger.error(f"Error persisting OHLCV for {symbol}_{timeframe}: {e}")
    
    async def get_training_dataset(self, symbol: str, days: int = 30) -> Dict:
        """Get comprehensive training dataset"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        training_data = {
            'signals': [],
            'ohlcv': {},
            'metadata': {
                'symbol': symbol,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days
            }
        }
        
        # Load signal data
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            signal_file = self.data_dir / "signals" / f"{symbol}_{date_str}.json"
            
            if signal_file.exists():
                with open(signal_file, 'r') as f:
                    daily_signals = json.load(f)
                    training_data['signals'].extend(daily_signals)
            
            current_date += timedelta(days=1)
        
        # Load OHLCV data for all timeframes
        for timeframe in ['5m', '1h', '4h', '1d']:
            ohlcv_file = self.data_dir / "ohlcv" / f"{symbol}_{timeframe}.parquet"
            if ohlcv_file.exists():
                df = pd.read_parquet(ohlcv_file)
                # Filter by date range
                df_filtered = df[
                    (pd.to_datetime(df['timestamp'], unit='ms') >= start_date) &
                    (pd.to_datetime(df['timestamp'], unit='ms') <= end_date)
                ]
                training_data['ohlcv'][timeframe] = df_filtered.to_dict('records')
        
        return training_data


# Usage Example
async def main():
    scanner = ContinuousMarketScanner('binance')
    
    # Start continuous scanning in background
    scan_task = asyncio.create_task(scanner.start_continuous_scanning())
    
    # Your agents can now access continuous data
    while True:
        # Get latest signals (for your SyncBus)
        signals = await scanner.get_latest_signals()
        tick_data = await scanner.get_market_tick_data()
        
        # Feed to your SyncBus/agents
        print(f"Latest signals: {len(signals)} markets")
        print(f"Market state: {len(tick_data)} tickers")
        
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
