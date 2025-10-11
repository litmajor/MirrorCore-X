import numpy as np
import math

# Robust utility for safe JSON serialization of numpy/pandas types, NaN/inf, and nested structures
def clean_for_json(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None  # replace NaN/inf with null
        return float(obj)
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(clean_for_json(x) for x in obj)
    # Optionally handle pandas types
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return clean_for_json(obj.to_dict())
        if isinstance(obj, pd.DataFrame):
            return clean_for_json(obj.to_dict(orient="records"))
    except ImportError:
        pass
    return obj
"""
Enhanced Scanner Architecture for MirrorCore-X - FIXED
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousMarketScanner:
    """
    Enhanced scanner that provides continuous data streams to cognitive agents
    Supports multiple timeframes and mean reversion detection
    """
    
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
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
        self.active_symbols = []
        
        # Scanner state
        self.is_running = False
        self.is_initialized = False
        self.scan_interval = 90  # 90 seconds - based on your 1-2min scan time
        self.momentum_bias = 0.6  # 60% momentum, 40% mean reversion balance
        
    async def start_continuous_scanning(self):
        """Start continuous market monitoring"""
        self.is_running = True
        
        logger.info("Starting market scanner initialization...")
        
        # Initialize data structures
        try:
            await self._initialize_markets()
            logger.info(f"Initialized {len(self.active_symbols)} markets")
        except Exception as e:
            logger.error(f"Failed to initialize markets: {e}")
            self.is_running = False
            return
        
        # Start parallel scanning tasks
        tasks = [
            self._continuous_price_updates(),
            self._periodic_signal_generation(),
            self._market_state_analysis()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in scanning tasks: {e}")
            self.is_running = False
    
    async def _initialize_markets(self):
        """Initialize market data storage"""
        try:
            loop = asyncio.get_event_loop()
            # Load markets (blocking)
            markets = await loop.run_in_executor(None, self.exchange.load_markets)
            # Filter for active USDT spot markets
            usdt_markets = [
                symbol for symbol, market in markets.items()
                if '/USDT' in symbol
                and market.get('active', True)
                and market.get('spot', True)
            ]
            # Get top markets by volume
            logger.info("Fetching tickers to rank markets...")
            tickers = await loop.run_in_executor(None, self.exchange.fetch_tickers)
            # Sort by 24h volume
            sorted_markets = sorted(
                [s for s in usdt_markets if s in tickers],
                key=lambda x: tickers[x].get('quoteVolume', 0) or 0,
                reverse=True
            )
            # Take top 30 for performance
            self.active_symbols = sorted_markets[:30]
            logger.info(f"Selected markets: {self.active_symbols[:5]}... (and {len(self.active_symbols)-5} more)")
            # Initialize data structures
            for symbol in self.active_symbols:
                self.market_data[symbol] = {
                    timeframe: pd.DataFrame()
                    for timeframe in self.timeframes.values()
                }
                self.signal_history[symbol] = []
                self.market_state[symbol] = {
                    'price': tickers[symbol].get('last', 0),
                    'volume': tickers[symbol].get('baseVolume', 0),
                    'price_change': 0,
                    'tick_timestamp': datetime.now()
                }
            self.is_initialized = True
            logger.info("Market initialization complete")
        except Exception as e:
            logger.error(f"Error in _initialize_markets: {e}")
            raise
    
    async def _continuous_price_updates(self):
        """Continuous price monitoring (market ticks)"""
        logger.info("Starting continuous price updates...")
        loop = asyncio.get_event_loop()
        while self.is_running:
            if not self.is_initialized:
                await asyncio.sleep(5)
                continue
            try:
                # Get ticker updates for monitored markets only (blocking)
                tickers = await loop.run_in_executor(None, lambda: self.exchange.fetch_tickers(self.active_symbols))
                tick_data = {
                    'timestamp': datetime.now(),
                    'tickers': tickers
                }
                # Process tick data
                await self._process_tick(tick_data)
                logger.debug(f"Updated {len(tickers)} tickers")
                # Short interval for real-time updates
                await asyncio.sleep(5)  # 5-second ticks
            except Exception as e:
                logger.error(f"Error in price updates: {e}")
                await asyncio.sleep(10)
    
    async def _periodic_signal_generation(self):
        """Generate trading signals periodically"""
        logger.info("Starting periodic signal generation...")
        
        # Wait for initialization
        while not self.is_initialized:
            await asyncio.sleep(1)
        
        # Initial OHLCV load
        logger.info("Loading initial OHLCV data...")
        await self._update_ohlcv_data()
        
        while self.is_running:
            try:
                # Update OHLCV data for all timeframes
                await self._update_ohlcv_data()
                
                # Generate signals for each market and timeframe
                signals = await self._generate_multi_timeframe_signals()
                
                # Store signals with timestamp
                self._store_signals(signals)
                
                logger.info(f"Generated signals for {len(signals)} markets")
                
                # Longer interval for signal generation
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in signal generation: {e}")
                await asyncio.sleep(30)
    
    async def _update_ohlcv_data(self):
        """Update OHLCV data for all timeframes"""
        logger.debug("Updating OHLCV data...")
        loop = asyncio.get_event_loop()
        for symbol in self.active_symbols:
            for style, timeframe in self.timeframes.items():
                try:
                    # Fetch recent candles (blocking)
                    ohlcv = await loop.run_in_executor(
                        None, lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                    )
                    if ohlcv:
                        df = pd.DataFrame(
                            ohlcv,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                        # Update stored data (keep last 500 candles)
                        self.market_data[symbol][timeframe] = df.tail(500)
                        logger.debug(f"Updated {symbol} {timeframe}: {len(df)} candles")
                    # Rate limiting
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error updating {symbol} {timeframe}: {e}")
                    await asyncio.sleep(1)
    
    async def _generate_multi_timeframe_signals(self):
        """Generate signals across multiple timeframes"""
        all_signals = {}
        
        for symbol in self.active_symbols:
            symbol_signals = {}
            
            for style, timeframe in self.timeframes.items():
                df = self.market_data[symbol].get(timeframe)
                
                if df is None or len(df) < 50:  # Need enough data
                    continue
                
                # Generate signals for this timeframe
                signals = await self._analyze_timeframe(symbol, df, style)
                symbol_signals[style] = signals
            
            if symbol_signals:  # Only add if we have signals
                # Combine multi-timeframe signals
                all_signals[symbol] = self._combine_timeframe_signals(symbol_signals)
        
        return all_signals
    
    async def _analyze_timeframe(self, symbol: str, df: pd.DataFrame, style: str):
        """Analyze single timeframe for signals"""
        signals = {}
        
        # Current scanner logic (momentum)
        momentum_signals = self._detect_momentum(df)
        
        # Mean reversion detection with reverse momentum logic
        reversion_signals = self._detect_smart_mean_reversion(df, style)
        
        # Candle clustering analysis
        cluster_signals = self._detect_candle_clustering(df)
        
        # Momentum with clustering validation
        enhanced_momentum = self._detect_enhanced_momentum(df, cluster_signals)
        
        # Market regime detection
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
            'price_change_pct': float(price_change),
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
            'price_change_pct': float(price_change),
            'volume_ratio': float(volume_ratio),
            'momentum_score': float(momentum_score),
            'is_momentum': momentum_score > 0.02,
            'cluster_validated': cluster_validation,
            'direction': 'bullish' if price_change > 0 else 'bearish',
            'strength': 'strong' if momentum_score > 0.05 else 'moderate' if momentum_score > 0.02 else 'weak'
        }
    
    def _detect_smart_mean_reversion(self, df: pd.DataFrame, style: str) -> Dict:
        """Smart mean reversion - reverse logic from momentum"""
        if len(df) < 20:
            return {}
        
        # Calculate indicators
        df = df.copy()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Volume analysis for reversion
        volume_sma = df['volume'].rolling(20).mean()
        recent_volume = df['volume'].tail(5).mean()
        
        latest = df.iloc[-1]
        
        # Reverse momentum logic - look for momentum exhaustion
        recent_moves = df['close'].pct_change().tail(5)
        consecutive_moves = self._count_consecutive_moves(recent_moves)
        momentum_exhaustion = consecutive_moves >= 4
        
        # Volume exhaustion
        volume_trend = df['volume'].tail(3).pct_change().mean()
        volume_exhaustion = recent_volume > volume_sma.iloc[-1] * 1.5 and volume_trend < -0.1
        
        # Style-specific reversion thresholds
        rsi_overbought = 75 if style == 'scalp' else 70
        rsi_oversold = 25 if style == 'scalp' else 30
        
        # Mean reversion signals
        signals = {
            'is_overbought': bool(latest['rsi'] > rsi_overbought and latest['close'] > latest['bb_upper']),
            'is_oversold': bool(latest['rsi'] < rsi_oversold and latest['close'] < latest['bb_lower']),
            'momentum_exhaustion': momentum_exhaustion,
            'volume_exhaustion': volume_exhaustion,
            'distance_from_mean': float((latest['close'] - latest['sma_20']) / latest['sma_20']),
            'bb_squeeze': bool((latest['bb_upper'] - latest['bb_lower']) / latest['sma_20'] < 0.1)
        }
        
        # Enhanced reversion detection
        recent_gain = (latest['close'] - df.iloc[-10]['close']) / df.iloc[-10]['close'] if len(df) >= 10 else 0
        signals['excessive_gain'] = recent_gain > 0.20
        signals['excessive_loss'] = recent_gain < -0.15
        
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
        """Detect candle clustering for trend formation"""
        if len(df) < self.cluster_lookback:
            return {'cluster_detected': False}
        
        # Analyze recent candles
        recent_candles = df.tail(self.cluster_lookback)
        
        # Calculate volume statistics
        volume_sma = df['volume'].rolling(50).mean().iloc[-1]
        if volume_sma == 0:
            return {'cluster_detected': False}
            
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
        
        bullish_clusters = 0
        bearish_clusters = 0
        
        for idx, candle in high_vol_candles.iterrows():
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            is_bullish = candle['close'] > candle['open']
            
            if candle_range > 0 and body_size > candle_range * 0.6:
                if is_bullish:
                    bullish_clusters += 1
                else:
                    bearish_clusters += 1
        
        total_clusters = len(high_vol_candles)
        
        if total_clusters == 0:
            return {'cluster_detected': False}
        
        directional_ratio = max(bullish_clusters, bearish_clusters) / total_clusters
        latest_cluster_direction = 'bullish' if bullish_clusters > bearish_clusters else 'bearish'
        follow_through = self._analyze_follow_through(all_candles, latest_cluster_direction)
        
        return {
            'cluster_detected': True,
            'cluster_count': int(total_clusters),
            'bullish_clusters': int(bullish_clusters),
            'bearish_clusters': int(bearish_clusters),
            'directional_ratio': float(directional_ratio),
            'dominant_direction': latest_cluster_direction,
            'directional_cluster': directional_ratio > 0.7,
            'cluster_strength': float(directional_ratio),
            'follow_through_strength': float(follow_through),
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
        df = df.copy()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_30'] = df['close'].ewm(span=30).mean()
        
        trend_strength = (df['ema_10'].iloc[-1] - df['ema_30'].iloc[-1]) / df['ema_30'].iloc[-1]
        
        return {
            'volatility_regime': 'high' if current_vol > avg_vol * 1.5 else 'normal',
            'trend_regime': 'strong_up' if trend_strength > 0.02 else 'strong_down' if trend_strength < -0.02 else 'sideways',
            'volatility_ratio': float(current_vol / avg_vol) if avg_vol > 0 else 1.0,
            'trend_strength': float(trend_strength)
        }
    
    def _scalp_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis specific to scalping timeframe"""
        volume_sma = df['volume'].rolling(20).mean()
        current_volume = df['volume'].iloc[-1]
        
        latest_candles = df.tail(3)
        candle_range = latest_candles['high'].iloc[-1] - latest_candles['low'].iloc[-1]
        body_size = abs(latest_candles['close'].iloc[-1] - latest_candles['open'].iloc[-1])
        is_doji = body_size < candle_range * 0.1 if candle_range > 0 else False
        
        return {
            'volume_spike': bool(current_volume > volume_sma.iloc[-1] * 2),
            'is_doji': is_doji,
            'quick_scalp_opportunity': bool(current_volume > volume_sma.iloc[-1] * 1.5 and not is_doji)
        }
    
    def _day_trade_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis specific to day trading timeframe (4h candles)"""
        if len(df) < 24:
            return {}
        
        daily_ranges = (df['high'] - df['low']) / df['close']
        avg_daily_range = daily_ranges.rolling(10).mean()
        current_range = daily_ranges.iloc[-1]
        
        volume_profile = df['volume'].rolling(6).mean()
        current_session_volume = df['volume'].tail(2).mean()
        
        body_sizes = abs(df['close'] - df['open']) / df['close']
        avg_body_size = body_sizes.rolling(10).mean().iloc[-1]
        current_body = body_sizes.iloc[-1]
        
        return {
            'high_volatility_session': bool(current_range > avg_daily_range.iloc[-1] * 1.3),
            'volume_expansion': bool(current_session_volume > volume_profile.iloc[-1] * 1.2),
            'strong_directional_move': bool(current_body > avg_body_size * 1.5),
            'day_trade_setup': bool(current_range > avg_daily_range.iloc[-1] * 1.2 and current_session_volume > volume_profile.iloc[-1] * 1.1),
            'session_momentum': 'building' if current_body > avg_body_size else 'fading'
        }
    
    def _position_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analysis for position trading (7d style)"""
        weekly_change = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] if len(df) >= 7 else 0
        
        return {
            'weekly_trend': 'bullish' if weekly_change > 0.1 else 'bearish' if weekly_change < -0.1 else 'neutral',
            'position_worthy': bool(abs(weekly_change) > 0.15),
            'weekly_change_pct': float(weekly_change)
        }
    
    def _combine_timeframe_signals(self, timeframe_signals: Dict) -> Dict:
        """Combine signals from multiple timeframes"""
        combined = {
            'multi_timeframe_consensus': False,
            'conflicting_signals': False,
            'dominant_timeframe': None,
            'signals_by_timeframe': timeframe_signals
        }
        
        # Check for consensus
        bullish_count = sum(
            1 for tf_signals in timeframe_signals.values() 
            if tf_signals.get('momentum', {}).get('direction') == 'bullish'
        )
        
        bearish_count = sum(
            1 for tf_signals in timeframe_signals.values() 
            if tf_signals.get('momentum', {}).get('direction') == 'bearish'
        )
        
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
        delta = prices.diff().astype(float)
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        # Avoid division by zero
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def _process_tick(self, tick_data: Dict):
        """Process real-time tick data"""
        for symbol, ticker in tick_data['tickers'].items():
            if symbol in self.market_state:
                prev_price = self.market_state[symbol].get('price', ticker.get('last', 0))
                current_price = ticker.get('last', 0)
                price_change = current_price - prev_price
                
                self.market_state[symbol].update({
                    'price': current_price,
                    'volume': ticker.get('baseVolume', 0),
                    'price_change': price_change,
                    'tick_timestamp': tick_data['timestamp']
                })
    
    async def _market_state_analysis(self):
        """Analyze overall market state"""
        logger.info("Starting market state analysis...")
        
        while not self.is_initialized:
            await asyncio.sleep(1)
        
        while self.is_running:
            try:
                market_analysis = self._analyze_market_conditions()
                self.market_state['global'] = market_analysis
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in market state analysis: {e}")
                await asyncio.sleep(60)
    
    def _analyze_market_conditions(self) -> Dict:
        """Analyze overall market conditions"""
        # Simple market-wide analysis
        bullish_count = 0
        bearish_count = 0
        
        for symbol in self.active_symbols:
            if symbol in self.market_state:
                price_change = self.market_state[symbol].get('price_change', 0)
                if price_change > 0:
                    bullish_count += 1
                elif price_change < 0:
                    bearish_count += 1
        
        total = bullish_count + bearish_count
        bullish_ratio = bullish_count / total if total > 0 else 0.5
        
        return {
            'market_sentiment': 'bullish' if bullish_ratio > 0.6 else 'bearish' if bullish_ratio < 0.4 else 'neutral',
            'volatility_level': 'normal',
            'correlation_breakdown': False,
            'risk_on_off': 'risk_neutral',
            'bullish_markets': bullish_count,
            'bearish_markets': bearish_count
        }
    
    def _store_signals(self, signals: Dict):
        """Store signals with timestamp"""
        timestamp = datetime.now()
        
        for symbol, signal_data in signals.items():
            self.signal_history[symbol].append({
                'timestamp': timestamp,
                'signals': signal_data
            })
            
            # Keep only last 1000 signals
            self.signal_history[symbol] = self.signal_history[symbol][-1000:]
            
            # Persist to disk
            asyncio.create_task(
                self.data_storage.persist_signals(symbol, signal_data, timestamp)
            )
    
    async def get_training_data(self, symbol: str, days: int = 30) -> Dict:
        """Get historical data for ML/RL training"""
        return await self.data_storage.get_training_dataset(symbol, days)
    
    async def get_latest_signals(self, symbol: Optional[str] = None) -> Dict:
        """Get latest signals for SyncBus"""
        if symbol:
            return self.signal_history.get(symbol, [])[-1] if self.signal_history.get(symbol) else {}
        
        latest_signals = {}
        for sym, history in self.signal_history.items():
            if history:
                latest_signals[sym] = history[-1]
        
        return latest_signals
    
    async def get_market_tick_data(self) -> Dict:
        """Get current market tick data"""
        return self.market_state
    
    async def get_multi_timeframe_view(self, symbol: str) -> Dict:
        """Get comprehensive multi-timeframe view"""
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
        logger.info("Scanner stopped")


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
            # Sanitize symbol for filename
            safe_symbol = symbol.replace('/', '_')
            file_path = self.data_dir / "signals" / f"{safe_symbol}_{date_str}.json"
            
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
            
            # Clean all data before saving
            cleaned_data = clean_for_json(daily_data)
            # Save back to file
            with open(file_path, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting signals for {symbol}: {e}")
    
    async def persist_ohlcv(self, symbol: str, timeframe: str, ohlcv_data: pd.DataFrame):
        """Persist OHLCV data for training"""
        try:
            safe_symbol = symbol.replace('/', '_')
            file_path = self.data_dir / "ohlcv" / f"{safe_symbol}_{timeframe}.parquet"
            
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
        safe_symbol = symbol.replace('/', '_')
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            signal_file = self.data_dir / "signals" / f"{safe_symbol}_{date_str}.json"
            
            if signal_file.exists():
                with open(signal_file, 'r') as f:
                    daily_signals = json.load(f)
                    training_data['signals'].extend(daily_signals)
            
            current_date += timedelta(days=1)
        
        # Load OHLCV data for all timeframes
        for timeframe in ['5m', '1h', '4h', '1d']:
            ohlcv_file = self.data_dir / "ohlcv" / f"{safe_symbol}_{timeframe}.parquet"
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
    """Main execution function"""
    scanner = ContinuousMarketScanner('binance')
    
    try:
        # Start continuous scanning in background
        scan_task = asyncio.create_task(scanner.start_continuous_scanning())
        
        # Monitor scanner output
        await asyncio.sleep(10)  # Wait for initialization
        
        logger.info("Scanner initialized, entering monitoring loop...")
        
        iteration = 0
        while True:
            # Get latest signals
            signals = await scanner.get_latest_signals()
            tick_data = await scanner.get_market_tick_data()
            
            # Log status
            signal_count = len(signals)
            ticker_count = len([k for k in tick_data.keys() if k != 'global'])
            
            logger.info(f"Iteration {iteration}: {signal_count} markets with signals, {ticker_count} tickers tracked")
            
            # Display sample signal if available
            if signals:
                sample_symbol = list(signals.keys())[0]
                sample_signal = signals[sample_symbol]
                logger.info(f"Sample signal for {sample_symbol}:")
                logger.info(f"  Timestamp: {sample_signal.get('timestamp')}")
                
                # Check for multi-timeframe consensus
                if sample_signal.get('signals', {}).get('multi_timeframe_consensus'):
                    direction = sample_signal['signals'].get('dominant_direction')
                    logger.info(f"  ⚡ Multi-timeframe consensus: {direction}")
                
                # Check for reversion candidates
                for tf_name, tf_data in sample_signal.get('signals', {}).get('signals_by_timeframe', {}).items():
                    reversion = tf_data.get('reversion', {})
                    if reversion.get('reversion_candidate'):
                        logger.info(f"  🔄 Reversion candidate on {tf_name}: {reversion.get('reversion_direction')}")
                        logger.info(f"     Probability: {reversion.get('reversion_probability', 0):.2%}")
                
                # Check for clustering signals
                for tf_name, tf_data in sample_signal.get('signals', {}).get('signals_by_timeframe', {}).items():
                    clustering = tf_data.get('clustering', {})
                    if clustering.get('trend_formation_signal'):
                        logger.info(f"  📊 Trend formation detected on {tf_name}: {clustering.get('dominant_direction')}")
            
            # Check market state
            global_state = tick_data.get('global', {})
            if global_state:
                sentiment = global_state.get('market_sentiment')
                bull_count = global_state.get('bullish_markets', 0)
                bear_count = global_state.get('bearish_markets', 0)
                logger.info(f"Market sentiment: {sentiment} ({bull_count} bullish, {bear_count} bearish)")
            
            iteration += 1
            await asyncio.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        logger.info("Shutting down scanner...")
        scanner.stop_scanning()
        await asyncio.sleep(2)
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        scanner.stop_scanning()


if __name__ == "__main__":
    asyncio.run(main())