"""
Seven Additional Strategy Modules for MirrorCore-X Trading System

These strategies complement the existing UT Bot, Gradient Trend, and Volume Support/Resistance agents
with diverse approaches covering different market conditions and timeframes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import logging
from collections import deque
from scipy import stats
from sklearn.ensemble import IsolationForest
import talib

logger = logging.getLogger(__name__)

# Base strategy interface (matches your existing pattern)
class BaseStrategyAgent:
    """Base class for all strategy agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = {}
        self.confidence = 0.0
        
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update strategy with new data"""
        raise NotImplementedError
        
    def get_signal(self) -> Dict[str, Any]:
        """Get current signal"""
        return self.signals

# 1. MEAN REVERSION STRATEGY
class MeanReversionAgent(BaseStrategyAgent):
    """
    Bollinger Bands + RSI mean reversion strategy
    - Identifies overbought/oversold conditions
    - Uses Z-score for entry/exit signals
    - Best in ranging/sideways markets
    """
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14, 
                 zscore_threshold: float = 2.0):
        super().__init__("MEAN_REVERSION")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.zscore_threshold = zscore_threshold
        self.price_buffer = deque(maxlen=50)
        
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            df = data.get('market_data_df')
            if df is None or df.empty:
                return {}
                
            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(self.bb_period).mean()
            df['bb_std'] = df['close'].rolling(self.bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.bb_std)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.bb_std)
            
            # Calculate RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            
            # Z-score calculation
            df['zscore'] = (df['close'] - df['bb_middle']) / df['bb_std']
            
            signals = {}
            for idx, row in df.iterrows():
                symbol = row.get('symbol', 'UNKNOWN')
                
                # Mean reversion signals
                if (row['zscore'] > self.zscore_threshold and row['rsi'] > 70):
                    signal = 'Strong Sell'  # Overbought
                    confidence = min(0.9, abs(row['zscore']) / self.zscore_threshold)
                elif (row['zscore'] < -self.zscore_threshold and row['rsi'] < 30):
                    signal = 'Strong Buy'   # Oversold
                    confidence = min(0.9, abs(row['zscore']) / self.zscore_threshold)
                elif abs(row['zscore']) < 0.5:
                    signal = 'Hold'         # Near mean
                    confidence = 0.3
                else:
                    signal = 'Hold'
                    confidence = 0.1
                    
                signals[symbol] = {
                    'signal': signal,
                    'confidence': confidence,
                    'zscore': row['zscore'],
                    'rsi': row['rsi'],
                    'bb_position': (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
                }
            
            self.signals = signals
            return {f"{self.name}_signals": signals}
            
        except Exception as e:
            logger.error(f"MeanReversionAgent update failed: {e}")
            return {}

# 2. MOMENTUM BREAKOUT STRATEGY
class MomentumBreakoutAgent(BaseStrategyAgent):
    """
    ATR-based breakout strategy with volume confirmation
    - Identifies high-momentum breakouts
    - Uses Average True Range for volatility adjustment
    - Volume surge confirmation reduces false signals
    """
    
    def __init__(self, atr_period: int = 14, breakout_multiplier: float = 2.0, 
                 volume_threshold: float = 1.5, lookback_period: int = 20):
        super().__init__("MOMENTUM_BREAKOUT")
        self.atr_period = atr_period
        self.breakout_multiplier = breakout_multiplier
        self.volume_threshold = volume_threshold
        self.lookback_period = lookback_period
        
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            df = data.get('market_data_df')
            if df is None or df.empty:
                return {}
                
            # Calculate ATR
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, 
                                 df['close'].values, timeperiod=self.atr_period)
            
            # Calculate breakout levels
            df['high_breakout'] = df['high'].rolling(self.lookback_period).max()
            df['low_breakout'] = df['low'].rolling(self.lookback_period).min()
            
            # Volume analysis
            df['avg_volume'] = df['volume'].rolling(self.lookback_period).mean()
            df['volume_ratio'] = df['volume'] / df['avg_volume']
            
            # Price momentum
            df['price_change'] = df['close'].pct_change(5)  # 5-period momentum
            
            signals = {}
            for idx, row in df.iterrows():
                symbol = row.get('symbol', 'UNKNOWN')
                
                # Breakout conditions
                upward_breakout = (row['close'] > row['high_breakout'] and 
                                 row['volume_ratio'] > self.volume_threshold and
                                 row['price_change'] > 0.02)
                
                downward_breakout = (row['close'] < row['low_breakout'] and 
                                   row['volume_ratio'] > self.volume_threshold and
                                   row['price_change'] < -0.02)
                
                if upward_breakout:
                    signal = 'Strong Buy'
                    confidence = min(0.9, row['volume_ratio'] / self.volume_threshold * 0.7)
                elif downward_breakout:
                    signal = 'Strong Sell'
                    confidence = min(0.9, row['volume_ratio'] / self.volume_threshold * 0.7)
                else:
                    signal = 'Hold'
                    confidence = 0.2
                    
                signals[symbol] = {
                    'signal': signal,
                    'confidence': confidence,
                    'atr': row['atr'],
                    'volume_ratio': row['volume_ratio'],
                    'momentum': row['price_change']
                }
            
            self.signals = signals
            return {f"{self.name}_signals": signals}
            
        except Exception as e:
            logger.error(f"MomentumBreakoutAgent update failed: {e}")
            return {}

# 3. VOLATILITY REGIME STRATEGY
class VolatilityRegimeAgent(BaseStrategyAgent):
    """
    Adapts strategy based on market volatility regime
    - Low vol: Mean reversion bias
    - High vol: Trend following bias
    - Uses GARCH-like volatility clustering detection
    """
    
    def __init__(self, vol_window: int = 20, regime_threshold: float = 1.5):
        super().__init__("VOLATILITY_REGIME")
        self.vol_window = vol_window
        self.regime_threshold = regime_threshold
        self.vol_history = deque(maxlen=100)
        
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            df = data.get('market_data_df')
            if df is None or df.empty:
                return {}
                
            # Calculate realized volatility
            df['returns'] = df['close'].pct_change()
            df['realized_vol'] = df['returns'].rolling(self.vol_window).std() * np.sqrt(252)  # Annualized
            
            # Historical volatility percentile
            if len(df) > 50:
                df['vol_percentile'] = df['realized_vol'].rolling(50).rank(pct=True)
            else:
                df['vol_percentile'] = 0.5
                
            # Volatility regime classification
            df['vol_regime'] = np.where(df['vol_percentile'] > 0.8, 'HIGH',
                                      np.where(df['vol_percentile'] < 0.2, 'LOW', 'MEDIUM'))
            
            # MACD for trend
            df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'].values)
            
            signals = {}
            for idx, row in df.iterrows():
                symbol = row.get('symbol', 'UNKNOWN')
                
                vol_regime = row['vol_regime']
                macd_signal = 'bullish' if row['macd'] > row['macdsignal'] else 'bearish'
                
                if vol_regime == 'HIGH':
                    # High volatility: Trend following
                    if macd_signal == 'bullish' and row['macdhist'] > 0:
                        signal = 'Buy'
                        confidence = 0.7
                    elif macd_signal == 'bearish' and row['macdhist'] < 0:
                        signal = 'Sell'
                        confidence = 0.7
                    else:
                        signal = 'Hold'
                        confidence = 0.3
                        
                elif vol_regime == 'LOW':
                    # Low volatility: Mean reversion
                    if macd_signal == 'bearish' and row['macdhist'] < 0:
                        signal = 'Buy'  # Contrarian
                        confidence = 0.6
                    elif macd_signal == 'bullish' and row['macdhist'] > 0:
                        signal = 'Sell'  # Contrarian
                        confidence = 0.6
                    else:
                        signal = 'Hold'
                        confidence = 0.3
                else:
                    # Medium volatility: Neutral
                    signal = 'Hold'
                    confidence = 0.2
                    
                signals[symbol] = {
                    'signal': signal,
                    'confidence': confidence,
                    'vol_regime': vol_regime,
                    'vol_percentile': row['vol_percentile'],
                    'realized_vol': row['realized_vol']
                }
            
            self.signals = signals
            return {f"{self.name}_signals": signals}
            
        except Exception as e:
            logger.error(f"VolatilityRegimeAgent update failed: {e}")
            return {}

# 4. PAIRS TRADING STRATEGY
class PairsTradingAgent(BaseStrategyAgent):
    """
    Statistical arbitrage between correlated pairs
    - Identifies co-integrated pairs
    - Uses z-score of spread for entry/exit
    - Market neutral strategy
    """
    
    def __init__(self, lookback_period: int = 60, zscore_entry: float = 2.0, 
                 zscore_exit: float = 0.5, min_correlation: float = 0.7):
        super().__init__("PAIRS_TRADING")
        self.lookback_period = lookback_period
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.min_correlation = min_correlation
        self.pairs_data = {}
        
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            df = data.get('market_data_df')
            if df is None or df.empty:
                return {}
                
            # Get unique symbols
            symbols = df['symbol'].unique()
            if len(symbols) < 2:
                return {}
                
            signals = {}
            
            # Create price matrix
            price_matrix = df.pivot(index='timestamp', columns='symbol', values='close')
            
            # Find pairs with high correlation
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols[i+1:], i+1):
                    if sym1 == sym2:
                        continue
                        
                    pair_key = f"{sym1}_{sym2}"
                    
                    if len(price_matrix) >= self.lookback_period:
                        # Calculate correlation
                        corr = price_matrix[sym1].corr(price_matrix[sym2])
                        
                        if abs(corr) > self.min_correlation:
                            # Calculate spread
                            spread = price_matrix[sym1] - price_matrix[sym2]
                            spread_mean = spread.rolling(self.lookback_period).mean()
                            spread_std = spread.rolling(self.lookback_period).std()
                            zscore = (spread - spread_mean) / spread_std
                            
                            current_zscore = zscore.iloc[-1] if not zscore.empty else 0
                            
                            # Generate signals
                            if current_zscore > self.zscore_entry:
                                # Spread too high: short sym1, long sym2
                                signals[sym1] = {'signal': 'Sell', 'confidence': 0.7, 'pair': sym2}
                                signals[sym2] = {'signal': 'Buy', 'confidence': 0.7, 'pair': sym1}
                            elif current_zscore < -self.zscore_entry:
                                # Spread too low: long sym1, short sym2
                                signals[sym1] = {'signal': 'Buy', 'confidence': 0.7, 'pair': sym2}
                                signals[sym2] = {'signal': 'Sell', 'confidence': 0.7, 'pair': sym1}
                            elif abs(current_zscore) < self.zscore_exit:
                                # Exit positions
                                signals[sym1] = {'signal': 'Hold', 'confidence': 0.8, 'pair': sym2}
                                signals[sym2] = {'signal': 'Hold', 'confidence': 0.8, 'pair': sym1}
                                
                            # Store pair data
                            self.pairs_data[pair_key] = {
                                'correlation': corr,
                                'zscore': current_zscore,
                                'spread': spread.iloc[-1] if not spread.empty else 0
                            }
            
            self.signals = signals
            return {f"{self.name}_signals": signals}
            
        except Exception as e:
            logger.error(f"PairsTradingAgent update failed: {e}")
            return {}

# 5. ANOMALY DETECTION STRATEGY
class AnomalyDetectionAgent(BaseStrategyAgent):
    """
    Machine learning-based anomaly detection
    - Uses Isolation Forest to detect unusual price/volume patterns
    - Contrarian approach: trade against anomalies
    - Self-adapting to market microstructure changes
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        super().__init__("ANOMALY_DETECTION")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
        self.is_fitted = False
        self.feature_buffer = deque(maxlen=200)
        
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for anomaly detection"""
        features = pd.DataFrame()
        
        # Price features
        features['price_change'] = df['close'].pct_change()
        features['price_volatility'] = df['close'].rolling(5).std()
        features['price_momentum'] = df['close'].pct_change(5)
        
        # Volume features
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Technical indicators
        features['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        features['bb_position'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Microstructure features
        features['spread_proxy'] = (df['high'] - df['low']) / df['close']
        features['price_impact'] = abs(df['close'] - df['open']) / df['volume']
        
        return features.fillna(0)
        
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            df = data.get('market_data_df')
            if df is None or df.empty:
                return {}
                
            features_df = self._extract_features(df)
            
            # Build training buffer
            for idx, row in features_df.iterrows():
                self.feature_buffer.append(row.values)
                
            # Train model if enough data
            if len(self.feature_buffer) >= 50 and not self.is_fitted:
                X = np.array(list(self.feature_buffer))
                self.model.fit(X)
                self.is_fitted = True
                logger.info("AnomalyDetectionAgent: Model fitted")
                
            signals = {}
            if self.is_fitted:
                # Detect anomalies
                X_current = features_df.values
                anomaly_scores = self.model.decision_function(X_current)
                is_anomaly = self.model.predict(X_current) == -1
                
                for idx, row in df.iterrows():
                    symbol = row.get('symbol', 'UNKNOWN')
                    
                    if idx < len(anomaly_scores):
                        anomaly_score = anomaly_scores[idx]
                        is_outlier = is_anomaly[idx]
                        
                        if is_outlier:
                            # Contrarian signal on anomalies
                            if anomaly_score < -0.5:  # Strong negative anomaly
                                signal = 'Buy'  # Price likely oversold
                                confidence = min(0.8, abs(anomaly_score))
                            elif anomaly_score > 0.5:  # Strong positive anomaly
                                signal = 'Sell'  # Price likely overbought
                                confidence = min(0.8, abs(anomaly_score))
                            else:
                                signal = 'Hold'
                                confidence = 0.3
                        else:
                            signal = 'Hold'
                            confidence = 0.1
                            
                        signals[symbol] = {
                            'signal': signal,
                            'confidence': confidence,
                            'anomaly_score': anomaly_score,
                            'is_anomaly': is_outlier
                        }
            
            self.signals = signals
            return {f"{self.name}_signals": signals}
            
        except Exception as e:
            logger.error(f"AnomalyDetectionAgent update failed: {e}")
            return {}

# 6. SENTIMENT MOMENTUM STRATEGY
class SentimentMomentumAgent(BaseStrategyAgent):
    """
    Combines technical momentum with sentiment analysis
    - Uses price momentum + volume + implied sentiment
    - Momentum confirmation across multiple timeframes
    - Sentiment-weighted position sizing
    """
    
    def __init__(self, short_period: int = 5, long_period: int = 20, sentiment_weight: float = 0.3):
        super().__init__("SENTIMENT_MOMENTUM")
        self.short_period = short_period
        self.long_period = long_period
        self.sentiment_weight = sentiment_weight
        
    def _calculate_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """Calculate implied sentiment from price action"""
        # Sentiment proxy from price/volume relationship
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        
        # Strong moves with high volume = positive sentiment
        sentiment = price_change * np.log1p(df['volume'] / df['volume'].rolling(10).mean())
        return sentiment.rolling(5).mean().fillna(0)
        
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            df = data.get('market_data_df')
            if df is None or df.empty:
                return {}
                
            # Calculate multiple momentum timeframes
            df['momentum_short'] = df['close'].pct_change(self.short_period)
            df['momentum_long'] = df['close'].pct_change(self.long_period)
            df['momentum_volume'] = (df['close'].pct_change() * 
                                   np.log1p(df['volume'] / df['volume'].rolling(10).mean()))
            
            # Calculate sentiment
            df['sentiment'] = self._calculate_sentiment(df)
            
            # Moving averages
            df['ema_short'] = df['close'].ewm(span=self.short_period).mean()
            df['ema_long'] = df['close'].ewm(span=self.long_period).mean()
            
            signals = {}
            for idx, row in df.iterrows():
                symbol = row.get('symbol', 'UNKNOWN')
                
                # Momentum alignment
                momentum_bullish = (row['momentum_short'] > 0 and 
                                  row['momentum_long'] > 0 and
                                  row['ema_short'] > row['ema_long'])
                
                momentum_bearish = (row['momentum_short'] < 0 and 
                                  row['momentum_long'] < 0 and
                                  row['ema_short'] < row['ema_long'])
                
                # Sentiment adjustment
                sentiment_bullish = row['sentiment'] > 0.001
                sentiment_bearish = row['sentiment'] < -0.001
                
                # Combined signals
                if momentum_bullish and sentiment_bullish:
                    signal = 'Strong Buy'
                    confidence = 0.8 + abs(row['sentiment']) * self.sentiment_weight
                elif momentum_bullish and not sentiment_bearish:
                    signal = 'Buy'
                    confidence = 0.6
                elif momentum_bearish and sentiment_bearish:
                    signal = 'Strong Sell'
                    confidence = 0.8 + abs(row['sentiment']) * self.sentiment_weight
                elif momentum_bearish and not sentiment_bullish:
                    signal = 'Sell'
                    confidence = 0.6
                else:
                    signal = 'Hold'
                    confidence = 0.2
                    
                signals[symbol] = {
                    'signal': signal,
                    'confidence': min(0.9, confidence),
                    'momentum_short': row['momentum_short'],
                    'momentum_long': row['momentum_long'],
                    'sentiment': row['sentiment']
                }
            
            self.signals = signals
            return {f"{self.name}_signals": signals}
            
        except Exception as e:
            logger.error(f"SentimentMomentumAgent update failed: {e}")
            return {}

# 7. REGIME CHANGE DETECTION STRATEGY
class RegimeChangeAgent(BaseStrategyAgent):
    """
    Detects structural breaks and regime changes in market behavior
    - Uses Hidden Markov Models concepts
    - Identifies trend/range regime transitions
    - Early warning system for market shifts
    """
    
    def __init__(self, window_size: int = 50, sensitivity: float = 2.0):
        super().__init__("REGIME_CHANGE")
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.regime_history = deque(maxlen=100)
        self.current_regime = 'UNKNOWN'
        
    def _detect_regime_change(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect regime changes using statistical methods"""
        if len(returns) < self.window_size:
            return {'regime': 'UNKNOWN', 'confidence': 0.0, 'change_detected': False}
            
        # Rolling statistics
        rolling_mean = returns.rolling(self.window_size // 2).mean()
        rolling_std = returns.rolling(self.window_size // 2).std()
        rolling_skew = returns.rolling(self.window_size // 2).skew()
        
        # Regime classification based on volatility and trend
        recent_vol = rolling_std.iloc[-5:].mean() if len(rolling_std) >= 5 else 0
        recent_trend = rolling_mean.iloc[-5:].mean() if len(rolling_mean) >= 5 else 0
        recent_skew = rolling_skew.iloc[-5:].mean() if len(rolling_skew) >= 5 else 0
        
        # Historical percentiles
        vol_percentile = stats.percentileofscore(rolling_std.dropna(), recent_vol) / 100
        trend_percentile = stats.percentileofscore(rolling_mean.dropna(), abs(recent_trend)) / 100
        
        # Regime determination
        if vol_percentile > 0.8:
            new_regime = 'HIGH_VOLATILITY'
        elif vol_percentile < 0.2:
            new_regime = 'LOW_VOLATILITY'
        elif trend_percentile > 0.7 and abs(recent_trend) > recent_vol:
            new_regime = 'TRENDING'
        else:
            new_regime = 'RANGING'
            
        # Detect regime change
        change_detected = (new_regime != self.current_regime and 
                         self.current_regime != 'UNKNOWN')
        
        confidence = max(vol_percentile, trend_percentile) if change_detected else 0.3
        
        return {
            'regime': new_regime,
            'confidence': confidence,
            'change_detected': change_detected,
            'vol_percentile': vol_percentile,
            'trend_strength': trend_percentile
        }
        
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            df = data.get('market_data_df')
            if df is None or df.empty:
                return {}
                
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            signals = {}
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                
                if len(symbol_df) < 20:
                    continue
                    
                regime_info = self._detect_regime_change(symbol_df['returns'])
                
                # Update current regime
                if regime_info['change_detected']:
                    self.current_regime = regime_info['regime']
                    self.regime_history.append({
                        'timestamp': symbol_df['timestamp'].iloc[-1],
                        'regime': self.current_regime,
                        'symbol': symbol
                    })
                
                # Generate signals based on regime
                current_regime = regime_info['regime']
                confidence = regime_info['confidence']
                
                if regime_info['change_detected']:
                    if current_regime == 'TRENDING':
                        # Early trend detection
                        recent_momentum = symbol_df['returns'].tail(5).mean()
                        signal = 'Buy' if recent_momentum > 0 else 'Sell'
                        signal_confidence = 0.8
                    elif current_regime == 'HIGH_VOLATILITY':
                        # Volatility breakout
                        signal = 'Hold'  # Wait for direction
                        signal_confidence = 0.4
                    elif current_regime == 'RANGING':
                        # Mean reversion opportunity
                        current_price = symbol_df['close'].iloc[-1]
                        mean_price = symbol_df['close'].tail(20).mean()
                        signal = 'Buy' if current_price < mean_price else 'Sell'
                        signal_confidence = 0.6
                    else:
                        signal = 'Hold'
                        signal_confidence = 0.3
                else:
                    signal = 'Hold'
                    signal_confidence = 0.2
                    
                signals[symbol] = {
                    'signal': signal,
                    'confidence': signal_confidence,
                    'regime': current_regime,
                    'regime_confidence': confidence,
                    'change_detected': regime_info['change_detected']
                }
            
            self.signals = signals
            return {f"{self.name}_signals": signals}
            
        except Exception as e:
            logger.error(f"RegimeChangeAgent update failed: {e}")
            return {}

# Integration helper function
def register_additional_strategies(strategy_trainer):
    """
    Register all 7 additional strategies with the StrategyTrainerAgent
    
    Args:
        strategy_trainer: StrategyTrainerAgent instance
    """
    try:
        # Register all new strategies
        strategy_trainer.register_strategy("MEAN_REVERSION", MeanReversionAgent())
        strategy_trainer.register_strategy("MOMENTUM_BREAKOUT", MomentumBreakoutAgent())
        strategy_trainer.register_strategy("VOLATILITY_REGIME", VolatilityRegimeAgent())
        strategy_trainer.register_strategy("PAIRS_TRADING", PairsTradingAgent())
        strategy_trainer.register_strategy("ANOMALY_DETECTION", AnomalyDetectionAgent())
        strategy_trainer.register_strategy("SENTIMENT_MOMENTUM", SentimentMomentumAgent())
        strategy_trainer.register_strategy("REGIME_CHANGE", RegimeChangeAgent())
        
        logger.info("Successfully registered 7 additional strategies")
        
    except Exception as e:
        logger.error(f"Failed to register additional strategies: {e}")
