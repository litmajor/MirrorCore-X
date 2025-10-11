# Import additional strategies
from additional_strategies import (
    MeanReversionAgent,
    MomentumBreakoutAgent,
    VolatilityRegimeAgent,
    PairsTradingAgent,
    AnomalyDetectionAgent,
    SentimentMomentumAgent,
    RegimeChangeAgent,
    register_additional_strategies
)

# MirrorCore-X: A Multi-Agent Cognitive Trading System
# Enhanced with High-Performance SyncBus and Data Pipeline Separation

# Key Features:
# - High-Performance SyncBus with delta updates and interest-based routing
# - Fault-resistant agent isolation and circuit breakers
# - Separated market data pipeline from agent state management
# - Real-time monitoring and command interface capabilities
# - Merged RL integration, additional agents, and features from previous version

import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import defaultdict, deque, Counter
from pydantic import BaseModel, Field, ConfigDict
from bayes_opt import BayesianOptimization
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime, timezone
from dataclasses import dataclass
import os
from tabulate import tabulate
import pickle

# Direct imports for optimizer and agent classes (no fallback/mocks)
from mirror_optimizer import ProductionSafeMirrorOptimizer, SafeOptimizableAgent
from scanner import MomentumScanner
from trade_analyzer_agent import TradeAnalyzerAgent
from arch_ctrl import ARCH_CTRL
from strategy_trainer_agent import UTSignalAgent, GradientTrendAgent, SupportResistanceAgent
from additional_strategies import (
    MeanReversionAgent,
    MomentumBreakoutAgent,
    VolatilityRegimeAgent,
    PairsTradingAgent,
    AnomalyDetectionAgent,
    SentimentMomentumAgent,
    RegimeChangeAgent
)
# --- ExecutionDaemon: Handles Order Execution and Management ---
class ExecutionDaemon:
    """Executes orders, manages open positions, and handles order errors."""
    def __init__(self, exchange: Any, dry_run: bool = True):
        self.exchange = exchange
        self.dry_run = dry_run
        self.open_orders = {}
        self.closed_orders = []
        self.last_error = None
        logger.info(f"ExecutionDaemon initialized (dry_run={dry_run})")

    async def execute_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None, order_type: str = 'market', params: Optional[dict] = None) -> dict:
        """Executes an order on the exchange or simulates if dry_run."""
        params = params or {}
        try:
            if self.dry_run:
                # Simulate order execution
                order_id = f"sim_{symbol}_{side}_{int(time.time())}"
                order = {
                    'id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'type': order_type,
                    'status': 'closed',
                    'timestamp': time.time(),
                    'simulated': True
                }
                self.closed_orders.append(order)
                logger.info(f"Simulated order executed: {order}")
                return order
            else:
                # Live order execution
                if order_type == 'market':
                    result = await self.exchange.create_market_order(symbol, side, amount, params)
                else:
                    result = await self.exchange.create_limit_order(symbol, side, amount, price, params)
                self.open_orders[result['id']] = result
                logger.info(f"Order placed: {result}")
                return result
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Order execution failed: {e}")
            return {'error': str(e)}

    async def close_order(self, order_id: str) -> dict:
        """Closes an open order if possible."""
        try:
            if self.dry_run:
                order = next((o for o in self.closed_orders if o['id'] == order_id), None)
                if order:
                    order['status'] = 'closed'
                    logger.info(f"Simulated order closed: {order_id}")
                    return order
                return {'error': 'Order not found'}
            else:
                if order_id in self.open_orders:
                    result = await self.exchange.cancel_order(order_id)
                    self.closed_orders.append(result)
                    del self.open_orders[order_id]
                    logger.info(f"Order closed: {order_id}")
                    return result
                return {'error': 'Order not found'}
        except Exception as e:
            logger.error(f"Failed to close order {order_id}: {e}")
            return {'error': str(e)}

    def get_open_orders(self) -> dict:
        """Returns all open orders."""
        return self.open_orders

    def get_closed_orders(self) -> list:
        """Returns all closed orders."""
        return self.closed_orders

    def get_last_error(self) -> Optional[str]:
        """Returns the last error encountered."""
        return self.last_error


class TradingConfig:
    """Configuration for trading scanner parameters."""
    def __init__(self,
                 timeframes: Optional[list] = None,
                 momentum_threshold: float = 0.05,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
                 macd_signal_threshold: float = 0.0,
                 volume_min: float = 1000.0,
                 symbols: Optional[list] = None):
        self.timeframes = timeframes or ['1h', '4h', '1d']
        self.momentum_threshold = momentum_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_signal_threshold = macd_signal_threshold
        self.volume_min = volume_min
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']

    def to_dict(self):
        return {
            'timeframes': self.timeframes,
            'momentum_threshold': self.momentum_threshold,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'macd_signal_threshold': self.macd_signal_threshold,
            'volume_min': self.volume_min,
            'symbols': self.symbols
        }

class StrategyTrainerAgent:
    def __init__(self):
        self.strategies = {}
        self.performance_tracker = {}
        self.learning_rate = 0.01
        self.confidence_threshold = 0.6
        self.performance_history = []

    def register_strategy(self, name, agent):
        self.strategies[name] = agent
        self.performance_tracker[name] = []

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "confidence_threshold": self.confidence_threshold
        }

    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        self.learning_rate = float(params.get("learning_rate", self.learning_rate))
        self.confidence_threshold = float(params.get("confidence_threshold", self.confidence_threshold))

    def validate_params(self, params: Dict[str, Any]) -> bool:
        lr = params.get("learning_rate", self.learning_rate)
        conf = params.get("confidence_threshold", self.confidence_threshold)
        return (0.001 <= lr <= 0.1 and 0.1 <= conf <= 0.9)

    def evaluate(self) -> float:
        if not self.performance_history:
            return 0.5
        return sum(self.performance_history[-20:]) / min(20, len(self.performance_history))

    def update_performance(self, strategy_name: str, pnl: float):
        if strategy_name in self.performance_tracker:
            self.performance_tracker[strategy_name].append(pnl)
            # Add to overall performance
            normalized_score = max(0, min(1, (pnl + 100) / 200))  # Normalize PnL to 0-1
            self.performance_history.append(normalized_score)
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

    def grade_strategies(self) -> Dict[str, str]:
        grades = {}
        for name, performance in self.performance_tracker.items():
            if performance:
                avg_performance = sum(performance[-10:]) / min(10, len(performance))
                if avg_performance > 50:
                    grades[name] = "A"
                elif avg_performance > 0:
                    grades[name] = "B"
                else:
                    grades[name] = "C"
            else:
                grades[name] = "N/A"
        return grades

# --- Psychological & Meta-Cognitive Agents ---
class EgoProcessor:
    """Tracks and updates the system's ego state (confidence, stress, etc.)."""
    def __init__(self):
        self.confidence = 0.5
        self.stress = 0.5
        self.recent_pnl = 0.0
        self.history = []
        logger.info("EgoProcessor initialized")

    async def update(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        pnl = performance_metrics.get('total_pnl', 0.0)
        win_rate = performance_metrics.get('win_rate', 0.5)
        # Confidence increases with win rate and PnL, stress increases with losses
        self.confidence = np.clip(0.5 + 0.5 * (win_rate - 0.5) + 0.1 * np.tanh(pnl / 1000), 0.0, 1.0)
        self.stress = np.clip(0.5 - 0.5 * (win_rate - 0.5) - 0.1 * np.tanh(pnl / 1000), 0.0, 1.0)
        self.recent_pnl = pnl
        state = {
            'confidence': self.confidence,
            'stress': self.stress,
            'recent_pnl': self.recent_pnl
        }
        self.history.append(state)
        return state

class FearAnalyzer:
    """Monitors risk, drawdown, and volatility to quantify system 'fear'."""
    def __init__(self, max_drawdown: float = -0.1):
        self.max_drawdown = max_drawdown
        self.fear_level = 0.0
        self.history = []
        logger.info("FearAnalyzer initialized")

    async def update(self, performance_metrics: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        pnl = performance_metrics.get('total_pnl', 0.0)
        initial_capital = 10000.0
        drawdown = (pnl / initial_capital) if initial_capital else 0.0
        volatility = 0.0
        if market_data:
            volatility = market_data.get('volatility', 0.0)
        # Fear increases with drawdown and volatility
        self.fear_level = np.clip(abs(drawdown / self.max_drawdown) + volatility, 0.0, 1.0)
        self.history.append(self.fear_level)
        return self.fear_level

class SelfAwarenessAgent:
    """Monitors internal system health, drift, and trust."""
    def __init__(self):
        self.drift = 0.0
        self.trust = 0.8
        self.deviations = 0
        self.history = []
        logger.info("SelfAwarenessAgent initialized")

    async def update(self, agent_states: Dict[str, Any]) -> Dict[str, Any]:
        # Drift: how much agent states deviate from expected
        # Trust: decreases if many agents are isolated or failing
        isolated_agents = [k for k, v in agent_states.items() if v.get('circuit_open', False)]
        self.deviations = len(isolated_agents)
        self.drift = np.clip(self.deviations / max(len(agent_states), 1), 0.0, 1.0)
        self.trust = np.clip(1.0 - self.drift, 0.0, 1.0)
        state = {
            'drift': self.drift,
            'trust': self.trust,
            'deviations': self.deviations
        }
        self.history.append(state)
        return state

class DecisionMirror:
    """Records, analyzes, and reflects on recent decisions for feedback."""
    def __init__(self):
        self.decision_log = []
        self.last_feedback = None
        logger.info("DecisionMirror initialized")

    async def update(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        self.decision_log.append(decision)
        # Simple feedback: reward consistency, penalize erratic changes
        feedback = {'consistency': 1.0, 'erratic': False}
        if len(self.decision_log) > 2:
            prev = self.decision_log[-2]
            if abs(decision.get('final_position', 0.0) - prev.get('final_position', 0.0)) > 0.7:
                feedback['consistency'] = 0.0
                feedback['erratic'] = True
        self.last_feedback = feedback
        return feedback

class MetaAgent:
    """Coordinates meta-cognitive functions and triggers system-level adaptations."""
    def __init__(self, ego_processor: EgoProcessor, fear_analyzer: FearAnalyzer, self_awareness: SelfAwarenessAgent, decision_mirror: DecisionMirror):
        self.ego_processor = ego_processor
        self.fear_analyzer = fear_analyzer
        self.self_awareness = self_awareness
        self.decision_mirror = decision_mirror
        self.meta_history = []
        logger.info("MetaAgent initialized")

    async def update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Gather states from all sub-agents
        ego_state = await self.ego_processor.update(context.get('performance_metrics', {}))
        fear_level = await self.fear_analyzer.update(context.get('performance_metrics', {}), context.get('market_data', {}))
        self_state = await self.self_awareness.update(context.get('agent_states', {}))
        decision_feedback = await self.decision_mirror.update(context.get('decision', {}))
        meta_insights = {
            'ego_state': ego_state,
            'fear_level': fear_level,
            'self_state': self_state,
            'decision_feedback': decision_feedback
        }
        self.meta_history.append(meta_insights)
        return meta_insights

# --- TradingBot: Handles Directive Execution ---
class TradingBot:
    def __init__(self, exchange: Any, dry_run: bool = True):
        self.exchange = exchange
        self.dry_run = dry_run
        self.positions = {}
        logger.info("TradingBot initialized")

    async def process_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading directive."""
        try:
            symbol = directive.get('symbol', '')
            action = directive.get('action', '')
            amount = directive.get('amount', 0)
            price = directive.get('price', 0)

            if self.dry_run:
                logger.info(f"Dry run: {action} {amount:.2f} of {symbol} at {price:.2f}")
                return {'symbol': symbol, 'action': action, 'amount': amount, 'price': price, 'executed': True}

            # Real execution logic (simplified)
            if action in ['buy', 'sell']:
                order = await self.exchange.create_order(symbol, action, amount, price)
                self.positions[symbol] = self.positions.get(symbol, 0) + (amount if action == 'buy' else -amount)
                return {'symbol': symbol, 'action': action, 'amount': amount, 'price': price, 'executed': True}
            return {}
        except Exception as e:
            logger.error(f"Failed to process directive for {directive.get('symbol', '')}: {str(e)}")
            return {'error': str(e)}

# --- TradingOracleEngine: Core Decision-Making Engine ---
class TradingOracleEngine:

    async def evaluate_synthetic_scenarios(self, synthetic_data: List[Dict]) -> List[Dict]:
        """Evaluate trading directives against synthetic market scenarios."""
        try:
            directives = []
            for scenario in synthetic_data:
                market_context = {
                    'portfolio_value': 10000.0,
                    'volatility': scenario.get('volatility', 0.01),
                    'sentiment': scenario.get('sentiment', 0.0)
                }
                scanner_data = scenario.get('scanner_data', [])
                rl_predictions = scenario.get('rl_predictions', [])
                scenario_directives = await self.evaluate_trading_timelines(
                    market_context, scanner_data, rl_predictions
                )
                directives.extend(scenario_directives)
            return directives
        except Exception as e:
            logger.error(f"Failed to evaluate synthetic scenarios: {str(e)}")
            return []

# --- Imagination Engine (Placeholder for Scenario Generation) ---

    """Core decision-making engine that aggregates signals and generates trading directives."""
    def __init__(self, strategy_trainer: StrategyTrainerAgent, sync_bus: 'HighPerformanceSyncBus'):
        self.strategy_trainer = strategy_trainer
        self.sync_bus = sync_bus
        self.decision_threshold = 0.6  # Confidence threshold for actionable decisions
        self.max_position_size = 0.1   # Max position size as fraction of portfolio
        self.risk_factor = 0.05        # Risk adjustment factor
        self.signal_weights = {
            'UT_BOT': 0.3,
            'GRADIENT_TREND': 0.3,
            'VBSR': 0.2,
            'RL': 0.2
        }
        self.performance_metrics = {
            'decisions_made': 0,
            'successful_trades': 0,
            'total_pnl': 0.0
        }
        logger.info("TradingOracleEngine initialized")

    async def evaluate_trading_timelines(self, market_context: Dict[str, Any], scanner_data: List[Dict], rl_predictions: List[Dict]) -> List[Dict]:
        """Evaluate market and scanner data to generate trading directives."""
        try:
            directives = []
            portfolio_value = market_context.get('portfolio_value', 10000.0)
            market_volatility = market_context.get('volatility', 0.01)
            market_sentiment = market_context.get('sentiment', 0.0)

            for scan in scanner_data:
                try:
                    symbol = scan.get('symbol')
                    if not symbol:
                        continue

                    # Aggregate signals from registered strategies
                    aggregated_signal = await self._aggregate_signals(symbol, scan, rl_predictions)
                    confidence_score = aggregated_signal.get('confidence', 0.0)
                    position_signal = aggregated_signal.get('position', 0.0)

                    # Apply risk adjustments based on market context
                    adjusted_position = self._apply_risk_adjustments(
                        position_signal, market_volatility, market_sentiment, portfolio_value
                    )

                    # Generate directive if confidence exceeds threshold
                    if abs(confidence_score) >= self.decision_threshold:
                        directive = {
                            'symbol': symbol,
                            'action': 'buy' if adjusted_position > 0 else 'sell' if adjusted_position < 0 else 'hold',
                            'amount': abs(adjusted_position) * portfolio_value * self.max_position_size,
                            'price': scan.get('price', 1.0),
                            'confidence': confidence_score,
                            'strategy_weights': aggregated_signal.get('weights', {}),
                            'timestamp': time.time(),
                            'market_regime': self._determine_market_regime(market_volatility, market_sentiment)
                        }
                        directives.append(directive)
                        self.performance_metrics['decisions_made'] += 1
                        logger.debug(f"Generated directive for {symbol}: {directive['action']}, confidence={confidence_score:.2f}")

                except Exception as e:
                    logger.error(f"Error processing symbol {scan.get('symbol', '')}: {str(e)}")
                    continue

            # Update global state with directives
            await self.sync_bus.update_state('trading_directives', directives)
            return directives

        except Exception as e:
            logger.error(f"Failed to evaluate trading timelines: {str(e)}")
            return []

    async def _aggregate_signals(self, symbol: str, scan_data: Dict, rl_predictions: List[Dict]) -> Dict[str, Any]:
        """Aggregate signals from multiple strategies and RL predictions."""
        try:
            signals = {}
            weights = {}
            total_weight = 0.0
            weighted_sum = 0.0

            # Fetch strategy grades from StrategyTrainerAgent
            strategy_grades = self.strategy_trainer.grade_strategies()

            # Process rule-based strategy signals
            for strategy_name, agent in self.strategy_trainer.strategies.items():
                try:
                    # Assume agents have a method to generate signals
                    signal = self._get_strategy_signal(agent, scan_data)
                    grade = strategy_grades.get(strategy_name, 'B')
                    weight = self.signal_weights.get(strategy_name, 0.1)
                    if grade == 'A':
                        weight *= 1.2  # Boost weight for high-performing strategies
                    elif grade == 'C':
                        weight *= 0.8  # Reduce weight for poor-performing strategies
                    signals[strategy_name] = signal
                    weights[strategy_name] = weight
                    weighted_sum += signal * weight
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Strategy {strategy_name} failed for {symbol}: {str(e)}")
                    continue

            # Incorporate RL predictions if available
            rl_position = 0.0
            rl_confidence = 0.5
            for pred in rl_predictions:
                if pred.get('symbol') == symbol:
                    rl_position = pred.get('rl_position', 0.0)
                    rl_confidence = pred.get('confidence', 0.5)
                    weight = self.signal_weights.get('RL', 0.1)
                    signals['RL'] = rl_position
                    weights['RL'] = weight
                    weighted_sum += rl_position * weight
                    total_weight += weight
                    break

            # Calculate aggregated signal and confidence
            aggregated_position = weighted_sum / total_weight if total_weight > 0 else 0.0
            confidence = min(rl_confidence, max(abs(weighted_sum), 0.1))

            return {
                'position': aggregated_position,
                'confidence': confidence,
                'signals': signals,
                'weights': weights
            }

        except Exception as e:
            logger.error(f"Signal aggregation failed for {symbol}: {str(e)}")
            return {'position': 0.0, 'confidence': 0.0, 'signals': {}, 'weights': {}}

    def _get_strategy_signal(self, agent: Any, scan_data: Dict) -> float:
        """Extract signal from a strategy agent."""
        # Placeholder: assumes agents have a `generate_signal` method
        # In a real implementation, this would call specific agent methods
        signal_map = {
            'Strong Buy': 1.0, 'Buy': 0.6, 'Weak Buy': 0.3, 'Neutral': 0.0,
            'Weak Sell': -0.3, 'Sell': -0.6, 'Strong Sell': -1.0
        }
        signal = scan_data.get('signal', 'Neutral')  # Fallback to scanner signal
        return signal_map.get(signal, 0.0)

    def _apply_risk_adjustments(self, position: float, volatility: float, sentiment: float, portfolio_value: float) -> float:
        """Apply risk adjustments to position size."""
        risk_adjusted_position = position
        if volatility > 0.05:  # High volatility reduces position size
            risk_adjusted_position *= (1 - self.risk_factor * volatility * 10)
        if sentiment < -0.3:  # Negative sentiment reduces position size
            risk_adjusted_position *= 0.8
        elif sentiment > 0.3:  # Positive sentiment slightly boosts position
            risk_adjusted_position *= 1.1
        return np.clip(risk_adjusted_position, -self.max_position_size, self.max_position_size)

    def _determine_market_regime(self, volatility: float, sentiment: float) -> str:
        """Determine current market regime."""
        if volatility > 0.05:
            return "volatile"
        elif abs(sentiment) > 0.3:
            return "trending" if sentiment > 0 else "correcting"
        return "normal"

    async def update_performance(self, trade_record: 'TradeRecord'):
        """Update performance metrics based on trade outcomes."""
        try:
            self.performance_metrics['total_pnl'] += trade_record.pnl
            if trade_record.pnl > 0:
                self.performance_metrics['successful_trades'] += 1
            await self.sync_bus.update_state('oracle_performance', self.performance_metrics)
            logger.debug(f"Updated oracle performance: total_pnl={self.performance_metrics['total_pnl']:.2f}")
        except Exception as e:
            logger.error(f"Failed to update performance: {str(e)}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return current performance metrics."""
        return self.performance_metrics


# RL library import for PPO
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    # logger.warning("RL components not available. Running without RL capabilities.") # Removed to avoid dependency on logger before it's defined

logger = logging.getLogger(__name__)

# --- System Configuration ---
# Basic logging configuration for standalone execution
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler('mirrorcore_x.log') # Commented out to avoid file creation issues in restricted environments
        ]
    )

# Enhanced Data Contracts
class MarketData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    price: float = Field(gt=0.0, description="Current market price")
    volume: float = Field(ge=0.0, description="Trading volume")
    high: float = Field(gt=0.0, description="High price in period")
    low: float = Field(gt=0.0, description="Low price in period")
    open: float = Field(gt=0.0, description="Open price in period")
    timestamp: float = Field(gt=0.0, description="Data timestamp")
    volatility: Optional[float] = Field(default=None, ge=0.0, description="Price volatility")
    sentiment: Optional[float] = Field(default=0.0, ge=-1.0, le=1.0, description="Market sentiment")

class ScannerOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    momentum_7d: float = Field(description="7-day momentum score")
    signal: str = Field(description="Trading signal")
    rsi: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Relative Strength Index")
    macd: Optional[float] = Field(default=None, description="MACD indicator")
    timestamp: float = Field(gt=0.0, description="Signal timestamp")
    price: float = Field(gt=0.0, description="Current price")

class TradeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    entry: float = Field(gt=0.0, description="Entry price")
    exit: Optional[float] = Field(default=None, gt=0.0, description="Exit price")
    pnl: float = Field(description="Profit and loss")
    strategy: str = Field(default="UNSPECIFIED", description="Strategy used")
    timestamp: float = Field(default_factory=time.time, description="Trade timestamp")

class PsychProfile(BaseModel): # Added PsychProfile model
    model_config = ConfigDict(extra="ignore")
    emotional_state: str
    confidence_level: float = Field(ge=0.0, le=1.0)
    stress_level: float = Field(ge=0.0, le=1.0)
    recent_pnl: float

# Merged RL Configuration and Components from mirrax
@dataclass
class RLConfig:
    lookback_window: int = 20
    max_position_size: float = 1.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    reward_scaling: float = 1.0
    risk_penalty: float = 0.1
    drawdown_penalty: float = 0.2
    sharpe_reward_weight: float = 0.3
    return_reward_weight: float = 0.7
    learning_rate: float = 3e-4
    total_timesteps: int = 100000
    eval_freq: int = 5000
    save_freq: int = 10000

class SignalEncoder:
    def __init__(self, scaling_method: str = 'robust'):
        self.scaling_method = scaling_method
        self.scaler = RobustScaler() if scaling_method == 'robust' else MinMaxScaler()
        self.feature_names = [
            'price', 'volume', 'momentum_short', 'momentum_long', 
            'rsi', 'macd', 'bb_position', 'volume_ratio',
            'composite_score', 'trend_score', 'confidence_score',
            'fear_greed', 'btc_dominance', 'volatility',
            'ichimoku_bullish', 'vwap_bullish', 'ema_crossover'
        ]
        self.fitted = False
    
    def encode(self, df: pd.DataFrame) -> np.ndarray:
        for col in self.feature_names:
            if col not in df.columns:
                if 'bullish' in col or 'crossover' in col:
                    df[col] = 0.0
                elif col in ['fear_greed', 'btc_dominance']:
                    df[col] = 50.0
                elif col == 'rsi':
                    df[col] = 50.0
                else:
                    df[col] = 0.0
        features = df[self.feature_names].fillna(0.0)
        if not self.fitted:
            normalized = self.scaler.fit_transform(features)
            self.fitted = True
        else:
            normalized = self.scaler.transform(features)
        return normalized.astype(np.float32)
    
    def transform_live(self, latest_data: Dict) -> np.ndarray:
        row = np.array([[latest_data.get(k, self._get_default_value(k)) 
                        for k in self.feature_names]])
        return self.scaler.transform(row).astype(np.float32)
    
    def _get_default_value(self, feature: str) -> float:
        if 'bullish' in feature or 'crossover' in feature:
            return 0.0
        elif feature in ['fear_greed', 'btc_dominance', 'rsi']:
            return 50.0
        return 0.0

class TradingEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, market_data: np.ndarray, price_data: np.ndarray, config: RLConfig):
        super().__init__()
        self.config = config
        self.market_data = market_data
        self.price_data = price_data
        self.initial_balance = 10000.0
        self.lookback = self.config.lookback_window
        self.max_steps = len(market_data) - self.lookback - 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        n_features = market_data.shape[1]
        portfolio_features = 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback, n_features + portfolio_features), dtype=np.float32
        )
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.max_balance = self.initial_balance
        self.trade_count = 0
        self.win_count = 0
        self.balance_history = [self.initial_balance]
        self.position_history = [0.0]
        self.action_history = []
        self.reward_history = []
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray):
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, False, {}
        
        target_position = np.clip(action[0], -1.0, 1.0)
        position_change = target_position - self.position
        current_price = self.price_data[self.current_step]
        reward = 0.0
        
        if abs(position_change) > 0.01:
            reward += self._execute_trade(position_change, current_price)
        
        if self.position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            self.total_pnl = self.position * price_change * self.balance
        
        self.balance = self.initial_balance + self.total_pnl
        self.max_balance = max(self.max_balance, self.balance)
        reward += self._calculate_reward(current_price)
        self.current_step += 1
        self.balance_history.append(self.balance)
        self.position_history.append(self.position)
        self.action_history.append(target_position)
        self.reward_history.append(reward)
        
        done = (self.current_step >= self.max_steps or self.balance <= self.initial_balance * 0.5)
        info = {
            'balance': self.balance,
            'position': self.position,
            'pnl': self.total_pnl,
            'trade_count': self.trade_count,
            'win_rate': self.win_count / max(self.trade_count, 1),
            'drawdown': (self.max_balance - self.balance) / self.max_balance
        }
        return self._get_observation(), reward, done, False, info
    
    def _execute_trade(self, position_change: float, current_price: float) -> float:
        trade_cost = abs(position_change) * self.config.transaction_cost * self.balance
        old_position = self.position
        self.position = np.clip(self.position + position_change, -1.0, 1.0)
        
        if abs(self.position) > 0.01:
            if abs(old_position) < 0.01:
                self.entry_price = current_price
            elif np.sign(self.position) == np.sign(old_position):
                old_value = abs(old_position) * self.entry_price
                new_value = abs(position_change) * current_price
                total_volume = abs(old_position) + abs(position_change)
                self.entry_price = (old_value + new_value) / total_volume
        
        if abs(old_position) > 0.01 and abs(self.position) < 0.01:
            price_change = (current_price - self.entry_price) / self.entry_price
            trade_pnl = old_position * price_change * self.balance
            self.trade_count += 1
            if trade_pnl > 0:
                self.win_count += 1
        
        return -trade_cost / self.initial_balance
    
    def _calculate_reward(self, current_price: float) -> float:
        reward = 0.0
        if len(self.balance_history) > 1:
            balance_return = (self.balance - self.balance_history[-2]) / self.balance_history[-2]
            reward += balance_return * self.config.return_reward_weight
        
        if len(self.balance_history) > 10:
            returns = np.diff(self.balance_history[-10:]) / np.array(self.balance_history[-11:-1])
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_approx = np.mean(returns) / np.std(returns)
                reward += sharpe_approx * self.config.sharpe_reward_weight
        
        drawdown = (self.max_balance - self.balance) / self.max_balance
        if drawdown > 0.1:
            reward -= drawdown * self.config.drawdown_penalty
        
        risk_penalty = abs(self.position) ** 2 * self.config.risk_penalty
        reward -= risk_penalty
        return float(reward * self.config.reward_scaling)
    
    def _get_observation(self) -> np.ndarray:
        start_idx = max(0, self.current_step - self.lookback)
        market_features = self.market_data[start_idx:self.current_step]
        portfolio_state = np.array([[
            self.position, 
            self.balance / self.initial_balance - 1, 
            self.total_pnl / self.initial_balance, 
            (self.max_balance - self.balance) / self.max_balance
        ]])
        portfolio_features = np.repeat(portfolio_state, self.lookback, axis=0)
        
        if market_features.shape[0] < self.lookback:
            padding = np.zeros((self.lookback - market_features.shape[0], market_features.shape[1]))
            market_features = np.vstack([padding, market_features])
        
        return np.concatenate([market_features, portfolio_features], axis=1).astype(np.float32)

class RLTradingAgent:
    def __init__(self, algorithm: str = 'PPO', config: Optional[RLConfig] = None, model_path: str = ''):
        self.algorithm = algorithm
        self.config = config or RLConfig()
        self.model = None
        self.encoder = SignalEncoder()
        self.is_trained = False
        if model_path and RL_AVAILABLE:
            self.load_model(model_path)
    
    def prepare_training_data(self, scanner_results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(scanner_results, 'fear_greed_history') and len(scanner_results.fear_greed_history) > 0:
            scanner_results['fear_greed'] = scanner_results.fear_greed_history[-1]
        else:
            scanner_results['fear_greed'] = 50
        
        if hasattr(scanner_results, 'btc_dominance_history') and len(scanner_results.btc_dominance_history) > 0:
            scanner_results['btc_dominance'] = scanner_results.btc_dominance_history[-1]
        else:
            scanner_results['btc_dominance'] = 50
        
        scanner_results['volatility'] = scanner_results['momentum_short'].rolling(5).std().fillna(0)
        
        for col in ['ichimoku_bullish', 'vwap_bullish', 'ema_crossover']:
            if col not in scanner_results.columns:
                scanner_results[col] = 0.0
            else:
                scanner_results[col] = scanner_results[col].astype(float)
        
        market_features = self.encoder.encode(scanner_results)
        price_data = np.array(scanner_results['price'].values, dtype=np.float32)
        return market_features, price_data
    
    def create_environment(self, market_data: np.ndarray, price_data: np.ndarray) -> TradingEnvironment:
        return TradingEnvironment(market_data=market_data, price_data=price_data, config=self.config)
    
    def train(self, scanner_results: pd.DataFrame, save_path: str = 'rl_trading_model', verbose: int = 1):
        if not RL_AVAILABLE:
            logger.warning("RL training skipped - stable_baselines3 not available")
            return {'training_completed': False, 'error': 'RL not available'}
        
        logger.info(f"Starting RL training with {self.algorithm}")
        market_data, price_data = self.prepare_training_data(scanner_results)
        env = self.create_environment(market_data, price_data)
        
        if not RL_AVAILABLE:
            raise ImportError("stable_baselines3 PPO is not available")
        from stable_baselines3 import PPO
        self.model = PPO(
            'MlpPolicy', env, learning_rate=self.config.learning_rate, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
            max_grad_norm=0.5, verbose=verbose
        )
        self.model.learn(total_timesteps=self.config.total_timesteps)
        self.save_model(save_path)
        self.is_trained = True
        logger.info("RL training completed")
        return {
            'training_completed': True,
            'total_timesteps': self.config.total_timesteps,
            'algorithm': self.algorithm,
            'final_model_path': save_path
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        result = self.model.predict(observation, deterministic=deterministic)
        if isinstance(result, tuple) and len(result) == 2:
            action, state = result
            if isinstance(state, tuple):
                state = state[0] if len(state) > 0 else None
            return action, state
        else:
            return result, None
    
    def save_model(self, path: str):
        if self.model is None or not RL_AVAILABLE:
            raise ValueError("No model to save or RL not available")
        self.model.save(path)
        with open(f"{path}_encoder.pkl", 'wb') as f:
            pickle.dump(self.encoder, f)
        logger.info(f"Model and encoder saved to {path}")
    
    def load_model(self, path: str):
        if not RL_AVAILABLE or pickle is None:
            logger.warning("Cannot load model - RL not available")
            return
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(path)
            with open(f"{path}_encoder.pkl", 'rb') as f:
                self.encoder = pickle.load(f)
            self.is_trained = True
            logger.info(f"Model and encoder loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

class MetaController:
    def __init__(self, strategy: str = "confidence_blend"):
        self.strategy = strategy
        self.performance_history = []
    
    def decide(self, rule_decision: str, rl_action: np.ndarray, confidence_score: float = 0.5, 
               market_regime: str = "normal") -> Dict[str, Any]:
        rule_position = self._rule_to_position(rule_decision)
        rl_position = float(rl_action[0]) if isinstance(rl_action, np.ndarray) else rl_action
        
        if self.strategy == "confidence_blend":
            if confidence_score > 0.7:
                final_position = 0.7 * rl_position + 0.3 * rule_position
                method = "rl_weighted"
            elif confidence_score > 0.4:
                final_position = 0.5 * rl_position + 0.5 * rule_position
                method = "balanced"
            else:
                final_position = 0.3 * rl_position + 0.7 * rule_position
                method = "rule_weighted"
        else:
            final_position = 0.5 * rl_position + 0.5 * rule_position
            method = "balanced"
        
        final_position = np.clip(final_position, -1.0, 1.0)
        decision = {
            'final_position': final_position,
            'rule_position': rule_position,
            'rl_position': rl_position,
            'confidence_score': confidence_score,
            'method': method,
            'market_regime': market_regime
        }
        self.performance_history.append(decision)
        return decision
    
    def _rule_to_position(self, rule_decision: str) -> float:
        signal_map = {
            'Strong Buy': 1.0, 'Buy': 0.6, 'Weak Buy': 0.3, 'Neutral': 0.0,
            'Weak Sell': -0.3, 'Sell': -0.6, 'Strong Sell': -1.0,
            'Overbought': -0.4, 'Oversold': 0.4
        }
        return signal_map.get(rule_decision, 0.0)

class RLIntegrationLayer:
    """Layer that integrates RL capabilities with existing system"""
    
    def __init__(self, sync_bus, scanner):
        self.sync_bus = sync_bus
        self.scanner = scanner
        self.rl_agent = None
        self.meta_controller = None
        self.integration_metrics = {
            'rl_predictions_count': 0,
            'meta_decisions_count': 0,
            'integration_success_rate': 0.0
        }
        
        if RL_AVAILABLE:
            self._initialize_rl_components()
        
        logger.info(f"RL Integration Layer initialized (RL Available: {RL_AVAILABLE})")
    
    def _initialize_rl_components(self):
        """Initialize RL components if available"""
        try:
            if RL_AVAILABLE:
                config = RLConfig(
                    lookback_window=20,
                    max_position_size=0.1,
                    transaction_cost=0.001,
                    total_timesteps=50000,
                    learning_rate=3e-4
                )
                self.rl_agent = RLTradingAgent(algorithm='PPO', config=config)
                self.meta_controller = MetaController(strategy="confidence_blend")
                logger.info("RL components initialized successfully")
            else:
                self.rl_agent = None
                self.meta_controller = None
        except Exception as e:
            logger.error(f"RL component initialization failed: {e}")
            self.rl_agent = None
            self.meta_controller = None
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update RL integration layer"""
        if not RL_AVAILABLE or not self.rl_agent:
            return {}
        
        try:
            scanner_data = data.get('scanner_data', [])
            if not scanner_data:
                return {}
            
            scanner_df = pd.DataFrame(scanner_data)
            rl_predictions = []
            
            if self.rl_agent.is_trained:
                for _, row in scanner_df.iterrows():
                    try:
                        observation_data = self._prepare_rl_observation(row)
                        obs_array = self.rl_agent.encoder.transform_live(observation_data)
                        dummy_obs = np.repeat(obs_array, self.rl_agent.config.lookback_window, axis=0)
                        rl_action, _ = self.rl_agent.predict(dummy_obs, deterministic=True)
                        
                        rl_predictions.append({
                            'symbol': row['symbol'],
                            'rl_position': float(rl_action[0]),
                            'confidence': abs(rl_action[0]),
                            'timestamp': time.time()
                        })
                    except Exception as e:
                        logger.error(f"RL prediction failed for {row.get('symbol', 'unknown')}: {e}")
                        continue
                
                self.integration_metrics['rl_predictions_count'] += len(rl_predictions)
            
            if rl_predictions:
                await self.sync_bus.update_state('rl_predictions', rl_predictions)
            
            return {'rl_predictions': rl_predictions}
            
        except Exception as e:
            logger.error(f"RL integration update failed: {e}")
            return {}
    
    def _prepare_rl_observation(self, row: pd.Series) -> Dict[str, float]:
        """Prepare observation data for RL agent"""
        return {
            'price': row.get('price', 0.0),
            'volume': row.get('volume', 0.0),
            'momentum_short': row.get('momentum_7d', 0.0),
            'momentum_long': row.get('momentum_7d', 0.0),
            'rsi': row.get('rsi', 50.0),
            'macd': row.get('macd', 0.0),
            'bb_position': 0.5,
            'volume_ratio': 1.0,
            'composite_score': row.get('momentum_7d', 0.0),
            'trend_score': row.get('momentum_7d', 0.0),
            'confidence_score': 0.5,
            'fear_greed': 50.0,
            'btc_dominance': 50.0,
            'volatility': abs(row.get('momentum_7d', 0.0)),
            'ichimoku_bullish': 0.0,
            'vwap_bullish': 0.0,
            'ema_crossover': 0.0
        }
    
    async def train_rl_agent(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Train RL agent with historical data"""
        if not RL_AVAILABLE or not self.rl_agent:
            return {'status': 'rl_not_available'}
        
        try:
            logger.info("Starting RL agent training...")
            training_results = self.rl_agent.train(
                historical_data,
                save_path='models/integrated_rl_model'
            )
            logger.info("RL agent training completed")
            return training_results
            
        except Exception as e:
            logger.error(f"RL training failed: {e}")
            return {'status': 'training_failed', 'error': str(e)}

# Merged PerceptionLayer from mirrax
class PerceptionLayer:
    """Processes market and scanner data for downstream agents."""
    
    def __init__(self, scanner: MomentumScanner, sync_bus: 'HighPerformanceSyncBus'):
        self.scanner = scanner
        self.sync_bus = sync_bus
        self.last_update = 0.0
        logger.info("PerceptionLayer initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update perception with enhanced scanner and market data."""
        try:
            scanner_df = self.scanner.scan_market(timeframe='daily')
            if not isinstance(scanner_df, pd.DataFrame):
                logger.error("Scanner returned non-DataFrame result: %s", type(scanner_df))
                return {}
            if scanner_df.empty:
                logger.warning("Scanner returned empty DataFrame")
                return {}
            
            required_cols = ['symbol']
            price_cols = ['close', 'price']
            has_price = any(col in scanner_df.columns for col in price_cols)
            
            if not all(col in scanner_df.columns for col in required_cols) or not has_price:
                logger.error(f"Scanner DataFrame missing required columns. Available: {scanner_df.columns.tolist()}")
                return {}
            
            scanner_data = []
            for idx, row in scanner_df.iterrows():
                try:
                    row_dict = row.to_dict()
                    symbol = row_dict.get('symbol')
                    price = row_dict.get('price') or row_dict.get('close')
                    
                    if not symbol or price is None:
                        logger.warning(f"Malformed scanner row at index {idx}: missing symbol or price")
                        continue
                    
                    enhanced_data = {
                        'symbol': symbol,
                        'price': float(price),
                        'momentum_7d': row_dict.get('momentum_7d', row_dict.get('momentum_short', 0.0)),
                        'signal': row_dict.get('signal', 'Neutral'),
                        'rsi': row_dict.get('rsi', 50.0),
                        'macd': row_dict.get('macd', 0.0),
                        'timestamp': row_dict.get('timestamp', time.time()),
                        'enhanced_momentum_score': row_dict.get('enhanced_momentum_score', 0.0),
                        'reversion_probability': row_dict.get('reversion_probability', 0.0),
                        'cluster_validated': row_dict.get('cluster_validated', False),
                        'trend_formation_signal': row_dict.get('trend_formation_signal', False),
                        'volatility_regime': row_dict.get('volatility_regime', 'normal'),
                        'momentum_strength': row_dict.get('momentum_strength', 'weak')
                    }
                    
                    scanner_data.append(ScannerOutput(**enhanced_data).model_dump())
                except Exception as row_exc:
                    logger.error(f"Error processing scanner row at index {idx}: {row_exc}")
            
            await self.sync_bus.update_state('scanner_data', scanner_data)
            
            market_data = data.get('market_data', [])
            validated_data = []
            if market_data:
                for i, md in enumerate(market_data):
                    try:
                        enhanced_md = {
                            'symbol': md.get('symbol', 'UNKNOWN'),
                            'price': float(md.get('price', 1.0)),
                            'volume': float(md.get('volume', 0.0)),
                            'high': float(md.get('high', md.get('price', 1.0))),
                            'low': float(md.get('low', md.get('price', 1.0))),
                            'open': float(md.get('open', md.get('price', 1.0))),
                            'timestamp': float(md.get('timestamp', time.time())),
                            'volatility': md.get('volatility', 0.01),
                            'sentiment': md.get('sentiment', 0.0)
                        }
                        validated_data.append(MarketData(**enhanced_md).model_dump())
                    except Exception as md_exc:
                        logger.error(f"Error processing market data at index {i}: {md_exc}")
            
            await self.sync_bus.update_state('market_data', validated_data)
            
            return {'scanner_data': scanner_data, 'market_data': validated_data}
        
        except Exception as e:
            logger.error(f"PerceptionLayer update failed: {str(e)}")
            return {}

# Merged OracleEnhancedMirrorCore from mirrax (adapted to HighPerformanceSyncBus)
class OracleEnhancedMirrorCore:
    """Integrates scanner, oracle, and trade analyzer for trading decisions."""
    
    def __init__(self, sync_bus, scanner, trade_analyzer):
        self.sync_bus = sync_bus
        self.scanner = scanner
        self.trade_analyzer = trade_analyzer
        self.market_context = {
            'portfolio_value': 10000.0,
            'volatility': 0.0,
            'sentiment': 0.0
        }
        logger.info("OracleEnhancedMirrorCore initialized")
    
    async def _update_market_context(self, scanner_data: List[Dict], trades: List[Dict]) -> None:
        """Update market context."""
        try:
            scanner_df = pd.DataFrame(scanner_data)
            if not scanner_df.empty:
                self.market_context['volatility'] = float(
                    np.std(scanner_df['price'].tail(10)) /
                    np.mean(scanner_df['price'].tail(10))
                ) if len(scanner_df) >= 10 else 0.0
                self.market_context['sentiment'] = scanner_df['momentum_7d'].mean()
            
            self.market_context['portfolio_value'] = self.trade_analyzer.get_total_pnl() + 10000.0
            await self.sync_bus.update_state('market_context', self.market_context)
            logger.debug(f"Market context: volatility={self.market_context['volatility']:.4f}, portfolio={self.market_context['portfolio_value']:.2f}")
        
        except Exception as e:
            logger.error(f"Market context update failed: {str(e)}")
    
    async def enhanced_tick(self, oracle: TradingOracleEngine, trading_bot: TradingBot) -> Dict[str, Any]:
        """Process a tick integrating scanner, oracle, and bot."""
        try:
            scanner_data = await self.sync_bus.get_state('scanner_data') or []
            trades = await self.sync_bus.get_state('trades') or []
            rl_predictions = await self.sync_bus.get_state('rl_predictions') or []
            
            await self._update_market_context(scanner_data, trades)
            directives = await oracle.evaluate_trading_timelines(self.market_context, scanner_data, rl_predictions)
            
            execution_results = []
            for directive in directives:
                result = await trading_bot.process_directive(directive)
                if result:
                    execution_results.append(result)
            
            logger.info(f"Enhanced tick: {len(directives)} directives, {len(execution_results)} executed")
            return {'directives': directives, 'executions': execution_results}
        
        except Exception as e:
            logger.error(f"Enhanced tick failed: {str(e)}")
            return {'error': str(e)}

# Merged ReflectionCore from both versions (combined logic)
class ReflectionCore:
    """Reflects on performance and adjusts strategies. Merged from both versions."""
    
    def __init__(self, trade_analyzer: TradeAnalyzerAgent, strategy_trainer: StrategyTrainerAgent):
        self.trade_analyzer = trade_analyzer
        self.strategy_trainer = strategy_trainer
        self.trade_log = []
        self.pnl_curve = []
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
        logger.info("ReflectionCore initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update strategy performance. Merged logic."""
        try:
            trades = data.get('trades', [])
            for trade in trades[-10:]:
                self.strategy_trainer.update_performance(trade['strategy'], trade['pnl'])
            
            grades = self.strategy_trainer.grade_strategies()
            
            logger.debug(f"Reflection updated: {len(grades)} strategies graded")
            return {'strategy_grades': grades}
        
        except Exception as e:
            logger.error(f"ReflectionCore update failed: {str(e)}")
            return {}

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        execution_result = data.get('execution_result', {})
        market_data = data.get('market_data', {}) 

        trades = execution_result.get('trades', [])
        closed_trades = execution_result.get('closed_trades', [])

        for trade in trades: 
            if trade.get('symbol') and trade.get('entry') and trade.get('size'):
                self.trade_log.append({
                    "symbol": trade['symbol'],
                    "entry": trade['entry'],
                    "exit": None, 
                    "pnl": 0.0, 
                    "strategy": trade.get('strategy', 'UNSPECIFIED'),
                    "timestamp": trade.get('timestamp', time.time())
                })

        for closed_trade in closed_trades: 
            symbol = closed_trade.get('symbol')
            if symbol and symbol in [t['symbol'] for t in self.trade_log if t['exit'] is None]: 
                for log_entry in self.trade_log:
                    if log_entry['symbol'] == symbol and log_entry['exit'] is None:
                        log_entry['exit'] = closed_trade.get('exit', market_data.get('price')) 
                        log_entry['pnl'] = closed_trade.get('pnl', 0.0)
                        self.total_pnl += log_entry['pnl']

                        if log_entry['pnl'] > 0:
                            self.wins += 1
                            self.win_streak += 1
                            self.loss_streak = 0
                        else:
                            self.losses += 1
                            self.loss_streak += 1
                            self.win_streak = 0
                        break 

        self.trades_count = self.wins + self.losses
        if self.trades_count > 0:
            self.win_rate = self.wins / self.trades_count
        else:
            self.win_rate = 0.0

        self.pnl_curve.append(self.total_pnl)

        return {
            "trade_log": self.trade_log,
            "pnl_curve": self.pnl_curve,
            "performance_metrics": {
                "total_pnl": self.total_pnl,
                "win_rate": self.win_rate,
                "total_trades": self.trades_count,
                "wins": self.wins,
                "losses": self.losses,
                "win_streak": self.win_streak,
                "loss_streak": self.loss_streak
            }
        }

# Merged MirrorMindMetaAgent from both versions (combined logic)
class MirrorMindMetaAgent:
    """Coordinates system and triggers optimization. Merged from both versions."""

    def __init__(self, sync_bus, optimizer, trade_analyzer, arch_ctrl):
        self.sync_bus = sync_bus
        self.optimizer = optimizer
        self.trade_analyzer = trade_analyzer
        self.arch_ctrl = arch_ctrl
        self.last_optimization = 0.0
        self.optimization_interval = 3600.0
        self.max_drawdown_trigger = -0.1
        self.history = []
        self.session_grades = []
        logger.info("MirrorMindMetaAgent initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor performance and trigger optimization."""
        try:
            drawdown_stats = self.trade_analyzer.get_drawdown_stats() or {}
            current_drawdown = drawdown_stats.get('current_drawdown', 0.0)
            current_time = time.time()
            
            if not self.arch_ctrl.allow_action():
                logger.warning("Optimization skipped: emotional state unstable")
                return {'optimization_triggered': False}
            
            if (
                current_drawdown < self.max_drawdown_trigger or
                current_time - self.last_optimization > self.optimization_interval
            ):
                try:
                    # Optimization logic here
                    if hasattr(self.optimizer, 'optimize_component_safe') and callable(self.optimizer.optimize_component_safe):
                        if 'scanner' in self.optimizer.components:
                            results = self.optimizer.optimize_component_safe('scanner', self.optimizer.components['scanner'], iterations=10)
                        else:
                            logger.error("No 'scanner' component found in optimizer.")
                            results = None
                    else:
                        logger.error("Optimizer does not have an 'optimize_component_safe' method.")
                        results = None
                    self.last_optimization = current_time
                    await self.sync_bus.update_state('optimization_results', results)
                    best_scores = {k: v.get('best_score', 0.0) for k, v in self.optimizer.optimization_history.items()}
                    if any(score > 0.5 for score in best_scores.values()):
                        self.arch_ctrl.inject_emotion(confidence=min(self.arch_ctrl.confidence + 0.1, 1.0))
                    logger.info(f"Optimization triggered: {results}")
                    return {'optimization_triggered': True}
                except Exception as opt_exc:
                    logger.error(f"Optimization failed: {opt_exc}")
                    return {'optimization_triggered': False}
            return {'optimization_triggered': False}
            
            return {'optimization_triggered': False}
        
        except Exception as e:
            logger.error(f"MirrorMindMetaAgent update failed: {str(e)}")
            return {'optimization_triggered': False}

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ego_state = data.get('ego_state', {"confidence": 0.5, "stress": 0.5, "fear": 0.0})
        fear_state = data.get('fear_level', 0.0) # Assuming fear_level is directly passed
        self_state = data.get('self_state', {"drift": 0.0, "trust": 0.8, "deviations": 0})
        decision_count = data.get('decision_count', 0)
        execution_count = data.get('execution_count', 0)

        # Patch: If generate_insights is missing, use placeholder
        if hasattr(self, 'generate_insights'):
            grade, insights = self.generate_insights(ego_state, fear_state, self_state, decision_count, execution_count)
        else:
            grade, insights = "N/A", {}
        self.session_grades.append(grade)

        # Log insights and grade
        logger.info(f"[MirrorMindMetaAgent] Session Grade: {grade} | Insights: {insights}")

        return {
            'meta_insights': insights,
            'session_grade': grade
        }

    def generate_insights(self, ego_state, fear_state, self_state, decision_count, execution_count):
        insights = []
        # --- Emotional State Analysis ---
        confidence = ego_state.get("confidence", 0.5)
        stress = ego_state.get("stress", 0.5)
        fear = fear_state # Assuming fear_state is a float value
        drift = self_state.get("drift", 0.0)
        trust = self_state.get("trust", 0.8)

        if confidence > 0.7 and fear < 0.2:
            insights.append("System is confident and fearless — optimal conditions for assertive trading.")
        elif stress > 0.7:
            insights.append("High stress detected — consider reducing position size or pausing decisions.")
        elif confidence < 0.3:
            insights.append("Low confidence — system is unsure. Might be missing market clarity.")

        # --- Self Awareness ---
        if drift > 0.4: # Adjusted threshold for drift
            insights.append("Agent drift detected — internal coherence weakening significantly.")
        if trust > 0.9:
            insights.append("High self-trust indicates consistent behavior.")
        elif trust < 0.6:
            insights.append("Trust is falling — agents may be diverging in behavior.")

        # --- Action Analysis ---
        if decision_count == 0 and execution_count == 0:
            insights.append("No decisions or executions made — market may be flat or filters too strict.")
        elif execution_count > 0 and fear > 0.5:
            insights.append("Executing trades in high-fear regime — risky behavior.")

        # --- Grade the Session ---
        grade = "A"
        if trust < 0.6 or stress > 0.8:
            grade = "C"
        elif decision_count == 0 and confidence < 0.5:
            grade = "B"
        elif fear > 0.7:
            grade = "B-"

        # Store insights for summary
        self.history.append({"grade": grade, "insights": insights})

        return grade, insights

    def summarize_session(self):
        if not self.history:
             print("[MirrorMindMetaAgent] No session history to summarize.")
             return

        recent_history = self.history[-10:] # Summarize last 10 ticks
        grades = [h["grade"] for h in recent_history]
        all_insights = sum((h["insights"] for h in recent_history), [])
        most_common = Counter(all_insights).most_common(3)

        print("\n[MirrorMindMetaAgent] 🔁 Insight Summary (Last 10 Ticks):")
        print(" - Frequent Patterns:")
        if most_common:
            for insight, count in most_common:
                print(f"   • {insight} ({count} times)")
        else:
            print("   • No significant patterns detected.")

        if grades:
            # Simple grade averaging (A=4, B=3, C=2)
            grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
            numeric_grades = [grade_map.get(g[0], 0) for g in grades if g] # Use first letter of grade
            if numeric_grades:
                 avg_numeric = sum(numeric_grades) / len(numeric_grades)
                 # Map back to a grade approximation
                 if avg_numeric >= 3.5: avg_grade = "A-"
                 elif avg_numeric >= 3.0: avg_grade = "B+"
                 elif avg_numeric >= 2.5: avg_grade = "B"
                 elif avg_numeric >= 2.0: avg_grade = "C+"
                 else: avg_grade = "C"
                 print(f" - Average Grade: {avg_grade} ({avg_numeric:.2f})")
            else:
                 print(" - No valid grades to average.")
        else:
            print(" - No grades recorded.")

# Merged RiskSentinel from both versions (combined parameters)
class RiskSentinel:
    """Enforces risk limits for trading operations."""
    
    def __init__(self, max_drawdown: float = -0.2, max_position_limit: float = 0.1, max_loss_per_trade: float = -0.05, max_drawdown_pct: float = -0.10, max_open_positions: int = 5):
        self.max_drawdown = max_drawdown
        self.max_position_limit = max_position_limit
        self.max_loss_per_trade = max_loss_per_trade
        self.alerts = []
        self.max_alerts = 50
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.risk_block = False 
        logger.info("RiskSentinel initialized")
    
    async def check_risk(self, trade_analyzer: TradeAnalyzerAgent, positions: Dict[str, float], directive: Dict[str, Any]) -> bool:
        """Check risk limits."""
        try:
            drawdown_stats = trade_analyzer.get_drawdown_stats() or {}
            current_drawdown = drawdown_stats.get('current_drawdown', 0.0)
            
            if current_drawdown < self.max_drawdown:
                alert = f"Max drawdown breached: {current_drawdown:.4f} < {self.max_drawdown:.4f}"
                self.alerts.append(alert)
                logger.warning(alert)
                return False
            
            total_position = sum(abs(pos) for pos in positions.values())
            new_position = directive.get('amount', 0.0) / directive.get('price', 1.0)
            if total_position + new_position > self.max_position_limit:
                alert = f"Position limit breached: {total_position + new_position:.4f} > {self.max_position_limit:.4f}"
                self.alerts.append(alert)
                logger.warning(alert)
                return False
            
            potential_loss = -new_position * directive.get('price', 1.0) * 0.05
            portfolio_value = trade_analyzer.get_total_pnl() + 10000.0
            if potential_loss / portfolio_value < self.max_loss_per_trade:
                alert = f"Trade loss limit breached: {potential_loss/portfolio_value:.4f} < {self.max_loss_per_trade:.4f}"
                self.alerts.append(alert)
                logger.warning(alert)
                return False
            
            if len(self.alerts) > self.max_alerts:
                self.alerts.pop(0)
            
            return True
        
        except Exception as e:
            logger.error(f"Risk check failed: {str(e)}")
            return False

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        performance = data.get('performance_metrics', {})
        active_positions = data.get('active_positions', 0)
        execution_action = data.get('execution_result', {}).get('action', 'none')

        total_pnl = performance.get('total_pnl', 0.0)
        initial_capital = 1000 
        current_drawdown = total_pnl / initial_capital if initial_capital else 0.0

        if current_drawdown <= self.max_drawdown_pct:
            logger.warning(f"[RiskSentinel] Max drawdown ({self.max_drawdown_pct*100:.1f}%) reached! PnL: ${total_pnl:.2f}. Blocking new trades.")
            self.risk_block = True
        else:
             if self.risk_block and current_drawdown > self.max_drawdown_pct * 0.8: 
                 self.risk_block = False
                 logger.info(f"[RiskSentinel] Drawdown improved. Risk block lifted.")

        if active_positions >= self.max_open_positions:
            logger.warning(f"[RiskSentinel] Position limit ({self.max_open_positions}) reached! Active positions: {active_positions}. Blocking new trades.")
            self.risk_block = True

        if execution_action in ["blocked", "gated", "failed"]:
            self.risk_block = True 

        return {'risk_block': self.risk_block, 'active_positions': active_positions}

# --- Live Market Data Feed (ccxt integration) ---
import ccxt.async_support as ccxt_async

class LiveMarketDataFeed:
    """Fetches real-time market data using ccxt."""
    def __init__(self, exchange_name: str = 'binance', symbols: Optional[list] = None, api_config: Optional[dict] = None):
        self.exchange_name = exchange_name
        self.symbols = symbols or ['BTC/USDT']
        self.api_config = api_config or {}
        self.exchange = None
        logger.info(f"LiveMarketDataFeed initialized for {exchange_name}")

    async def connect(self):
        try:
            exchange_class = getattr(ccxt_async, self.exchange_name)
            self.exchange = exchange_class(self.api_config)
            await self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange_name} via ccxt.")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {str(e)}")
            self.exchange = None

    async def fetch(self, symbol: Optional[str] = None, timeframe: str = '1m') -> Optional[MarketData]:
        if not self.exchange:
            await self.connect()
        symbol = symbol or self.symbols[0]
        try:
            if self.exchange is None:
                raise RuntimeError("Exchange is not initialized")
            ticker = await self.exchange.fetch_ticker(symbol)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
            last_ohlcv = ohlcv[-1] if ohlcv else None
            if not last_ohlcv:
                return None
            open_, high, low, close, volume = last_ohlcv[1:6]
            market_data = MarketData(
                symbol=symbol,
                price=close,
                volume=volume,
                high=high,
                low=low,
                open=open_,
                timestamp=ticker['timestamp'] / 1000.0 if ticker.get('timestamp') else time.time(),
                volatility=abs(high - low) / close if close else 0.0,
                sentiment=0.0 # Placeholder, can be replaced with real sentiment analysis
            )
            return market_data
        except Exception as e:
            logger.error(f"Failed to fetch live market data: {str(e)}")
            return None

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            # logger.debug(f"Generated data: {symbol} @ {self.base_price:.2f}, vol={self.base_volume:.2f}")
            return None
        
        # Removed unreachable/invalid except block and undefined symbol

# Merged IntegratedTradingSystem from mirrax
class IntegratedTradingSystem:
    def __init__(self, scanner: MomentumScanner, rl_agent: RLTradingAgent, meta_controller: MetaController):
        self.scanner = scanner
        self.rl_agent = rl_agent
        self.meta_controller = meta_controller
        self.trading_history = []
    
    async def generate_signals(self, timeframe: str = 'daily') -> pd.DataFrame:
        scanner_results = self.scanner.scan_market(timeframe=timeframe)
        if scanner_results.empty:
            logger.warning("No scanner results available")
            return pd.DataFrame()
        
        signals_df = scanner_results.copy()
        if self.rl_agent.is_trained:
            rl_positions = []
            for _, row in scanner_results.iterrows():
                observation_data = {
                    'price': row['price'],
                    'volume': row.get('volume_usd', 0),
                    'momentum_short': row['momentum_short'],
                    'momentum_long': row['momentum_long'],
                    'rsi': row['rsi'],
                    'macd': row['macd'],
                    'bb_position': row.get('bb_position', 0.5),
                    'volume_ratio': row['volume_ratio'],
                    'composite_score': row['composite_score'],
                    'trend_score': row['trend_score'],
                    'confidence_score': row['confidence_score'],
                    'fear_greed': 50,
                    'btc_dominance': 50,
                    'volatility': row.get('volatility', 0),
                    'ichimoku_bullish': float(row.get('ichimoku_bullish', False)),
                    'vwap_bullish': float(row.get('vwap_bullish', False)),
                    'ema_crossover': float(row.get('ema_5_13_bullish', False))
                }
                obs = self.rl_agent.encoder.transform_live(observation_data)
                dummy_obs = np.repeat(obs, self.rl_agent.config.lookback_window, axis=0)
                rl_action, _ = self.rl_agent.predict(dummy_obs)
                rl_positions.append(float(rl_action[0]))
            signals_df['rl_position'] = rl_positions
        else:
            signals_df['rl_position'] = 0.0
        
        meta_decisions = []
        for _, row in signals_df.iterrows():
            decision = self.meta_controller.decide(
                rule_decision=row['signal'],
                rl_action=np.array([row['rl_position']]),
                confidence_score=row['confidence_score'],
                market_regime=self._detect_market_regime(row)
            )
            meta_decisions.append(decision)
        
        meta_df = pd.DataFrame(meta_decisions)
        signals_df = pd.concat([signals_df, meta_df], axis=1)
        return signals_df
    
    def _detect_market_regime(self, row: pd.Series) -> str:
        volatility = abs(row['momentum_short'])
        trend_strength = row['trend_score']
        if volatility > 0.05:
            return "volatile"
        elif trend_strength > 7:
            return "trending"
        return "normal"
    
    async def execute_trading_session(self, timeframe: str = 'daily', dry_run: bool = True) -> Dict[str, Any]:
        logger.info(f"Starting integrated trading session - timeframe: {timeframe}, dry_run: {dry_run}")
        try:
            signals = await self.generate_signals(timeframe)
            if signals.empty:
                return {'status': 'no_signals', 'trades': []}
            
            actionable = signals[abs(signals['final_position']) > 0.1]
            trades = []
            for _, signal in actionable.iterrows():
                trade = {
                    'symbol': signal['symbol'],
                    'signal_type': signal['signal'],
                    'rule_position': signal['rule_position'],
                    'rl_position': signal['rl_position'],
                    'final_position': signal['final_position'],
                    'confidence': signal['confidence_score'],
                    'method': signal['method'],
                    'price': signal['price'],
                    'composite_score': signal['composite_score'],
                    'timestamp': datetime.now(timezone.utc),
                    'dry_run': dry_run,
                    'executed': True
                }
                trades.append(trade)
                self.trading_history.append(trade)
            
            session_result = {
                'status': 'completed',
                'timeframe': timeframe,
                'total_signals': len(signals),
                'actionable_signals': len(actionable),
                'trades_executed': len([t for t in trades if t['executed']]),
                'trades': trades,
                'dry_run': dry_run,
                'session_timestamp': datetime.now(timezone.utc)
            }
            logger.info(f"Trading session completed: {session_result['trades_executed']} trades executed")
            return session_result
        except Exception as e:
            logger.error(f"Trading session failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report from trading history."""
        if not self.trading_history:
            return {'total_trades': 0, 'execution_rate': 0.0}
        
        total_trades = len(self.trading_history)
        executed_trades = len([t for t in self.trading_history if t.get('executed', False)])
        
        return {
            'total_trades': total_trades,
            'execution_rate': executed_trades / total_trades if total_trades > 0 else 0.0,
            'avg_confidence': np.mean([t['confidence'] for t in self.trading_history]),
            'methods_used': list({t['method'] for t in self.trading_history})
        }

# --- Emergency Controls & Audit Logging --- (from mirrorcore, unchanged)
class SecretsManager:
    """Manages API keys and sensitive configuration."""
    def __init__(self):
        self.api_key = os.environ.get("EXCHANGE_API_KEY")
        self.api_secret = os.environ.get("EXCHANGE_API_SECRET")
        self.passphrase = os.environ.get("EXCHANGE_API_PASSPHRASE")
        self.discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    def validate_credentials(self) -> bool:
        """Checks if essential credentials are present."""
        return all([self.api_key, self.api_secret, self.passphrase])

    def get_exchange_config(self) -> Dict[str, Any]:
        """Returns exchange configuration dictionary."""
        return {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'options': {
                'passphrase': self.passphrase,
                'defaultType': 'spot',
            }
        }

class EmergencyConfig(BaseModel):
    """Configuration for emergency shutdown triggers."""
    max_drawdown_pct: float = Field(default=10.0, description="Maximum allowable drawdown percentage before shutdown")
    max_latency_ms: float = Field(default=1000.0, description="Maximum acceptable API latency in milliseconds")
    max_api_errors: int = Field(default=10, description="Maximum consecutive API errors before shutdown")
    position_limit_usd: float = Field(default=100000.0, description="Maximum total USD value of open positions")
    auto_close_trades: bool = Field(default=True, description="Automatically close all open trades on emergency shutdown")

class AuditLogger:
    """Basic logger for audit events (can be extended for external services)."""
    def __init__(self):
        self.log_file = "audit.log"

    async def log_event(self, event_data: Dict[str, Any]):
        """Logs an event to a file."""
        event_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.log_file, "a") as f:
                json.dump(event_data, f)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

    async def log_error_event(self, error_data: Dict[str, Any]):
        """Logs an error event with additional context."""
        error_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        error_data['severity'] = error_data.get('severity', 'error')
        await self.log_event(error_data)

class MirrorCoreAuditLogger:
    """Integrates MirrorCore events with the AuditLogger."""
    def __init__(self, sync_bus: 'HighPerformanceSyncBus', audit_logger: 'AuditLogger'):
        self.sync_bus = sync_bus
        self.audit_logger = audit_logger

    async def log_system_event(self, event_type: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Logs a general system event."""
        await self.audit_logger.log_event({
            'event_type': event_type,
            'message': message,
            'component': 'MirrorCore',
            'context': context or {}
        })

    async def log_agent_action(self, agent_id: str, action: str, details: Dict[str, Any]):
        """Logs an action performed by an agent."""
        await self.audit_logger.log_event({
            'event_type': 'agent_action',
            'agent_id': agent_id,
            'action': action,
            'details': details,
            'component': 'Agent',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    async def log_trade_decision(self, trade_record: 'TradeRecord', decision_context: Dict[str, Any]):
        """Logs a trade decision made by the system."""
        await self.audit_logger.log_event({
            'event_type': 'trade_decision',
            'trade_details': trade_record.model_dump(),
            'decision_context': decision_context,
            'component': 'TradingEngine',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

class HeartbeatManager:
    """Manages system heartbeat and liveness checks."""
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.last_heartbeat = time.time()
        self.is_running = False

    async def start_heartbeat(self):
        """Starts the heartbeat loop."""
        self.is_running = True
        while self.is_running:
            self.last_heartbeat = time.time()
            logger.debug("System heartbeat active.")
            await asyncio.sleep(self.interval)

    def stop_heartbeat(self):
        """Stops the heartbeat loop."""
        self.is_running = False

    def check_liveness(self) -> bool:
        """Checks if the system is alive based on the last heartbeat."""
        return (time.time() - self.last_heartbeat) < (self.interval * 2)

class EmergencyController:
    """Monitors system health and triggers emergency shutdown."""
    def __init__(self, config: EmergencyConfig, exchange: Any, sync_bus: 'HighPerformanceSyncBus'):
        self.config = config
        self.exchange = exchange # Will be set after exchange is initialized
        self.sync_bus = sync_bus
        self.api_error_count = 0
        self.last_api_error_time = 0
        self.latency = 0.0
        self.is_emergency_state = False
        self.audit_logger = AuditLogger() # Use a local instance or pass one

    def set_exchange(self, exchange: Any):
        """Set the exchange object once it's initialized."""
        self.exchange = exchange

    async def check_health(self, current_pnl: float, current_positions_value: float, current_latency: float) -> bool:
        """Performs comprehensive health checks."""
        if self.is_emergency_state:
            return False # Already in emergency state

        # Check Drawdown
        # Assuming initial capital is available (e.g., from ExecutionDaemon or config)
        initial_capital = 1000.0 # Placeholder, should be dynamically obtained
        current_drawdown = (current_pnl - initial_capital) / initial_capital if initial_capital else 0.0
        if current_drawdown <= -abs(self.config.max_drawdown_pct / 100.0):
            await self._trigger_emergency_stop("MAX_DRAWDOWN_EXCEEDED", {"drawdown": current_drawdown})
            return False

        # Check Latency
        self.latency = current_latency
        if self.latency > self.config.max_latency_ms:
            await self._trigger_emergency_stop("HIGH_LATENCY", {"latency_ms": self.latency})
            return False

        # Check Position Limit
        if current_positions_value > self.config.position_limit_usd:
            await self._trigger_emergency_stop("POSITION_LIMIT_EXCEEDED", {"positions_usd": current_positions_value})
            return False

        return True

    def record_api_error(self):
        """Records an API error, potentially triggering shutdown."""
        now = time.time()
        if (now - self.last_api_error_time) > 60: # Reset count if error is older than 60 seconds
            self.api_error_count = 1
        else:
            self.api_error_count += 1
        self.last_api_error_time = now

        if self.api_error_count >= self.config.max_api_errors:
            asyncio.create_task(self._trigger_emergency_stop("MAX_API_ERRORS", {"error_count": self.api_error_count}))

    async def _trigger_emergency_stop(self, reason: str, context: Dict[str, Any]):
        """Initiates emergency shutdown sequence."""
        if self.is_emergency_state: return
        self.is_emergency_state = True
        logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED! Reason: {reason}. Context: {context}")

        await self.audit_logger.log_error_event({
            'error_type': 'emergency_shutdown',
            'reason': reason,
            'context': context,
            'component': 'EmergencyController',
            'severity': 'critical'
        })

        # Broadcast emergency stop command to all agents
        await self.sync_bus.broadcast_command('emergency_stop')

        # Close open positions if configured
        if self.config.auto_close_trades and self.exchange:
            logger.warning("Attempting to close all open positions...")
            # This logic should ideally call a method in ExecutionDaemon or similar
            # For now, we just log the intent. A proper implementation would fetch
            # open positions from the exchange or internal state and close them.
            await self.audit_logger.log_event({
                'event_type': 'emergency_action',
                'action': 'close_all_positions',
                'message': 'Initiated closing of all open trades on emergency.',
                'context': {'reason': reason},
                'component': 'EmergencyController',
                'severity': 'warning'
            })

    async def reset_emergency_state(self):
        """Resets emergency state if conditions improve."""
        # This should be called periodically or based on external signals
        self.is_emergency_state = False
        logger.info("Emergency state reset. System is operational again.")
        await self.audit_logger.log_event({
            'event_type': 'emergency_state_reset',
            'message': 'Emergency shutdown state has been reset.',
            'component': 'EmergencyController',
            'severity': 'info'
        })

def load_secrets_from_env_file(env_file: str = ".env"):
    """Loads environment variables from a .env file if it exists."""
    if os.path.exists(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")

# --- Market Data Pipeline (Separated from Agent State) --- (from mirrorcore, unchanged)
class MarketDataProcessor:
    """Validates and cleans raw market data"""

    def validate(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and validate incoming market data"""
        try:
            # Basic validation
            if not raw_data or 'price' not in raw_data:
                return {}

            # Basic numeric values
            validated_data = {
                'symbol': raw_data.get('symbol', 'UNKNOWN'),
                'price': float(raw_data['price']),
                'volume': float(raw_data.get('volume', 0)),
                'high': float(raw_data.get('high', raw_data['price'])),
                'low': float(raw_data.get('low', raw_data['price'])),
                'open': float(raw_data.get('open', raw_data['price'])),
                'timestamp': raw_data.get('timestamp', time.time()),
                'volatility': raw_data.get('volatility'),
                'sentiment': raw_data.get('sentiment', 0.0)
            }

            return validated_data

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Market data validation failed: {e}")
            return {}

class DataPipeline:
    """Separated market data pipeline"""

    def __init__(self, syncbus):
        self.syncbus = syncbus
        self.market_processor = MarketDataProcessor()
        self.tick_counter = 0

    async def process_tick(self, raw_market_data):
        """Process market tick separately from agent state"""
        # Step 1: Clean & validate market data
        clean_data = self.market_processor.validate(raw_market_data)

        if clean_data is None:
            return

        # Step 2: Update market state (separate from agent state)
        market_state = {
            'tick_id': self.tick_counter,
            'timestamp': time.time(),
            'market_data': clean_data,
            'data_sources': ['scanner', 'exchange']
        }

        # Step 3: Broadcast to SyncBus
        await self.syncbus.broadcast_market_update(market_state)

        self.tick_counter += 1

# --- Data Categorization Framework --- (from mirrorcore, unchanged)
class DataCategorizer:
    """Categorize data for different agent types"""

    def process(self, raw_data):
        """Categorize data for different agent types"""
        if not raw_data:
            return {}

        return {
            'technical': self._extract_technical(raw_data),
            'fundamental': self._extract_fundamental(raw_data),
            'sentiment': self._extract_sentiment(raw_data),
            'risk': self._extract_risk_signals(raw_data),
            'onchain': self._extract_onchain_metrics(raw_data),
            'macro': self._extract_macro_indicators(raw_data)
        }

    def _extract_technical(self, data):
        """Price, volume, momentum indicators"""
        market_data = data.get('market_data', {})
        return {
            'price': market_data.get('price', 0),
            'volume': market_data.get('volume', 0),
            'volatility': market_data.get('volatility', 0),
            'momentum': market_data.get('momentum_7d', 0)
        }

    def _extract_fundamental(self, data):
        """Placeholder for fundamental data extraction"""
        return {}

    def _extract_sentiment(self, data):
        """Social media sentiment, fear/greed index"""
        return {
            'sentiment': data.get('market_data', {}).get('sentiment', 0.0),
            'fear_greed': 50.0  # Default neutral
        }

    def _extract_risk_signals(self, data):
        """Risk management signals"""
        market_data = data.get('market_data', {})
        return {
            'volatility': market_data.get('volatility', 0),
            'volume_anomaly': market_data.get('volume', 0) > 100000
        }

    def _extract_onchain_metrics(self, data):
        """Placeholder for on-chain data extraction"""
        return {}

    def _extract_macro_indicators(self, data):
        """Placeholder for macro indicator extraction"""
        return {}


# --- High-Performance SyncBus with Fault Resistance --- (from mirrorcore, unchanged)
class HighPerformanceSyncBus:

    def _ensure_agent_health(self, agent_id: str):
        if agent_id not in self.agent_health:
            self.agent_health[agent_id] = {
                'success_count': 0,
                'failure_count': 0,
                'last_update': None
            }
    """Enhanced SyncBus with delta updates, interest-based routing, and fault resistance"""

    def __init__(self):
        self.agent_states = {}
        self.subscriptions = {}  # Who wants what updates
        self.update_queues = {}  # Per-agent async queues
        self.agent_health = {}
        self.circuit_breakers = {}
        self.lock = asyncio.Lock()
        self.tick_count = 0

        # Data categorization
        self.data_categorizer = DataCategorizer()

        # Performance tracking
        self.performance_metrics = {
            'total_ticks': 0,
            'failed_updates': 0,
            'circuit_breaker_activations': 0,
            'agent_restarts': 0
        }

        # Global state (enhanced structure)
        self.global_state = {
            'market_data': [],
            'scanner_data': [],
            'trades': [],
            'system_performance': {
                'pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0
            },
            'emotional_state': {},
            'market_context': {},
            'optimization_results': {},
            'strategy_grades': {},
            'agent_performance': {},
            'system_health': {}
        }

        logger.info("High-Performance SyncBus initialized with fault resistance")

    async def register_agent(self, agent_id: str, interests: Optional[List[str]] = None):
        """Register agent with specific data interests"""
        if agent_id in self.subscriptions: # Avoid re-registering
            return
        self.subscriptions[agent_id] = interests if interests is not None else ['all']
        self.update_queues[agent_id] = asyncio.Queue(maxsize=100)
        self.agent_health[agent_id] = {'success_count': 0, 'failure_count': 0, 'last_update': time.time()}
        self.circuit_breakers[agent_id] = {'is_open': False, 'failure_count': 0, 'last_failure': 0}

        logger.info(f"Registered agent {agent_id} with interests: {interests}")

    def attach(self, name: str, agent: Any) -> None:
        """Attach an agent to the bus with automatic registration"""
        # Ensure agent has a data_interests attribute, default to ['all']
        interests = getattr(agent, 'data_interests', ['all'])
        asyncio.create_task(self.register_agent(name, interests))
        self.agent_states[name] = {}
        # Dynamically attach agent as an attribute for easy access if needed
        setattr(self, name, agent)
        logger.info(f"Attached agent: {name}")

    def detach(self, name: str) -> None:
        """Detach an agent from the bus"""
        if name in self.subscriptions:
            del self.subscriptions[name]
        if name in self.update_queues:
            del self.update_queues[name]
        if name in self.agent_health:
            del self.agent_health[name]
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
        if name in self.agent_states:
            del self.agent_states[name]
        # Remove agent attribute if it exists
        if hasattr(self, name):
            delattr(self, name)
        logger.info(f"Detached agent: {name}")

    async def update_agent_state(self, agent_id: str, state_delta: Dict[str, Any]):
        """Only update what changed (delta updates) with circuit breaker protection"""
        # Check circuit breaker
        if self._is_circuit_open(agent_id):
            logger.warning(f"Agent {agent_id} circuit open, ignoring update")
            return

        self._ensure_agent_health(agent_id)
        try:
            async with self.lock:
                if agent_id not in self.agent_states:
                    self.agent_states[agent_id] = {}

                # Merge delta instead of full state
                self.agent_states[agent_id].update(state_delta)

                # Update health tracking
                self.agent_health[agent_id]['success_count'] += 1
                self.agent_health[agent_id]['last_update'] = time.time()

                # Only notify interested agents
                await self._notify_interested_agents(agent_id, state_delta)

        except Exception as e:
            await self._record_failure(agent_id, e)
            if self._should_open_circuit(agent_id):
                await self._isolate_agent(agent_id)

    async def get_relevant_states(self, requesting_agent_id: str) -> Dict[str, Any]:
        """Agent gets only states it cares about"""
        interests = self.subscriptions.get(requesting_agent_id, ['all'])
        relevant_states = {}

        async with self.lock:
            for agent_id, state in self.agent_states.items():
                if self._is_relevant(interests, agent_id, state):
                    relevant_states[agent_id] = state

        return relevant_states

    async def broadcast_market_update(self, market_state: Dict[str, Any]):
        """Broadcast market update to all interested agents"""
        categorized_data = self.data_categorizer.process(market_state)

        # Update global market state
        async with self.lock:
            self.global_state['market_data'].append(market_state)
            if len(self.global_state['market_data']) > 1000:
                self.global_state['market_data'] = self.global_state['market_data'][-1000:]

        # Notify agents based on their interests
        for agent_id, interests in self.subscriptions.items():
            if not self._is_circuit_open(agent_id):
                relevant_data = self._filter_data_by_interests(categorized_data, interests)
                if relevant_data:
                    try:
                        # Add market_data to the relevant_data for agents that need it directly
                        if 'market_data' in market_state and ('market_data' in interests or 'all' in interests):
                            relevant_data['market_data'] = market_state['market_data']
                        await self.update_queues[agent_id].put(relevant_data)
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for agent {agent_id}, dropping update")

    async def tick(self) -> None:
        """Enhanced tick processing with fault isolation"""
        self.tick_count += 1
        self.performance_metrics['total_ticks'] += 1

        logger.debug(f"Processing tick {self.tick_count}")

        tasks = []
        try:
            # Process agents with timeout and isolation
            # Iterate over a copy of agent_states keys to avoid issues if an agent is detached during processing
            for agent_id in list(self.agent_states.keys()):
                if not self._is_circuit_open(agent_id):
                    task = asyncio.create_task(
                        self._process_agent_safely(agent_id),
                        name=f"agent_{agent_id}"
                    )
                    tasks.append(task)


            # Execute with timeout
            if tasks:
                # Use asyncio.wait to apply a timeout to all tasks
                done, pending = await asyncio.wait(tasks, timeout=10)
                results = []
                # Collect results from completed tasks
                for task in tasks:
                    if task in done:
                        try:
                            results.append(task.result())
                        except Exception as e:
                            results.append(e)
                    else:
                        # Mark as timeout for pending tasks
                        results.append(asyncio.TimeoutError("Agent processing timed out"))
                await self._process_tick_results(results, tasks)

            # Update system health metrics
            await self._update_system_health()

        except Exception as e:
            logger.error(f"Global tick failed: {e}")
            self.performance_metrics['failed_updates'] += 1

    def _is_circuit_open(self, agent_id: str) -> bool:
        cb = self.circuit_breakers.get(agent_id, {})
        if cb.get('is_open', False):
            # Simple half-open logic after cooldown
            if time.time() - cb.get('last_failure', 0) > 60:  # 1 min cooldown
                cb['is_open'] = False
                cb['failure_count'] = 0
                return False
            return True
        return False

    def _should_open_circuit(self, agent_id: str) -> bool:
        health = self.agent_health.get(agent_id, {})
        cb = self.circuit_breakers.get(agent_id, {})
        if health.get('failure_count', 0) > 5:  # Threshold
            cb['is_open'] = True
            cb['last_failure'] = time.time()
            self.performance_metrics['circuit_breaker_activations'] += 1
            return True
        return False

    async def _isolate_agent(self, agent_id: str):
        """Isolate failed agent"""
        logger.warning(f"Isolating failed agent: {agent_id}")
        # Could add more logic like restarting or notifying

    async def _notify_interested_agents(self, source_agent: str, delta: Dict[str, Any]):
        """Route updates only to interested parties"""
        for agent_id, interests in self.subscriptions.items():
            if self._is_relevant(interests, source_agent, delta):
                try:
                    await self.update_queues[agent_id].put({
                        'source': source_agent,
                        'delta': delta,
                        'timestamp': time.time()
                    })
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for {agent_id}, dropping delta from {source_agent}")

    def _is_relevant(self, interests: List[str], source: str, data: Dict[str, Any]) -> bool:
        """Check if update is relevant to agent's interests"""
        if 'all' in interests:
            return True
        # Check if source or data keys match interests
        if source in interests:
            return True
        return any(key in interests for key in data.keys())

    def _filter_data_by_interests(self, data: Dict[str, Any], interests: List[str]) -> Dict[str, Any]:
        """Filter data based on agent interests"""
        if 'all' in interests:
            return data
        return {k: v for k, v in data.items() if k in interests}

    async def _process_agent_safely(self, agent_id: str):
        """Process agent update with error isolation"""
        try:
            agent = getattr(self, agent_id, None)
            if not agent:
                return None

            # Get relevant state for agent
            relevant_state = await self.get_relevant_states(agent_id)
            
            # Update agent
            if hasattr(agent, 'update'):
                return await agent.update(relevant_state)
            return None

        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}")
            await self._record_failure(agent_id, e)
            raise

    async def _record_failure(self, agent_id: str, error: Exception):
        """Record agent failure"""
        self._ensure_agent_health(agent_id)
        health = self.agent_health[agent_id]
        health['failure_count'] += 1
        health['last_update'] = time.time()
        self.performance_metrics['failed_updates'] += 1
        logger.error(f"Agent {agent_id} failed: {error}")

    async def _process_tick_results(self, results: List, tasks: List):
        """Process results from agent tasks"""
        for result, task in zip(results, tasks):
            if isinstance(result, Exception):
                agent_id = task.get_name().replace('agent_', '')
                logger.error(f"Agent {agent_id} task failed: {result}")
            elif isinstance(result, dict):
                agent_id = task.get_name().replace('agent_', '')
                await self.update_agent_state(agent_id, result)

    async def _update_system_health(self):
        """Update global system health metrics"""
        async with self.lock:
            self.global_state['system_health'] = {
                'active_agents': len([a for a in self.agent_health if not self._is_circuit_open(a)]),
                'failed_agents': sum(1 for cb in self.circuit_breakers.values() if cb['is_open']),
                'total_ticks': self.tick_count,
                'efficiency': (self.tick_count - self.performance_metrics['failed_updates']) / max(1, self.tick_count)
            }

    async def broadcast_command(self, command: str, payload: Optional[Dict] = None):
        """Broadcast system-wide commands like emergency_stop"""
        for agent_id in list(self.subscriptions.keys()):
            if not self._is_circuit_open(agent_id):
                try:
                    await self.update_queues[agent_id].put({
                        'type': 'command',
                        'command': command,
                        'payload': payload or {},
                        'timestamp': time.time()
                    })
                except asyncio.QueueFull:
                    logger.warning(f"Command queue full for {agent_id}")

    async def get_state(self, key: str) -> Any:
        """Get global state value"""
        async with self.lock:
            return self.global_state.get(key)

    async def update_state(self, key: str, value: Any):
        """Update global state"""
        async with self.lock:
            self.global_state[key] = value

    async def get_all_states(self) -> Dict[str, Any]:
        """Get all agent states"""
        async with self.lock:
            # Return a deep copy to prevent modification of internal states
            import copy
            return copy.deepcopy(self.agent_states)

    async def send_command(self, agent_id: str, command: str, params: Dict[str, Any] = None):
        """Send command to specific agent"""
        if agent_id not in self.update_queues:
            logger.warning(f"Agent {agent_id} not found for command '{command}'")
            return

        try:
            command_msg = {'type': 'command', 'command': command, 'params': params or {}, 'timestamp': time.time()}
            await self.update_queues[agent_id].put(command_msg)
        except asyncio.QueueFull:
            logger.warning(f"Command queue full for agent {agent_id}")

    async def broadcast_command(self, command: str, params: Dict[str, Any] = None):
        """Broadcast command to all agents"""
        for agent_id in list(self.agent_states.keys()): # Iterate over a copy of keys
            await self.send_command(agent_id, command, params)


# --- Enhanced Console Monitor ---
class ConsoleMonitor:
    """Enhanced console monitoring with agent grid display"""

    def __init__(self, syncbus: HighPerformanceSyncBus):
        self.syncbus = syncbus

    async def display_agent_grid(self):
        """Show agents in a formatted grid"""
        states = await self.syncbus.get_all_states()
        agent_health = self.syncbus.agent_health
        circuit_breakers = self.syncbus.circuit_breakers

        table = []
        for agent_id, state in states.items():
            health_info = agent_health.get(agent_id, {})
            success = health_info.get('success_count', 0)
            failures = health_info.get('failure_count', 0)
            health_score = success / max(1, failures + success) if (failures + success) > 0 else 1.0

            breaker_status = circuit_breakers.get(agent_id, {}).get('is_open', False)

            table.append([
                agent_id[:12],  # Agent name
                state.get('status', 'UNK')[:8],  # Status
                f"{state.get('confidence', 0):.2f}",  # Confidence
                state.get('last_signal', 'NONE')[:8],  # Last signal
                f"{state.get('pnl', 0):+.2f}",  # PnL
                self._health_indicator(health_score),  # Health
                str(breaker_status)[:5]  # Circuit status
            ])

        # Clear screen and display
        os.system('clear' if os.name == 'posix' else 'cls')
        print("="*80)
        print("MIRRORCORE-X AGENT STATUS DASHBOARD")
        print("="*80)
        print(tabulate(table, headers=['Agent', 'Status', 'Conf', 'Signal', 'PnL', 'Health', 'Circuit']))

        # System health summary
        system_health = await self.syncbus.get_state('system_health')
        if system_health:
            print(f"\nSystem Health: {system_health['healthy_agents']}/{system_health['total_agents']} agents healthy")
            print(f"Health Score: {system_health['health_score']:.2f}")
            print(f"Performance Metrics: Ticks={system_health['performance_metrics']['total_ticks']}, "
                  f"Failed Updates={system_health['performance_metrics']['failed_updates']}, "
                  f"CB Activations={system_health['performance_metrics']['circuit_breaker_activations']}, "
                  f"Agent Restarts={system_health['performance_metrics']['agent_restarts']}")


        print("="*80)

    def _health_indicator(self, health_score: float) -> str:
        """Convert health score to emoji indicator"""
        if health_score > 0.8:
            return "🟢"
        elif health_score > 0.5:
            return "🟡"
        else:
            return "🔴"

# --- Command Interface ---
class CommandInterface:
    """Command interface for human-system interaction"""

    def __init__(self, syncbus: HighPerformanceSyncBus):
        self.syncbus = syncbus
        self.command_handlers = {
            'status': self._get_status,
            'pause_agent': self._pause_agent,
            'resume_agent': self._resume_agent,
            'emergency_stop': self._emergency_stop,
            'health': self._get_health_report,
            'restart_agent': self._restart_agent,
            'list_agents': self._list_agents,
            'send_command': self._send_command_to_agent # Added handler
        }

    async def handle_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle command execution"""
        params = params or {}
        handler = self.command_handlers.get(command)

        if handler:
            try:
                result = await handler(params)
                return result
            except Exception as e:
                logger.error(f"Error executing command '{command}': {str(e)}")
                return {"error": f"Error executing command: {str(e)}"}

        return {"error": f"Unknown command: {command}"}

    async def _get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status"""
        system_health = await self.syncbus.get_state('system_health')
        return {
            "status": "operational",
            "tick_count": self.syncbus.tick_count,
            "system_health": system_health,
            "performance_metrics": self.syncbus.performance_metrics
        }

    async def _pause_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pause specific agent"""
        agent_id = params.get('agent_id')
        if not agent_id:
            return {"error": "agent_id parameter required"}

        await self.syncbus.send_command(agent_id, 'pause')
        return {"status": f"Pause command sent to agent {agent_id}"}

    async def _resume_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resume specific agent"""
        agent_id = params.get('agent_id')
        if not agent_id:
            return {"error": "agent_id parameter required"}

        await self.syncbus.send_command(agent_id, 'resume')
        return {"status": f"Resume command sent to agent {agent_id}"}

    async def _emergency_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency stop all trading"""
        await self.syncbus.broadcast_command('emergency_stop')
        return {"status": "Emergency stop activated for all agents"}

    async def _get_health_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed health report"""
        return {
            "agent_health": self.syncbus.agent_health,
            "circuit_breakers": self.syncbus.circuit_breakers,
            "performance_metrics": self.syncbus.performance_metrics
        }

    async def _restart_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Force restart agent circuit breaker"""
        agent_id = params.get('agent_id')
        if not agent_id:
            return {"error": "agent_id parameter required"}

        # Ensure the agent is actually registered before attempting restart
        if agent_id not in self.syncbus.agent_states:
             return {"error": f"Agent {agent_id} not found."}

        await self.syncbus._attempt_restart(agent_id, delay=1)
        return {"status": f"Restart initiated for agent {agent_id}"}

    async def _list_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all registered agents"""
        agents = []
        for agent_id in self.syncbus.agent_states.keys():
            health = self.syncbus.agent_health.get(agent_id, {})
            circuit = self.syncbus.circuit_breakers.get(agent_id, {})
            agents.append({
                "agent_id": agent_id,
                "interests": self.syncbus.subscriptions.get(agent_id, []),
                "health": health,
                "circuit_open": circuit.get('is_open', False)
            })

        return {"agents": agents}

    async def _send_command_to_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a custom command to a specific agent"""
        agent_id = params.get('agent_id')
        command = params.get('command')
        command_params = params.get('command_params', {})

        if not agent_id or not command:
            return {"error": "agent_id and command parameters are required"}

        await self.syncbus.send_command(agent_id, command, command_params)
        return {"status": f"Command '{command}' sent to agent {agent_id} with params {command_params}"}


# --- Enhanced Base Agent Class ---
class MirrorAgent:
    """Enhanced base agent class with data interests and command processing"""

    def __init__(self, name: str, data_interests: List[str] = None):
        self.name = name
        self.data_interests = data_interests or ['all']
        self.last_update_time = time.time() # Renamed to avoid conflict with syncbus.tick_count
        self.memory = deque(maxlen=1000)
        self.is_paused = False
        self.command_queue = deque()
        self.syncbus = None # Will be set by SyncBus attach

    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced update with command processing"""
        # Process any pending commands first
        await self._process_commands()

        if self.is_paused:
            return {}

        # Store relevant data in memory
        self.store_memory(data)

        # Subclasses implement specific logic
        result = await self._process_data(data)

        self.last_update_time = time.time()
        return result

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclasses"""
        # Default implementation returns empty dict
        return {}

    async def _process_commands(self):
        """Process pending commands"""
        while self.command_queue:
            command_msg = self.command_queue.popleft()
            command = command_msg.get('command')

            if command == 'pause':
                self.is_paused = True
                logger.info(f"Agent {self.name} paused")
            elif command == 'resume':
                self.is_paused = False
                logger.info(f"Agent {self.name} resumed")
            elif command == 'emergency_stop':
                self.is_paused = True
                logger.warning(f"Agent {self.name} emergency stopped")
            elif command == 'syncbus_command': # Handle commands sent via syncbus
                 # Allow agents to handle syncbus commands if needed
                 await self._handle_syncbus_command(command_msg.get('params', {}))

    async def _handle_syncbus_command(self, params: Dict[str, Any]):
         """Handle commands coming from the SyncBus (e.g., through command interface)"""
         # This method can be overridden by subclasses to handle specific commands
         pass

    def store_memory(self, item: Any):
        """Store item in agent's memory with timestamp"""
        self.memory.append({
            'timestamp': time.time(),
            'data': item
        })

# --- System Creation Function ---
async def create_mirrorcore_system(dry_run: bool = True, use_testnet: bool = True, 
                                     enable_oracle: bool = True, 
                                     enable_bayesian: bool = True,
                                     enable_imagination: bool = True) -> Tuple[HighPerformanceSyncBus, Dict[str, Any]]:
    """Create MirrorCore-X system with emergency controls, audit logging, and enhancement engines"""

    # Load secrets from environment
    load_secrets_from_env_file()
    secrets_manager = SecretsManager()

    # Validate credentials
    if not secrets_manager.validate_credentials():
        logger.warning("Some credentials missing - running in limited mode")

    # Initialize audit logging
    audit_logger = AuditLogger()
    await audit_logger.log_error_event({
        'error_type': 'system_startup',
        'message': 'MirrorCore-X system starting with enhancements',
        'component': 'main',
        'severity': 'info',
        'context': {
            'dry_run': dry_run, 
            'testnet': use_testnet,
            'oracle': enable_oracle,
            'bayesian': enable_bayesian,
            'imagination': enable_imagination
        }
    })

    # Create high-performance SyncBus
    sync_bus = HighPerformanceSyncBus()

    # Initialize data pipeline
    data_pipeline = DataPipeline(sync_bus)

    # Initialize emergency controls
    emergency_config = EmergencyConfig(
        max_drawdown_pct=15.0,
        max_latency_ms=500.0,
        max_api_errors=5,
        position_limit_usd=50000.0
    )

    # Create emergency controller (exchange will be set later)
    emergency_controller = EmergencyController(emergency_config, None, sync_bus)

    # Initialize heartbeat manager
    heartbeat_manager = HeartbeatManager()
    asyncio.create_task(heartbeat_manager.start_heartbeat())

    # Initialize MirrorCore audit logger
    mirror_audit = MirrorCoreAuditLogger(sync_bus, audit_logger)

    try:
        if use_testnet:
            # Use the correct testnet URL for Binance testnet
            exchange = ccxt.binance()
            exchange.enableRateLimit = True
            exchange.options = {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
            exchange.apiKey = secrets_manager.api_key or ''
            exchange.secret = secrets_manager.api_secret or ''
            exchange.password = secrets_manager.passphrase or ''
            exchange.urls = {'api': {'public': 'https://testnet.binance.vision/api', 'private': 'https://testnet.binance.vision/api'}}
            logger.warning("Using Binance testnet.")
        else:
            # Build config for live trading
            creds = secrets_manager.get_exchange_config()
            exchange = ccxt.binance()
            exchange.enableRateLimit = True
            exchange.options = {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
            exchange.apiKey = creds.get('apiKey', '')
            exchange.secret = creds.get('secret', '')
            exchange.password = creds.get('passphrase', '') or creds.get('options', {}).get('passphrase', '')

        if dry_run:
            logger.warning("Using dry_run mode. Exchange interactions will be simulated.")
            # Use a mock or simulated exchange if available, otherwise initialize with config
            exchange.enableRateLimit = True
        else:
            if not secrets_manager.validate_credentials():
                raise ValueError("API credentials missing for live trading. Please set EXHANGE_API_KEY, EXCHANGE_API_SECRET, EXCHANGE_API_PASSPHRASE.")
            exchange.enableRateLimit = True
        # Instantiate agents that need the exchange or are required for optimizer
        scanner = MomentumScanner(exchange=exchange)
        strategy_trainer = StrategyTrainerAgent()
        trade_analyzer = TradeAnalyzerAgent()
        arch_ctrl = ARCH_CTRL()


        ego_processor = EgoProcessor()
        fear_analyzer = FearAnalyzer()
        self_awareness_agent = SelfAwarenessAgent()
        decision_mirror = DecisionMirror()
        execution_daemon = ExecutionDaemon(exchange=exchange, dry_run=dry_run)
        reflection_core = ReflectionCore(trade_analyzer, strategy_trainer)
        mirror_mind_meta_agent = MirrorMindMetaAgent(sync_bus, ProductionSafeMirrorOptimizer({}), trade_analyzer, arch_ctrl)
        meta_agent = MetaAgent(ego_processor, fear_analyzer, self_awareness_agent, decision_mirror)
        risk_sentinel = RiskSentinel()
        # Remove MarketDataGenerator, use LiveMarketDataFeed
        market_data_feed = LiveMarketDataFeed()
        perception = PerceptionLayer(scanner, sync_bus)
        rl_integration = RLIntegrationLayer(sync_bus, scanner)
        rl_config = RLConfig()
        rl_agent = RLTradingAgent(algorithm='PPO', config=rl_config)
        meta_controller = MetaController(strategy="confidence_blend")
        integrated_trading_system = IntegratedTradingSystem(scanner, rl_agent, meta_controller)
        oracle = TradingOracleEngine(strategy_trainer, sync_bus)
        mirror_core = OracleEnhancedMirrorCore(sync_bus, scanner, trade_analyzer)

        # Import and create comprehensive optimizer
        # from mirror_optimizer import ProductionSafeMirrorOptimizer
        production_optimizer = ProductionSafeMirrorOptimizer

        # Collect all components for the optimizer
        all_components = {
            'scanner': scanner,
            'strategy_trainer': strategy_trainer,
            'trade_analyzer': trade_analyzer,
            'arch_ctrl': arch_ctrl,
            'ego_processor': ego_processor,
            'fear_analyzer': fear_analyzer,
            'self_awareness': self_awareness_agent,
            'decision_mirror': decision_mirror,
            'execution_daemon': execution_daemon,
            'reflection_core': reflection_core,
            'mirror_mind_meta': mirror_mind_meta_agent,
            'meta_agent': meta_agent,
            'risk_sentinel': risk_sentinel,
            'perception': perception,
            'rl_integration': rl_integration,
            'mirror_core': mirror_core
        }

        # Create comprehensive optimizer
        comprehensive_optimizer = ProductionSafeMirrorOptimizer(all_components)

        # Attach agents to SyncBus
        sync_bus.attach('scanner', scanner)
        sync_bus.attach('strategy_trainer', strategy_trainer)
        sync_bus.attach('trade_analyzer', trade_analyzer)
        sync_bus.attach('arch_ctrl', arch_ctrl)
        sync_bus.attach('ego_processor', ego_processor)
        sync_bus.attach('fear_analyzer', fear_analyzer)
        sync_bus.attach('self_awareness', self_awareness_agent)
        sync_bus.attach('decision_mirror', decision_mirror)
        sync_bus.attach('execution_daemon', execution_daemon)
        sync_bus.attach('reflection_core', reflection_core)
        sync_bus.attach('mirror_mind_meta', mirror_mind_meta_agent)
        sync_bus.attach('meta_agent', meta_agent)
        sync_bus.attach('risk_sentinel', risk_sentinel)
        sync_bus.attach('market_data_feed', market_data_feed)
        sync_bus.attach('perception', perception)
        sync_bus.attach('rl_integration', rl_integration)
        sync_bus.attach('integrated_trading_system', integrated_trading_system)
        sync_bus.attach('mirror_core', mirror_core)
        sync_bus.attach('comprehensive_optimizer', comprehensive_optimizer)

        # Register strategies (assuming these agents exist)
        # Register core strategies
        strategy_trainer.register_strategy("UT_BOT", UTSignalAgent())
        strategy_trainer.register_strategy("GRADIENT_TREND", GradientTrendAgent())
        strategy_trainer.register_strategy("VBSR", SupportResistanceAgent())

        # Register all additional strategies
        strategy_trainer.register_strategy("MEAN_REVERSION", MeanReversionAgent())
        strategy_trainer.register_strategy("MOMENTUM_BREAKOUT", MomentumBreakoutAgent())
        strategy_trainer.register_strategy("VOLATILITY_REGIME", VolatilityRegimeAgent())
        strategy_trainer.register_strategy("PAIRS_TRADING", PairsTradingAgent())
        strategy_trainer.register_strategy("ANOMALY_DETECTION", AnomalyDetectionAgent())
        strategy_trainer.register_strategy("SENTIMENT_MOMENTUM", SentimentMomentumAgent())
        strategy_trainer.register_strategy("REGIME_CHANGE", RegimeChangeAgent())

        logger.info("MirrorCore-X system created with enhanced SyncBus and merged features")

        return sync_bus, {
            'sync_bus': sync_bus,
            'data_pipeline': data_pipeline,
            'market_scanner': scanner,
            'strategy_trainer': strategy_trainer,
            'trade_analyzer': trade_analyzer,
            'execution_daemon': execution_daemon,
            'risk_sentinel': risk_sentinel,
            'arch_ctrl': arch_ctrl,
            'comprehensive_optimizer': comprehensive_optimizer,
            'emergency_controller': emergency_controller,
            'heartbeat_manager': heartbeat_manager,
            'audit_logger': audit_logger,
            'mirror_audit': mirror_audit,
            'secrets_manager': secrets_manager,
            'exchange': exchange,
            'market_data_feed': market_data_feed,
            'perception': perception,
            'rl_integration': rl_integration,
            'integrated_trading_system': integrated_trading_system,
            'mirror_core': mirror_core
        }

    except Exception as e:
        logger.error(f"Failed to create MirrorCore system: {e}")
        await audit_logger.log_error_event({
            'error_type': 'system_initialization_failed',
            'message': f"Failed to create MirrorCore system: {e}",
            'component': 'main',
            'severity': 'critical',
            'context': {'dry_run': dry_run, 'testnet': use_testnet}
        })
        raise

# --- Main Execution (merged demo session) ---
if __name__ == "__main__":
    async def main():
        # Set dry_run to True for simulation, False for live trading
        # Set use_testnet to True to use the exchange's test network
        sync_bus, components = await create_mirrorcore_system(dry_run=True, use_testnet=False)

        data_pipeline = components['data_pipeline']
        exchange = components['exchange']
        emergency_controller = components['emergency_controller']
        heartbeat_manager = components['heartbeat_manager']
        mirror_audit = components['mirror_audit']
        secrets_manager = components['secrets_manager']
        integrated_trading_system = components['integrated_trading_system']
        scanner = components['market_scanner']
        trade_analyzer = components['trade_analyzer']

        # Mock market data generation for simulation (merged with run_demo_session logic)
        async def mock_market_data_generator(ticks: int = 100, tick_interval: float = 0.1):
            symbol = "BTC/USDT"
            price = 30000.0
            for i in range(ticks):
                price += np.random.normal(0, 100) * (0.5 if sync_bus.tick_count % 2 == 0 else 1.0) # Simulate price fluctuation
                volatility = max(0.01, abs(np.random.normal(0, 0.005)))
                sentiment = np.random.uniform(-0.3, 0.3) # Simulate sentiment
                market_update = {
                    "symbol": symbol,
                    "price": price,
                    "volume": np.random.randint(1000, 5000),
                    "high": price + np.random.uniform(0, 50),
                    "low": price - np.random.uniform(0, 50),
                    "open": price + np.random.normal(0, 20),
                    "timestamp": time.time(),
                    "volatility": volatility,
                    "sentiment": sentiment
                }
                # Scanner data simulation
                scanner_update = {
                    "symbol": symbol,
                    "momentum_7d": np.random.uniform(-0.1, 0.1),
                    "signal": np.random.choice(["bullish", "bearish", "neutral"]),
                    "rsi": np.random.uniform(30, 70),
                    "macd": np.random.uniform(-50, 50),
                    "timestamp": time.time(),
                    "price": price
                }

                # Process market data through pipeline
                await data_pipeline.process_tick(market_update)

                # Simulate scanner output update (manually put into syncbus)
                await sync_bus.update_agent_state('scanner', {'scanner_data': scanner_update}) # Assuming scanner agent can receive this

                # Trigger a syncbus tick
                await sync_bus.tick()

                # Merged demo logic
                session_result = await integrated_trading_system.execute_trading_session(timeframe='daily', dry_run=True)
                if session_result['status'] == 'completed':
                    trades = session_result['trades']
                    for trade in trades:
                        await sync_bus.update_state('trades', [trade])
                        logger.info(f"Trade executed: {trade['symbol']}, Position: {trade['final_position']:.2f}, Method: {trade['method']}")

                await asyncio.sleep(tick_interval)
                
                if i % 20 == 0:
                    if integrated_trading_system.trading_history:
                        performance = integrated_trading_system.get_performance_report()
                        print(f"[RL Performance] Total Trades: {performance['total_trades']}, Execution Rate: {performance['execution_rate']:.2f}")
                
                if (i + 1) % 10 == 0:
                    trade_analyzer.summary(top_n=3)
                    grades = await sync_bus.get_state('strategy_grades') or {}
                    perf = await sync_bus.get_state('system_performance') or {}
                    logger.info(f"Tick {i+1}/{ticks}: {len(grades)} strategies, pnl={perf.get('pnl', 0.0):.2f}")
            
            trade_analyzer.performance_metrics()
            trade_analyzer.export_to_csv("trade_log_final.csv")
            
            logger.info("Demo session completed")

        # Run the mock generator
        await mock_market_data_generator()

    import sys
    if sys.platform.startswith('win'):
        # Windows event loop policy fix
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Shutting down MirrorCore-X system")
    finally:
        logger.info("MirrorCore-X system stopped.")