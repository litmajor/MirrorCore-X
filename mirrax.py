from pydantic import ConfigDict
"""# MirrorCore-X: A Multi-Agent Cognitive Trading System
mirrorcore_x.py

A cognitive trading system integrating multiple agents for market analysis, decision-making,
trade execution, performance tracking, psychological modeling, strategy management, and
hyperparameter optimization. Designed for cryptocurrency and forex markets, it leverages
technical indicators, emotional control, and Bayesian optimization for enhanced performance.

Key Features:
- Multi-agent architecture with asynchronous coordination via SyncBus
- MomentumScanner for real-time market signal generation
- TradeAnalyzerAgent for performance metrics and trade logging
- ARCH_CTRL for emotional state management and trade gating
- StrategyTrainerAgent for dynamic strategy weighting
- MirrorOptimizerAgent for hyperparameter tuning using Bayesian optimization
- Robust error handling with Pydantic data validation
- Risk management via RiskSentinel
- Simulated market data generation for testing
- Support for dry-run and live trading modes
- Comprehensive logging and documentation

Dependencies:
- pandas, numpy, ccxt.async_support, asyncio, logging, time, json
- pydantic (for data validation)
- bayes_opt (for optimization)
- scanner.py (MomentumScanner, TradingConfig)
- trade_analyzer_agent.py (TradeAnalyzerAgent)
- arch_ctrl.py (ARCH_CTRL)
- strategy_trainer_agent.py (StrategyTrainerAgent, UTSignalAgent, GradientTrendAgent, SupportResistanceAgent)
- optimizer.py (MirrorOptimizerAgent, OptimizableAgent)
- External strategy modules: ut_bot, gradient_trend_filter, volume_sr_agent
"""

import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import defaultdict
from pydantic import BaseModel, Field
from bayes_opt import BayesianOptimization
import gym
from gym import spaces
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime, timezone
from dataclasses import dataclass

# External dependencies
from scanner import MomentumScanner, TradingConfig
from trade_analyzer_agent import TradeAnalyzerAgent
from arch_ctrl import ARCH_CTRL
from strategy_trainer_agent import StrategyTrainerAgent, UTSignalAgent, GradientTrendAgent, SupportResistanceAgent
from mirror_optimizer import MirrorOptimizerAgent, OptimizableAgent

# RL library import for PPO
from stable_baselines3 import PPO
import pickle

# Import RL components
# (These would be the classes from the RL system - importing the key ones)
try:
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logging.warning("RL components not available. Running without RL capabilities.")

logger = logging.getLogger(__name__)

# --- System Configuration ---
# Configure logging with file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mirrorcore_x.log')
    ]
)
logger = logging.getLogger(__name__)

# Data contracts for consistent integration
class MarketData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    """Market data contract for price and volume information."""
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
    """MomentumScanner output contract for trading signals."""
    symbol: str
    momentum_7d: float = Field(description="7-day momentum score")
    signal: str = Field(description="Trading signal")  # Regex removed for compatibility
    rsi: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Relative Strength Index")
    macd: Optional[float] = Field(default=None, description="MACD indicator")
    timestamp: float = Field(gt=0.0, description="Signal timestamp")
    price: float = Field(gt=0.0, description="Current price")

class TradeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    """Trade record contract for executed trades."""
    symbol: str
    entry: float = Field(gt=0.0, description="Entry price")
    exit: Optional[float] = Field(default=None, gt=0.0, description="Exit price")
    pnl: float = Field(description="Profit and loss")
    strategy: str = Field(default="UNSPECIFIED", description="Strategy used")
    timestamp: float = Field(default_factory=time.time, description="Trade timestamp")

class TradingBotConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    """Configuration for TradingBot."""
    stop_loss: float = Field(default=0.02, ge=0.0, le=0.1, description="Stop-loss percentage")
    take_profit: float = Field(default=0.05, ge=0.0, le=0.2, description="Take-profit percentage")
    max_position_size: float = Field(default=0.1, ge=0.0, le=1.0, description="Max position size as portfolio fraction")

class SyncBus:
    """Central coordination bus for asynchronous agent communication and state management."""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.rl_agents: Dict[str, Any] = {}
        self.global_state: Dict[str, Any] = {
            'market_data': [],          # List of MarketData
            'scanner_data': [],         # List of ScannerOutput
            'trades': [],               # List of TradeRecord
            'oracle_directives': [],    # List of trading directives
            'system_performance': {     # System-wide metrics
                'pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0
            },
            'emotional_state': {},      # ARCH_CTRL state
            'market_context': {},       # Portfolio and market conditions
            'optimization_results': {},  # Optimization history
            'strategy_grades': {},      # Strategy performance

            # New RL-specific state
            'rl_predictions': [],
            'rl_training_data': [],
            'rl_performance': {},
            'meta_decisions': [],
            'integration_metrics': {}
        }
        self.lock = asyncio.Lock()
        self.tick_count = 0
        self.rl_enabled = RL_AVAILABLE
        logger.info(f"SyncBus initialized (RL enabled: {self.rl_enabled})")
    

    def attach(self, name: str, agent: Any) -> None:
        """Attach an agent to the bus.

        Args:
            name: Unique agent identifier.
            agent: Agent instance.
        """
        self.agents[name] = agent
        logger.info(f"Attached agent: {name}")
    
    def detach(self, name: str) -> None:
        """Detach an agent from the bus.

        Args:
            name: Agent identifier.
        """
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Detached agent: {name}")
    
    async def update_state(self, key: str, value: Any) -> None:
        """Update global state with thread-safe access.

        Args:
            key: State key.
            value: New value (list or single item).
        """
        async with self.lock:
            if isinstance(self.global_state.get(key), list) and isinstance(value, list):
                self.global_state[key].extend(value)
            else:
                self.global_state[key] = value
            logger.debug(f"Updated state: {key} with {len(value) if isinstance(value, list) else 'single value'}")
    
    async def get_state(self, key: str) -> Any:
        """Retrieve state value.

        Args:
            key: State key.

        Returns:
            State value or None if key not found.
        """
        async with self.lock:
            return self.global_state.get(key)
    
    async def tick(self) -> None:
        """Process one system tick, updating all agents asynchronously."""
        self.tick_count += 1
        logger.debug(f"Processing tick {self.tick_count}")
        
        try:
            traditional_tasks = [
                asyncio.wait_for(
                    agent.update(self.global_state) if hasattr(agent, 'update') else asyncio.sleep(0),
                    timeout=5.0
                )
                for agent in self.agents.values()
            ]
            # Process RL agents if available
            rl_tasks = []
            if self.rl_enabled:
                for rl_agent in self.rl_agents.values():
                    if hasattr(rl_agent, 'update'):
                        rl_tasks.append(
                            asyncio.wait_for(rl_agent.update(self.global_state), timeout=5.0)
                        )
            # Execute all tasks
            all_tasks = traditional_tasks + rl_tasks
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
             # Process results
            agent_names = list(self.agents.keys()) + list(self.rl_agents.keys())
            for agent_name, result in zip(agent_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent_name} failed during tick: {str(result)}")
                elif result and isinstance(result, dict):
                    for key, value in result.items():
                        await self.update_state(key, value)
        
        except Exception as e:
            logger.error(f"Tick processing failed: {str(e)}")

class RLIntegrationLayer:
    """Layer that integrates RL capabilities with existing system"""
    
    def __init__(self, sync_bus: SyncBus, scanner: MomentumScanner):
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
                try:
                    config = RLConfig(
                        lookback_window=20,
                        max_position_size=0.1,
                        transaction_cost=0.001,
                        total_timesteps=50000,
                        learning_rate=3e-4
                    )
                except Exception:
                    config = None
                try:
                    self.rl_agent = RLTradingAgent(
                        algorithm='PPO',
                        config=config if config is not None else RLConfig()
                    )
                except Exception:
                    self.rl_agent = None
                try:
                    self.meta_controller = MetaController(strategy="confidence_blend")
                except Exception:
                    self.meta_controller = None
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
            
            # Convert scanner data to RL format
            scanner_df = pd.DataFrame(scanner_data)
            rl_predictions = []
            
            if self.rl_agent.is_trained:
                # Generate RL predictions for each symbol
                for _, row in scanner_df.iterrows():
                    try:
                        # Prepare observation data
                        observation_data = self._prepare_rl_observation(row)
                        
                        # Get RL prediction
                        obs_array = self.rl_agent.encoder.transform_live(observation_data)
                        dummy_obs = np.repeat(obs_array, self.rl_agent.config.lookback_window, axis=0)
                        rl_action, _ = self.rl_agent.predict(dummy_obs, deterministic=True)
                        
                        rl_predictions.append({
                            'symbol': row['symbol'],
                            'rl_position': float(rl_action[0]),
                            'confidence': abs(rl_action[0]),  # Use absolute value as confidence
                            'timestamp': time.time()
                        })
                        
                    except Exception as e:
                        logger.error(f"RL prediction failed for {row.get('symbol', 'unknown')}: {e}")
                        continue
                
                self.integration_metrics['rl_predictions_count'] += len(rl_predictions)
            
            # Update state with RL predictions
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
            'momentum_long': row.get('momentum_7d', 0.0),  # Use same if long not available
            'rsi': row.get('rsi', 50.0),
            'macd': row.get('macd', 0.0),
            'bb_position': 0.5,  # Default if not available
            'volume_ratio': 1.0,  # Default
            'composite_score': row.get('momentum_7d', 0.0),
            'trend_score': row.get('momentum_7d', 0.0),
            'confidence_score': 0.5,  # Default
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



class PerceptionLayer:
    """Processes market and scanner data for downstream agents."""
    
    def __init__(self, scanner: MomentumScanner, sync_bus: 'SyncBus'):
        """Initialize perception layer.

        Args:
            scanner: MomentumScanner instance.
            sync_bus: SyncBus instance.
        """
        self.scanner = scanner
        self.sync_bus = sync_bus
        self.last_update = 0.0
        logger.info("PerceptionLayer initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update perception with scanner and market data, with robust error handling and input validation.
        Args:
            data: Global state with market_data.
        Returns:
            Dict with scanner_data and market_data.
        """
        try:
            scanner_df = await asyncio.wait_for(self.scanner.scan_market(), timeout=10.0)
            if not isinstance(scanner_df, pd.DataFrame):
                logger.error("Scanner returned non-DataFrame result: %s", type(scanner_df))
                return {}
            if scanner_df.empty:
                logger.warning("Scanner returned empty DataFrame")
                return {}
            if not all(col in scanner_df.columns for col in ['symbol', 'close']):
                logger.error(f"Scanner DataFrame missing required columns: {scanner_df.columns.tolist()}")
                return {}
            scanner_data = []
            for idx, row in scanner_df.iterrows():
                try:
                    row_dict = row.to_dict()
                    # Validate required fields
                    if not row_dict.get('symbol') or row_dict.get('close') is None:
                        logger.warning(f"Malformed scanner row at index {idx}: {row_dict}")
                        continue
                    scanner_data.append(ScannerOutput(**{**row_dict, 'price': row_dict.get('close', 0.0)}).model_dump())
                except Exception as row_exc:
                    logger.error(f"Error processing scanner row at index {idx}: {row_exc}")
            await self.sync_bus.update_state('scanner_data', scanner_data)
            market_data = data.get('market_data', [])
            validated_data = []
            if market_data:
                for i, md in enumerate(market_data):
                    try:
                        validated_data.append(MarketData(**md).model_dump())
                    except Exception as md_exc:
                        logger.error(f"Malformed market_data at index {i}: {md} | Error: {md_exc}")
                await self.sync_bus.update_state('market_data', validated_data)
            self.last_update = time.time()
            logger.info(f"Perception updated: {len(scanner_data)} scanner records, {len(validated_data)} market records")
            return {'scanner_data': scanner_data, 'market_data': validated_data}
        except asyncio.TimeoutError:
            logger.error("Scanner timed out in PerceptionLayer.update")
            return {}
        except Exception as e:
            logger.error(f"Perception update failed: {str(e)}", exc_info=True)
            return {}

class EgoProcessor:
    """Manages psychological state and sentiment bias."""
    
    def __init__(self, arch_ctrl: ARCH_CTRL, sync_bus: SyncBus):
        """Initialize ego processor.

        Args:
            arch_ctrl: ARCH_CTRL instance.
            sync_bus: SyncBus instance.
        """
        self.arch_ctrl = arch_ctrl
        self.sync_bus = sync_bus
        self.sentiment_bias = 0.0
        logger.info("EgoProcessor initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update ego state based on market and trades.

        Args:
            data: Global state with market_data and trades.

        Returns:
            Dict with sentiment_bias and emotional_state.
        """
        try:
            market_data = data.get('market_data', [])
            trades = data.get('trades', [])
            
            volatility = (
                np.std([md['price'] for md in market_data[-10:]]) /
                np.mean([md['price'] for md in market_data[-10:]])
            ) if len(market_data) >= 10 else 0.0
            loss_occurred = any(trade['pnl'] < 0 for trade in trades[-5:]) if trades else False
            
            self.arch_ctrl.update(
                market_volatility=float(volatility),
                loss_occurred=loss_occurred,
                architect_override=False
            )
            self.sentiment_bias = max(min(self.arch_ctrl.confidence - self.arch_ctrl.fear, 1.0), -1.0)
            
            # Use getattr to avoid attribute error if get_emotional_state is missing
            emotional_state = getattr(self.arch_ctrl, 'get_emotional_state', lambda: {} )()
            await self.sync_bus.update_state('emotional_state', emotional_state)
            
            logger.debug(f"Ego updated: sentiment_bias={self.sentiment_bias:.2f}, volatility={volatility:.4f}")
            return {'sentiment_bias': self.sentiment_bias, 'emotional_state': emotional_state}
        
        except Exception as e:
            logger.error(f"EgoProcessor update failed: {str(e)}")
            return {}

class FearAnalyzer:
    """Analyzes fear levels based on market and trade outcomes."""
    
    def __init__(self, arch_ctrl: ARCH_CTRL):
        """Initialize fear analyzer.

        Args:
            arch_ctrl: ARCH_CTRL instance.
        """
        self.arch_ctrl = arch_ctrl
        self.fear_trend = []
        self.max_trend_length = 20
        logger.info("FearAnalyzer initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update fear analysis.

        Args:
            data: Global state with trades.

        Returns:
            Dict with fear_trend.
        """
        try:
            trades = data.get('trades', [])
            recent_losses = sum(1 for trade in trades[-5:] if trade['pnl'] < 0) if trades else 0
            
            if recent_losses >= 3:
                self.arch_ctrl.inject_emotion(fear=min(self.arch_ctrl.fear + 0.2, 1.0))
            
            self.fear_trend.append(self.arch_ctrl.fear)
            if len(self.fear_trend) > self.max_trend_length:
                self.fear_trend.pop(0)
            
            logger.debug(f"Fear level: {self.arch_ctrl.fear:.2f}, recent_losses={recent_losses}")
            return {'fear_trend': self.fear_trend}
        
        except Exception as e:
            logger.error(f"FearAnalyzer update failed: {str(e)}")
            return {}

class SelfAwarenessAgent:
    """Monitors system performance and agent deviations."""
    
    def __init__(self, trade_analyzer: TradeAnalyzerAgent, sync_bus: 'SyncBus'):
        """Initialize self-awareness agent.

        Args:
            trade_analyzer: TradeAnalyzerAgent instance.
            sync_bus: SyncBus instance.
        """
        self.trade_analyzer = trade_analyzer
        self.sync_bus = sync_bus
        self.performance_history = []
        self.max_history_length = 100
        logger.info("SelfAwarenessAgent initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update self-awareness metrics.

        Args:
            data: Global state with trades.

        Returns:
            Dict with performance_history.
        """
        try:
            drawdown_stats = self.trade_analyzer.get_drawdown_stats() or {}
            max_drawdown = drawdown_stats.get('max_drawdown', 0.0)
            
            performance = {
                'pnl': self.trade_analyzer.get_total_pnl(),
                'max_drawdown': max_drawdown,
                'win_rate': (
                    len([t for t in data.get('trades', []) if t['pnl'] > 0]) /
                    len(data.get('trades', [])) if data.get('trades') else 0.0
                ),
                'sharpe_ratio': (
                    (self.trade_analyzer.performance_metrics() or {}).get('sharpe_ratio', 0.0)
                    if self.trade_analyzer.performance_metrics() is not None else 0.0
                ),
                'timestamp': time.time()
            }
            
            self.performance_history.append(performance)
            if len(self.performance_history) > self.max_history_length:
                self.performance_history.pop(0)
            
            logger.debug(f"Self-awareness updated: max_drawdown={max_drawdown:.4f}, pnl={performance['pnl']:.4f}")
            await self.sync_bus.update_state('system_performance', performance)
            return {'performance_history': self.performance_history}
        
        except Exception as e:
            logger.error(f"SelfAwarenessAgent update failed: {str(e)}")
            return {}

class DecisionMirror:
    """Validates trading decisions based on emotional state."""
    
    def __init__(self, arch_ctrl: ARCH_CTRL):
        """Initialize decision mirror.

        Args:
            arch_ctrl: ARCH_CTRL instance.
        """
        self.arch_ctrl = arch_ctrl
        logger.info("DecisionMirror initialized")
    
    async def validate_decision(self, directive: Dict[str, Any]) -> bool:
        """Validate a trading directive.

        Args:
            directive: Trading directive.

        Returns:
            True if valid, False otherwise.
        """
        try:
            if not directive:
                logger.warning("Empty directive provided")
                return False
            if not self.arch_ctrl.allow_action():
                logger.warning(f"Decision blocked: fear={self.arch_ctrl.fear:.2f}, confidence={self.arch_ctrl.confidence:.2f}")
                return False
            return True
        except Exception as e:
            logger.error(f"Decision validation failed: {str(e)}")
            return False

class ExecutionDaemon:
    """Executes trading directives in dry-run or live mode."""
    
    def __init__(self, exchange: ccxt.Exchange, trade_analyzer: TradeAnalyzerAgent, arch_ctrl: ARCH_CTRL, risk_sentinel: 'RiskSentinel', dry_run: bool = True):
        """Initialize execution daemon.

        Args:
            exchange: CCXT exchange instance.
            trade_analyzer: TradeAnalyzerAgent instance.
            arch_ctrl: ARCH_CTRL instance.
            risk_sentinel: RiskSentinel instance.
            dry_run: Simulate trades if True.
        """
        self.exchange = exchange
        self.trade_analyzer = trade_analyzer
        self.arch_ctrl = arch_ctrl
        self.risk_sentinel = risk_sentinel
        self.dry_run = dry_run
        self.position_cache = defaultdict(float)
        self.order_cache = defaultdict(dict)
        logger.info(f"ExecutionDaemon initialized: dry_run={dry_run}")
    
    async def execute(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading directive.

        Args:
            directive: Trading directive with symbol, action, amount, price, strategy.

        Returns:
            Dict with trade and order details.
        """
        try:
            if not await DecisionMirror(self.arch_ctrl).validate_decision(directive):
                return {}
            
            symbol = directive.get('symbol')
            action = directive.get('action')
            amount = directive.get('amount', 0.0)
            price = directive.get('price', 0.0)
            strategy = directive.get('strategy', 'UNSPECIFIED')
            
            if action == 'hold' or amount <= 0 or not symbol:
                logger.debug(f"Skipping directive: action={action}, amount={amount}, symbol={symbol}")
                return {}
            
            # Check risk limits
            if not await self.risk_sentinel.check_risk(self.trade_analyzer, self.position_cache, directive):
                logger.warning(f"Directive blocked by RiskSentinel: {symbol}, {action}")
                return {}
            
            trade = {}
            order = None
            if self.dry_run:
                entry_price = price
                exit_price = price * (1.01 if action == 'buy' else 0.99)  # Mock 1% profit/loss
                pnl = (exit_price - entry_price) * amount if action == 'buy' else (entry_price - exit_price) * amount
                
                trade = TradeRecord(
                    symbol=symbol,
                    entry=entry_price,
                    exit=exit_price,
                    pnl=pnl,
                    strategy=strategy
                ).dict()
                
                self.trade_analyzer.record_trade(trade)
                self.position_cache[symbol] += amount if action == 'buy' else -amount
                # Removed sync_bus update; handled elsewhere or not needed
                
                logger.info(f"[DRY-RUN] Executed {action} for {symbol}: amount={amount:.4f}, PnL={pnl:.4f}")
            
            else:
                order_type = 'limit' if price else 'market'
                side = 'buy' if action == 'buy' else 'sell'
                
                order = await self.exchange.create_order(symbol, order_type, side, amount, price)
                trade = TradeRecord(
                    symbol=symbol,
                    entry=order['price'],
                    exit=None,
                    pnl=0.0,
                    strategy=strategy
                ).dict()
                
                self.position_cache[symbol] += amount if action == 'buy' else -amount
                self.order_cache[symbol][order['id']] = trade
                self.trade_analyzer.record_trade(trade)
                await self.trade_analyzer.sync_bus.update_state('trades', [trade])
                
                logger.info(f"[LIVE] Executed {action} for {symbol}: order_id={order['id']}, amount={amount:.4f}")
            
            return {'trade': trade, 'order': order if not self.dry_run else None}
        
        except Exception as e:
            logger.error(f"Execution failed for {directive.get('symbol', 'UNKNOWN')}: {str(e)}")
            return {}

class TradingBot:
    """Manages trade execution with stop-loss and take-profit logic."""
    
    def __init__(self, execution_daemon: ExecutionDaemon, bot_config: TradingBotConfig):
        """Initialize trading bot.

        Args:
            execution_daemon: ExecutionDaemon instance.
            bot_config: TradingBotConfig instance.
        """
        self.execution_daemon = execution_daemon
        self.config = bot_config
        self.active_trades: Dict[str, Dict] = {}  # symbol: trade_details
        logger.info("TradingBot initialized")
    
    async def process_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trading directive with risk management.

        Args:
            directive: Trading directive.

        Returns:
            Dict with execution result.
        """
        try:
            symbol = directive.get('symbol')
            action = directive.get('action')
            price = directive.get('price', 0.0)
            
            if symbol and action != 'hold':
                trade = self.active_trades.get(symbol)
                if trade:
                    entry_price = trade['entry_price']
                    if action == 'sell' and price <= entry_price * (1 - self.config.stop_loss):
                        logger.info(f"Stop-loss triggered for {symbol}: price={price:.2f}")
                        directive['action'] = 'sell'
                    elif action == 'sell' and price >= entry_price * (1 + self.config.take_profit):
                        logger.info(f"Take-profit triggered for {symbol}: price={price:.2f}")
                        directive['action'] = 'sell'
            result = await self.execution_daemon.execute(directive)
            if result.get('trade'):
                trade = result['trade']
                if action in ['buy', 'sell'] and symbol:
                    self.active_trades[str(symbol)] = {
                        'entry_price': trade['entry'],
                        'amount': directive['amount'],
                        'strategy': trade['strategy'],
                        'timestamp': trade['timestamp']
                    }
                if action == 'sell' and symbol and str(symbol) in self.active_trades:
                    del self.active_trades[str(symbol)]
            return result
        
        except Exception as e:
            logger.error(f"TradingBot processing failed: {str(e)}")
            return {}
        

class TradingOracleEngine(OptimizableAgent):
    """Enhanced oracle that incorporates RL predictions with existing logic"""
    
    def __init__(self, strategy_trainer: StrategyTrainerAgent, trade_analyzer: TradeAnalyzerAgent, 
                 rl_integration: Optional[RLIntegrationLayer] = None):
        # Keep your existing initialization
        self.strategy_trainer = strategy_trainer
        self.trade_analyzer = trade_analyzer
        self.rl_integration = rl_integration
        
        # Existing parameters
        self.confidence_threshold = 0.7
        self.prediction_horizon = 5
        self.risk_weight = 0.5
        
        # New RL integration parameters
        self.rl_weight = 0.3  # Weight for RL predictions
        self.rule_weight = 0.7  # Weight for rule-based predictions
        
        logger.info("Enhanced Trading Oracle Engine initialized")
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return optimizable hyperparameters (includes new RL weights)"""
        params = {
            "confidence_threshold": self.confidence_threshold,
            "prediction_horizon": self.prediction_horizon,
            "risk_weight": self.risk_weight
        }
        
        if self.rl_integration and RL_AVAILABLE:
            params.update({
                "rl_weight": self.rl_weight,
                "rule_weight": self.rule_weight
            })
        
        return params
    
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set hyperparameters"""
        self.confidence_threshold = float(params.get("confidence_threshold", self.confidence_threshold))
        self.prediction_horizon = int(params.get("prediction_horizon", self.prediction_horizon))
        self.risk_weight = float(params.get("risk_weight", self.risk_weight))
        
        if self.rl_integration and RL_AVAILABLE:
            self.rl_weight = float(params.get("rl_weight", self.rl_weight))
            self.rule_weight = float(params.get("rule_weight", self.rule_weight))
    
    def evaluate(self) -> float:
        """Evaluate performance using TradeAnalyzerAgent metrics (unchanged)"""
        try:
            metrics = self.trade_analyzer.performance_metrics() or {}
            return metrics.get('sharpe_ratio', 0.0)
        except Exception as e:
            logger.error(f"Oracle evaluation failed: {e}")
            return 0.0
    
    async def evaluate_trading_timelines(self, market_context: Dict[str, Any], 
                                       scanner_data: List[Dict], 
                                       rl_predictions: Optional[List[Dict]] = None) -> List[Dict]:
        """Enhanced directive generation with RL integration"""
        try:
            directives = []
            scanner_df = pd.DataFrame(scanner_data)
            
            if scanner_df.empty:
                return directives
            
            # Get traditional signals (your existing logic)
            signals = self.strategy_trainer.update({'market_data_df': scanner_df})
            weighted_signal = self.strategy_trainer.get_weighted_signal(
                signals.get(f"{self.strategy_trainer.name}_signals", {})
            )
            
            # Create RL predictions lookup
            rl_lookup = {}
            if rl_predictions and RL_AVAILABLE:
                rl_lookup = {pred['symbol']: pred for pred in rl_predictions}
            
            for _, row in scanner_df.iterrows():
                symbol = row['symbol']
                traditional_signal = row['signal']
                momentum = row['momentum_7d']
                price = row['price']
                
                # Traditional confidence calculation (your existing logic)
                traditional_confidence = min(0.9, abs(weighted_signal) + 0.3 
                                           if weighted_signal * momentum > 0 else 0.3)
                
                # Get RL prediction if available
                rl_position = 0.0
                rl_confidence = 0.0
                if symbol in rl_lookup:
                    rl_position = rl_lookup[symbol]['rl_position']
                    rl_confidence = rl_lookup[symbol]['confidence']
                
                # Blend traditional and RL signals
                if RL_AVAILABLE and self.rl_integration and hasattr(self.rl_integration, 'meta_controller') and self.rl_integration.meta_controller:
                    # Use meta controller to blend decisions
                    try:
                        meta_decision = self.rl_integration.meta_controller.decide(
                            rule_decision=traditional_signal,
                            rl_action=np.array([rl_position]),
                            confidence_score=traditional_confidence,
                            market_regime="normal"
                        )
                        final_position = meta_decision.get('final_position', self._rule_to_position(traditional_signal))
                        final_confidence = max(traditional_confidence, rl_confidence)
                        action_method = meta_decision.get('method', "meta_blend")
                    except Exception as meta_exc:
                        logger.error(f"MetaController decision failed: {meta_exc}")
                        final_position = self._rule_to_position(traditional_signal)
                        final_confidence = traditional_confidence
                        action_method = "rule_based_only"
                else:
                    # Fall back to traditional signals only
                    final_position = self._rule_to_position(traditional_signal)
                    final_confidence = traditional_confidence
                    action_method = "rule_based_only"
                
                # Skip if confidence too low
                if final_confidence < self.confidence_threshold:
                    continue
                
                # Determine action
                action = (
                    'buy' if final_position > 0.1 else
                    'sell' if final_position < -0.1 else
                    'hold'
                )
                
                if action == 'hold':
                    continue
                
                # Calculate position size
                amount = min(
                    abs(final_position) * self.risk_weight * market_context.get('portfolio_value', 10000.0) / price,
                    100.0
                )
                
                directive = {
                    'symbol': symbol,
                    'action': action,
                    'amount': amount,
                    'price': price,
                    'strategy': self.strategy_trainer.get_best_strategy() or 'integrated',
                    'confidence': final_confidence,
                    'timestamp': time.time(),
                    # Additional metadata for analysis
                    'traditional_signal': traditional_signal,
                    'rl_position': rl_position,
                    'final_position': final_position,
                    'method': action_method
                }
                
                directives.append(directive)
            
            logger.info(f"Generated {len(directives)} integrated trading directives")
            return directives
            
        except Exception as e:
            logger.error(f"Enhanced trading timeline evaluation failed: {e}")
            return []
    
    def _rule_to_position(self, rule_decision: str) -> float:
        """Convert rule-based signal to position size"""
        signal_map = {
            'Strong Buy': 1.0,
            'Buy': 0.6,
            'Weak Buy': 0.3,
            'Neutral': 0.0,
            'Weak Sell': -0.3,
            'Sell': -0.6,
            'Strong Sell': -1.0
        }
        return signal_map.get(rule_decision, 0.0)
class OracleEnhancedMirrorCore:
    """Integrates scanner, oracle, and trade analyzer for trading decisions."""
    
    def __init__(self, sync_bus: SyncBus, scanner: MomentumScanner, trade_analyzer: TradeAnalyzerAgent):
        """Initialize enhanced mirror core.

        Args:
            sync_bus: SyncBus instance.
            scanner: MomentumScanner instance.
            trade_analyzer: TradeAnalyzerAgent instance.
        """
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
        """Update market context.

        Args:
            scanner_data: Scanner outputs.
            trades: Trade records.
        """
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
        """Process a tick integrating scanner, oracle, and bot.

        Args:
            oracle: TradingOracleEngine instance.
            trading_bot: TradingBot instance.

        Returns:
            Dict with directives or error.
        """
        try:
            scanner_data = await self.sync_bus.get_state('scanner_data') or []
            trades = await self.sync_bus.get_state('trades') or []
            
            await self._update_market_context(scanner_data, trades)
            directives = await oracle.evaluate_trading_timelines(self.market_context, scanner_data)
            
            # Process directives through trading bot
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

class ReflectionCore:
    """Reflects on performance and adjusts strategies."""
    
    def __init__(self, trade_analyzer: TradeAnalyzerAgent, strategy_trainer: StrategyTrainerAgent):
        """Initialize reflection core.

        Args:
            trade_analyzer: TradeAnalyzerAgent instance.
            strategy_trainer: StrategyTrainerAgent instance.
        """
        self.trade_analyzer = trade_analyzer
        self.strategy_trainer = strategy_trainer
        logger.info("ReflectionCore initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update strategy performance.

        Args:
            data: Global state with trades.

        Returns:
            Dict with strategy grades.
        """
        try:
            trades = data.get('trades', [])
            for trade in trades[-10:]:
                self.strategy_trainer.update_performance(trade['strategy'], trade['pnl'])
            
            grades = self.strategy_trainer.grade_strategies()
            # Removed sync_bus update; handled elsewhere or not needed
            
            logger.debug(f"Reflection updated: {len(grades)} strategies graded")
            return {'strategy_grades': grades}
        
        except Exception as e:
            logger.error(f"ReflectionCore update failed: {str(e)}")
            return {}

class MirrorMindMetaAgent:
    """Coordinates system and triggers optimization."""
    
    def __init__(self, sync_bus: SyncBus, optimizer: MirrorOptimizerAgent, trade_analyzer: TradeAnalyzerAgent, arch_ctrl: ARCH_CTRL):
        """Initialize meta-agent.

        Args:
            sync_bus: SyncBus instance.
            optimizer: MirrorOptimizerAgent instance.
            trade_analyzer: TradeAnalyzerAgent instance.
            arch_ctrl: ARCH_CTRL instance.
        """
        self.sync_bus = sync_bus
        self.optimizer = optimizer
        self.trade_analyzer = trade_analyzer
        self.arch_ctrl = arch_ctrl
        self.last_optimization = 0.0
        self.optimization_interval = 3600.0
        self.max_drawdown_trigger = -0.1
        logger.info("MirrorMindMetaAgent initialized")
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor performance and trigger optimization.

        Args:
            data: Global state with system_performance.

        Returns:
            Dict indicating optimization status.
        """
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
                bounds = {
                    'scanner': {
                        'momentum_period': (5, 50),
                        'rsi_window': (5, 50),
                        'volume_threshold': (0.5, 10.0)
                    },
                    'oracle': {
                        'confidence_threshold': (0.5, 0.95),
                        'prediction_horizon': (1, 20),
                        'risk_weight': (0.1, 1.0)
                    },
                    'mean_reversion': {
                        'bb_period': (10, 40),
                        'bb_std': (1.5, 3.0),
                        'rsi_period': (10, 20),
                        'zscore_threshold': (1.5, 3.0)
                    },
                    'momentum_breakout': {
                        'atr_period': (10, 20),
                        'breakout_multiplier': (1.5, 3.0),
                        'volume_threshold': (1.2, 2.0)
                    },
                    'volatility_regime': {
                        'vol_window': (15, 30),
                        'regime_threshold': (1.2, 2.0)
                    }
                }
                
                try:
                    # Call optimize_all_agents synchronously (not awaitable)
                    results = self.optimizer.optimize_all_agents(bounds, iterations=10)
                    self.last_optimization = current_time
                    await self.sync_bus.update_state('optimization_results', results)
                    
                    best_scores = {k: v.get('best_score', 0.0) for k, v in self.optimizer.get_optimization_history().items()}
                    if any(score > 0.5 for score in best_scores.values()):
                        self.arch_ctrl.inject_emotion(confidence=min(self.arch_ctrl.confidence + 0.1, 1.0))
                    
                    logger.info(f"Optimization triggered: {results}")
                    return {'optimization_triggered': True}
                
                except asyncio.TimeoutError:
                    logger.error("Optimization timed out")
                    return {'optimization_triggered': False}
            
            return {'optimization_triggered': False}
        
        except Exception as e:
            logger.error(f"MirrorMindMetaAgent update failed: {str(e)}")
            return {'optimization_triggered': False}

class RiskSentinel:
    """Enforces risk limits for trading operations."""
    
    def __init__(self, max_drawdown: float = -0.2, max_position_limit: float = 0.1, max_loss_per_trade: float = -0.05):
        """Initialize risk sentinel.

        Args:
            max_drawdown: Max allowable drawdown.
            max_position_limit: Max position size.
            max_loss_per_trade: Max loss per trade.
        """
        self.max_drawdown = max_drawdown
        self.max_position_limit = max_position_limit
        self.max_loss_per_trade = max_loss_per_trade
        self.alerts = []
        self.max_alerts = 50
        logger.info("RiskSentinel initialized")
    
    async def check_risk(self, trade_analyzer: TradeAnalyzerAgent, positions: Dict[str, float], directive: Dict[str, Any]) -> bool:
        """Check risk limits.

        Args:
            trade_analyzer: TradeAnalyzerAgent instance.
            positions: Current positions.
            directive: Trading directive.

        Returns:
            True if within limits, False otherwise.
        """
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

class MarketDataGenerator:
    """Simulates market data for testing."""
    
    def __init__(self, sentiment_bias: float = 0.0, volatility: float = 0.01):
        """Initialize market data generator.

        Args:
            sentiment_bias: Sentiment influence (-1 to 1).
            volatility: Base volatility.
        """
        self.base_price = 100.0
        self.base_volume = 1000.0
        self.sentiment_bias = sentiment_bias
        self.volatility = volatility
        self.time = 0
        logger.info("MarketDataGenerator initialized")
    
    async def generate(self, symbol: str = "BTC/USDT") -> MarketData:
        """Generate market data point.

        Args:
            symbol: Trading pair.

        Returns:
            MarketData instance or None.
        """
        try:
            self.time += 1
            price_change = np.random.normal(0, self.volatility) + self.sentiment_bias * 0.005
            price = self.base_price * (1 + price_change)
            volume = self.base_volume * (1 + np.random.normal(0, 0.1))
            
            self.base_price = max(price, 0.01)
            self.base_volume = max(volume, 0.0)
            
            data = MarketData(
                symbol=symbol,
                price=self.base_price,
                volume=self.base_volume,
                high=self.base_price * (1 + self.volatility * 0.5),
                low=self.base_price * (1 - self.volatility * 0.5),
                open=self.base_price * (1 - self.volatility * 0.1),
                timestamp=time.time(),
                volatility=self.volatility,
                sentiment=self.sentiment_bias
            )
            
            logger.debug(f"Generated data: {symbol} @ {self.base_price:.2f}, vol={self.base_volume:.2f}")
            return data
        
        except Exception as e:
            logger.error(f"Market data generation failed: {str(e)}")
            # Return a default MarketData instance with safe fallback values
            return MarketData(
                symbol=symbol,
                price=1.0,
                volume=0.0,
                high=1.0,
                low=1.0,
                open=1.0,
                timestamp=time.time(),
                volatility=self.volatility,
                sentiment=self.sentiment_bias

            )



# RL-specific classes from the second implementation
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

import gymnasium as gym
from gymnasium import spaces

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
            'balance': self.balance, 'position': self.position, 'pnl': self.total_pnl,
            'trade_count': self.trade_count, 'win_rate': self.win_count / max(self.trade_count, 1),
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
        if model_path:
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
        if 'ichimoku_bullish' in scanner_results.columns:
            scanner_results['ichimoku_bullish'] = scanner_results['ichimoku_bullish'].apply(float)
        else:
            scanner_results['ichimoku_bullish'] = 0.0
        if 'vwap_bullish' in scanner_results.columns:
            scanner_results['vwap_bullish'] = scanner_results['vwap_bullish'].astype(float)
        else:
            scanner_results['vwap_bullish'] = 0.0
        scanner_results['ema_crossover'] = scanner_results.get('ema_5_13_bullish', False)
        if isinstance(scanner_results['ema_crossover'], pd.Series):
            scanner_results['ema_crossover'] = scanner_results['ema_crossover'].astype(float)
        else:
            scanner_results['ema_crossover'] = scanner_results['ema_crossover'].astype(float)
        market_features = self.encoder.encode(scanner_results)
        price_data = np.array(scanner_results['price'].values, dtype=np.float32)
        return market_features, price_data
    
    def create_environment(self, market_data: np.ndarray, price_data: np.ndarray) -> TradingEnvironment:
        return TradingEnvironment(market_data=market_data, price_data=price_data, config=self.config)
    
    def train(self, scanner_results: pd.DataFrame, save_path: str = 'rl_trading_model', verbose: int = 1):
        logger.info(f"Starting RL training with {self.algorithm}")
        market_data, price_data = self.prepare_training_data(scanner_results)
        env = self.create_environment(market_data, price_data)
        self.model = PPO(
            'MlpPolicy', env, learning_rate=self.config.learning_rate, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
            max_grad_norm=0.5, verbose=verbose
        )
        self.model.learn(total_timesteps=self.config.total_timesteps)
        self.save_model(save_path)
        self.is_trained = True
        logger.info("RL training completed")
        return {'training_completed': True, 'total_timesteps': self.config.total_timesteps, 'algorithm': self.algorithm, 'final_model_path': save_path}
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        result = self.model.predict(observation, deterministic=deterministic)
        if isinstance(result, tuple) and len(result) == 2:
            action, state = result
            # If state is a tuple, take the first element (usually the hidden state for recurrent policies)
            if isinstance(state, tuple):
                state = state[0] if len(state) > 0 else None
            return action, state
        else:
            # Fallback in case predict returns only action
            return result, None
    
    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        with open(f"{path}_encoder.pkl", 'wb') as f:
            pickle.dump(self.encoder, f)
        logger.info(f"Model and encoder saved to {path}")
    
    def load_model(self, path: str):
        try:
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
    
    def decide(self, rule_decision: str, rl_action: np.ndarray, confidence_score: float = 0.5, market_regime: str = "normal") -> Dict[str, Any]:
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
            'final_position': final_position, 'rule_position': rule_position, 'rl_position': rl_position,
            'confidence_score': confidence_score, 'method': method, 'market_regime': market_regime
        }
        self.performance_history.append(decision)
        return decision
    
    def _rule_to_position(self, rule_decision: str) -> float:
        signal_map = {
            'Strong Buy': 1.0, 'Buy': 0.6, 'Weak Buy': 0.3, 'Neutral': 0.0,
            'Weak Sell': -0.3, 'Sell': -0.6, 'Strong Sell': -1.0, 'Overbought': -0.4, 'Oversold': 0.4
        }
        return signal_map.get(rule_decision, 0.0)

class IntegratedTradingSystem:
    def __init__(self, scanner: MomentumScanner, rl_agent: RLTradingAgent, meta_controller: MetaController):
        self.scanner = scanner
        self.rl_agent = rl_agent
        self.meta_controller = meta_controller
        self.trading_history = []
    
    async def generate_signals(self, timeframe: str = 'daily') -> pd.DataFrame:
        scanner_results = await self.scanner.scan_market(timeframe=timeframe)
        if scanner_results.empty:
            logger.warning("No scanner results available")
            return pd.DataFrame()
        signals_df = scanner_results.copy()
        if self.rl_agent.is_trained:
            rl_positions = []
            for _, row in scanner_results.iterrows():
                observation_data = {
                    'price': row['price'], 'volume': row.get('volume_usd', 0),
                    'momentum_short': row['momentum_short'], 'momentum_long': row['momentum_long'],
                    'rsi': row['rsi'], 'macd': row['macd'], 'bb_position': row.get('bb_position', 0.5),
                    'volume_ratio': row['volume_ratio'], 'composite_score': row['composite_score'],
                    'trend_score': row['trend_score'], 'confidence_score': row['confidence_score'],
                    'fear_greed': 50, 'btc_dominance': 50, 'volatility': row.get('volatility', 0),
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
                rule_decision=row['signal'], rl_action=np.array([row['rl_position']]),
                confidence_score=row['confidence_score'], market_regime=self._detect_market_regime(row)
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
                    'symbol': signal['symbol'], 'signal_type': signal['signal'],
                    'rule_position': signal['rule_position'], 'rl_position': signal['rl_position'],
                    'final_position': signal['final_position'], 'confidence': signal['confidence_score'],
                    'method': signal['method'], 'price': signal['price'],
                    'composite_score': signal['composite_score'], 'timestamp': datetime.now(timezone.utc),
                    'dry_run': dry_run
                }
                trade['executed'] = True  # Simulate execution in dry_run
                trades.append(trade)
                self.trading_history.append(trade)
            session_result = {
                'status': 'completed', 'timeframe': timeframe, 'total_signals': len(signals),
                'actionable_signals': len(actionable), 'trades_executed': len([t for t in trades if t['executed']]),
                'trades': trades, 'dry_run': dry_run, 'session_timestamp': datetime.now(timezone.utc)
            }
            logger.info(f"Trading session completed: {session_result['trades_executed']} trades executed")
            return session_result
        except Exception as e:
            logger.error(f"Trading session failed: {e}")
            return {'status': 'error', 'error': str(e)}


async def create_mirrorcore_system(dry_run: bool = True) -> tuple:
    from strategy_adapters import register_all_strategies_with_trainer

    """
    Initialize the MirrorCore-X system with RL integration.

    Args:
        dry_run: Simulate trades if True.

    Returns:
        Tuple of (SyncBus, TradeAnalyzerAgent, MomentumScanner, exchange, IntegratedTradingSystem).
    """
    try:
        exchange = ccxt.kucoinfutures({'enableRateLimit': True})
        sync_bus = SyncBus()
        trade_analyzer = TradeAnalyzerAgent()
        arch_ctrl = ARCH_CTRL()
        scanner = MomentumScanner(exchange=exchange)
        strategy_trainer = StrategyTrainerAgent(
            min_weight=0.1, max_weight=1.0, lookback_window=20, pnl_scale_factor=0.1
        )
        try:
            success = register_all_strategies_with_trainer(strategy_trainer)
            if not success:
                logger.warning("Some strategies failed to register, falling back to defaults")
                strategy_trainer.register_strategy("UT_BOT", UTSignalAgent())
                strategy_trainer.register_strategy("GRADIENT_TREND", GradientTrendAgent())
                strategy_trainer.register_strategy("VBSR", SupportResistanceAgent())
                strategy_trainer.register_strategy("default", GradientTrendAgent())
        except Exception as e:
            logger.warning(f"Strategy registration failed: {str(e)}, using default")
            strategy_trainer.register_strategy("default", GradientTrendAgent())
        perception = PerceptionLayer(scanner, sync_bus)
        ego = EgoProcessor(arch_ctrl, sync_bus)
        self_awareness = SelfAwarenessAgent(trade_analyzer, sync_bus)
        fear_analyzer = FearAnalyzer(arch_ctrl)
        risk_sentinel = RiskSentinel()
        oracle = TradingOracleEngine(strategy_trainer, trade_analyzer)
        mirror_core = OracleEnhancedMirrorCore(sync_bus, scanner, trade_analyzer)
        reflection = ReflectionCore(trade_analyzer, strategy_trainer)
        market_generator = MarketDataGenerator()
        bot_config = TradingBotConfig(stop_loss=0.02, take_profit=0.05, max_position_size=0.1)
        execution = ExecutionDaemon(exchange, trade_analyzer, arch_ctrl, risk_sentinel, dry_run)
        trading_bot = TradingBot(execution, bot_config)
        optimizable_agents = {"scanner": scanner, "oracle": oracle}
        optimizer = MirrorOptimizerAgent(optimizable_agents)
        meta_agent = MirrorMindMetaAgent(sync_bus, optimizer, trade_analyzer, arch_ctrl)
        # Initialize RL components
        rl_config = RLConfig()
        rl_agent = RLTradingAgent(algorithm='PPO', config=rl_config)
        meta_controller = MetaController(strategy="confidence_blend")
        integrated_trading_system = IntegratedTradingSystem(scanner, rl_agent, meta_controller)
        sync_bus.attach("perception", perception)
        sync_bus.attach("ego", ego)
        sync_bus.attach("fear_analyzer", fear_analyzer)
        sync_bus.attach("self_awareness", self_awareness)
        sync_bus.attach("execution", execution)
        sync_bus.attach("trading_bot", trading_bot)
        sync_bus.attach("oracle", oracle)
        sync_bus.attach("mirror_core", mirror_core)
        sync_bus.attach("reflection", reflection)
        sync_bus.attach("strategy_trainer", strategy_trainer)
        sync_bus.attach("meta_agent", meta_agent)
        sync_bus.attach("risk_sentinel", risk_sentinel)
        sync_bus.attach("market_generator", market_generator)
        sync_bus.attach("integrated_trading_system", integrated_trading_system)
        await sync_bus.update_state('market_context', {
            'portfolio_value': 10000.0, 'volatility': 0.01, 'sentiment': 0.0
        })
        logger.info("MirrorCore-X system with RL integration initialized")
        return sync_bus, trade_analyzer, scanner, exchange, integrated_trading_system
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise

async def run_demo_session(ticks: int = 100, dry_run: bool = True, tick_interval: float = 0.1):
    """
    Run a demo trading session with RL and existing integrations.

    Args:
        ticks: Number of ticks.
        dry_run: Simulate trades if True.
        tick_interval: Delay between ticks (seconds).
    """
    exchange = None
    trade_analyzer = None
    try:
        from bayesian_integration import add_bayesian_to_mirrorcore
        from imagination_integration import add_imagination_to_mirrorcore
        sync_bus, trade_analyzer, scanner, exchange, integrated_trading_system = await create_mirrorcore_system(dry_run)
        bayesian_integration = await add_bayesian_to_mirrorcore(sync_bus, enable=True)
        strategy_trainer = sync_bus.agents.get('strategy_trainer')
        execution_daemon = sync_bus.agents.get('execution')
        oracle_engine = sync_bus.agents.get('oracle')
        imagination_integration = await add_imagination_to_mirrorcore(
            sync_bus, strategy_trainer, execution_daemon, oracle_engine, enable=True
        )
        # Train RL agent with scanner data
        scanner_results = await scanner.scan_market(timeframe='daily')
        if not scanner_results.empty:
            integrated_trading_system.rl_agent.train(scanner_results, save_path='rl_trading_model')
        for i in range(ticks):
            if dry_run:
                market_data = await sync_bus.agents['market_generator'].generate()
                if market_data:
                    await sync_bus.update_state('market_data', [market_data.model_dump() if hasattr(market_data, 'model_dump') else market_data.dict()])
            # Run integrated trading session
            session_result = await integrated_trading_system.execute_trading_session(timeframe='daily', dry_run=dry_run)
            if session_result['status'] == 'completed':
                trades = session_result['trades']
                for trade in trades:
                    await sync_bus.update_state('trade', trade)
                    logger.info(f"Trade executed: {trade['symbol']}, Position: {trade['final_position']:.2f}, Method: {trade['method']}")
            await sync_bus.tick()
            await asyncio.sleep(tick_interval)
            if i % 20 == 0:
                insights = bayesian_integration.get_bayesian_insights()
                if insights:
                    best_strategy = insights.get('best_strategy')
                    regime = insights.get('market_context', {}).get('regime')
                    print(f"Tick {i}: Best={best_strategy}, Regime={regime}")
                if imagination_integration:
                    results = await imagination_integration.run_imagination_cycle(num_scenarios=10, scenario_length=30)
                    summary = imagination_integration.summary()
                    print(f"[Imagination] Robustness summary: {summary}")
                # Log RL performance
                if integrated_trading_system.trading_history:
                    performance = integrated_trading_system.get_performance_report()
                    print(f"[RL Performance] Total Trades: {performance['total_trades']}, Execution Rate: {performance['execution_rate']:.2f}")
            if (i + 1) % 10 == 0:
                trade_analyzer.summary(top_n=3)
                grades = await sync_bus.get_state('strategy_grades') or {}
                perf = await sync_bus.get_state('system_performance') or {}
                logger.info(f"Tick {i+1}/{ticks}: {len(grades)} strategies, pnl={perf.get('pnl', 0.0):.2f}")
        meta_agent = sync_bus.agents.get('meta_agent')
        if meta_agent and meta_agent.optimizer:
            meta_agent.optimizer.save_results("optimization_results.json")
            logger.info("Optimization results saved")
        trade_analyzer.performance_metrics()
        trade_analyzer.export_to_csv("trade_log_final.csv")
        await bayesian_integration.export_beliefs("final_beliefs.json")
        if imagination_integration:
            summary = imagination_integration.summary()
            logger.info(f"Final Imagination robustness summary: {summary}")
        # Export RL performance report
        if integrated_trading_system.trading_history:
            performance = integrated_trading_system.get_performance_report()
            logger.info(f"Final RL Performance: {performance}")
        logger.info("Demo session completed")
    except Exception as e:
        logger.error(f"Demo session failed: {str(e)}")
        if trade_analyzer is not None:
            trade_analyzer.export_to_csv("trade_log_error.csv")
    finally:
        if exchange is not None:
            try:
                await exchange.close()
            except Exception as close_exc:
                logger.warning(f"Exchange session close failed: {str(close_exc)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_demo_session(ticks=100, dry_run=True, tick_interval=0.1))