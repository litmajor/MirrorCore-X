# MirrorCore-X: A Multi-Agent Cognitive Trading System
Enhanced with High-Performance SyncBus and Data Pipeline Separation

Key Features:
- High-Performance SyncBus with delta updates and interest-based routing
- Fault-resistant agent isolation and circuit breakers
- Separated market data pipeline from agent state management
- Real-time monitoring and command interface capabilities
"""

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
from gym import spaces
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime, timezone
from dataclasses import dataclass
import os
from tabulate import tabulate

# External dependencies
# Assuming these modules exist in the same directory or are importable
# from scanner import MomentumScanner, TradingConfig
# from trade_analyzer_agent import TradeAnalyzerAgent
# from arch_ctrl import ARCH_CTRL
# from strategy_trainer_agent import StrategyTrainerAgent, UTSignalAgent, GradientTrendAgent, SupportResistanceAgent
# from mirror_optimizer import MirrorOptimizerAgent, OptimizableAgent

# Mock imports for standalone execution if necessary
class MomentumScanner:
    def __init__(self, exchange=None, config=None): pass
    async def scan_market(self, timeframe="7d"): return pd.DataFrame()
    def get_strong_signals(self, timeframe="7d"): return pd.DataFrame()

class TradeAnalyzerAgent:
    def __init__(self): pass

class ARCH_CTRL:
    def __init__(self):
        self.fear = 0.0
        self.stress = 0.0
        self.confidence = 0.5
        self.emotional_state = "UNCERTAIN"

# Placeholder for OptimizableAgent interface
class OptimizableAgent:
    def get_hyperparameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def validate_params(self, params: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def evaluate(self) -> float:
        raise NotImplementedError

class StrategyTrainerAgent(OptimizableAgent):
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


class UTSignalAgent: pass
class GradientTrendAgent: pass
class SupportResistanceAgent: pass

class MirrorOptimizerAgent: # Mock for MirrorOptimizerAgent
    def __init__(self): pass
    def optimize(self, agent_config, data, target_metric): return {"best_params": {}, "best_score": 0.0}

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

# --- Emergency Controls & Audit Logging ---

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

    async def log_system_event(self, event_type: str, message: str, context: Dict[str, Any] = None):
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
                'message': 'Initiated closing of all open trades.',
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

# --- Market Data Pipeline (Separated from Agent State) ---
class MarketDataProcessor:
    """Validates and cleans raw market data"""

    def validate(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate incoming market data"""
        try:
            # Basic validation
            if not raw_data or 'price' not in raw_data:
                return None

            # Ensure numeric values
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
            return None

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

# --- Data Categorization Framework ---
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


# --- High-Performance SyncBus with Fault Resistance ---
class HighPerformanceSyncBus:
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

    async def register_agent(self, agent_id: str, interests: List[str] = None):
        """Register agent with specific data interests"""
        if agent_id in self.subscriptions: # Avoid re-registering
            return
        self.subscriptions[agent_id] = interests or ['all']
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

        try:
            # Process agents with timeout and isolation
            tasks = []
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
                # Use a reasonable timeout for gather, e.g., 10 seconds per tick
                results = await asyncio.gather(*tasks, return_exceptions=True, timeout=10)
                await self._process_tick_results(results, tasks)

            # Update system health metrics
            await self._update_system_health()

        except asyncio.TimeoutError:
            logger.error(f"Tick processing timed out for tick {self.tick_count}")
            self.performance_metrics['failed_updates'] += 1
            # Potentially isolate agents that timed out
            for i, task in enumerate(tasks):
                # Check if the task is done and has an exception (likely timeout)
                if task.done() and task.exception() is not None:
                    agent_id = task.get_name().replace('agent_', '')
                    if agent_id in self.agent_states: # Ensure agent is still registered
                        await self._record_failure(agent_id, asyncio.TimeoutError("Agent processing timed out"))
                        if self._should_open_circuit(agent_id):
                            await self._isolate_agent(agent_id)

        except Exception as e:
            logger.error(f"Critical error in tick processing: {str(e)}")
            self.performance_metrics['failed_updates'] += 1

    async def _process_agent_safely(self, agent_id: str) -> Tuple[str, Any]:
        """Process single agent with safety wrapper"""
        try:
            # Get queued updates for this agent
            updates = []
            while not self.update_queues[agent_id].empty():
                try:
                    update = self.update_queues[agent_id].get_nowait()
                    updates.append(update)
                except asyncio.QueueEmpty:
                    break

            # Simulate agent processing by calling its update method
            # NOTE: This assumes agents have an async `update` method.
            agent_instance = getattr(self, agent_id, None) # Get agent instance via attribute
            if agent_instance and hasattr(agent_instance, 'update'):
                # Combine queued updates and pass them
                combined_data = {}
                for update_dict in updates:
                    for key, value in update_dict.items():
                        combined_data[key] = value

                # Add any global states the agent might need
                global_states = await self.get_all_states()
                combined_data.update(global_states)
                combined_data['tick_count'] = self.tick_count
                combined_data['timestamp'] = time.time()
                combined_data['syncbus'] = self # Allow agents to interact with syncbus if needed

                # Call the agent's update method
                agent_result = await agent_instance.update(combined_data)
            else:
                agent_result = {'status': 'no_update_method'}

            return agent_id, agent_result

        except Exception as e:
            await self._record_failure(agent_id, e)
            return agent_id, {'error': str(e), 'status': 'failed'}

    async def _process_tick_results(self, results: List, tasks: List):
        """Process tick results and handle failures"""
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_name = tasks[i].get_name() if hasattr(tasks[i], 'get_name') else 'unknown_task'
                agent_id = task_name.replace('agent_', '') if task_name.startswith('agent_') else 'unknown'
                logger.error(f"Agent {agent_id} task failed during tick: {str(result)}")
                await self._record_failure(agent_id, result)
                if self._should_open_circuit(agent_id):
                    await self._isolate_agent(agent_id)

            elif isinstance(result, tuple) and len(result) == 2:
                agent_id, agent_result = result
                if agent_result.get('status') == 'failed' or 'error' in agent_result:
                    error_msg = agent_result.get('error', 'Unknown agent error')
                    logger.error(f"Agent {agent_id} reported failure: {error_msg}")
                    await self._record_failure(agent_id, Exception(error_msg))
                    if self._should_open_circuit(agent_id):
                        await self._isolate_agent(agent_id)
                else:
                    self._record_success(agent_id)
                    # Update agent state if agent returned a delta
                    if isinstance(agent_result, dict) and 'state_delta' in agent_result:
                         await self.update_agent_state(agent_id, agent_result['state_delta'])
                    elif isinstance(agent_result, dict) and agent_id in self.agent_states: # Assume direct state update
                        await self.update_agent_state(agent_id, agent_result)


    def _is_circuit_open(self, agent_id: str) -> bool:
        """Check if circuit breaker is open for agent"""
        breaker = self.circuit_breakers.get(agent_id, {})
        return breaker.get('is_open', False)

    def _should_open_circuit(self, agent_id: str) -> bool:
        """Determine if circuit should be opened"""
        breaker = self.circuit_breakers.get(agent_id, {})
        failure_count = breaker.get('failure_count', 0)

        # Open circuit after 5 failures in short time
        return failure_count >= 5

    async def _isolate_agent(self, agent_id: str):
        """Isolate failed agent without affecting others"""
        logger.error(f"Isolating failed agent: {agent_id}")

        # Open circuit breaker
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id]['is_open'] = True
            self.performance_metrics['circuit_breaker_activations'] += 1

            # Schedule recovery attempt
            asyncio.create_task(self._attempt_restart(agent_id, delay=30))
        else:
            logger.warning(f"Agent {agent_id} not found in circuit breakers for isolation.")


    async def _attempt_restart(self, agent_id: str, delay: int = 30):
        """Attempt to restart failed agent after cooldown"""
        await asyncio.sleep(delay)

        # Reset circuit breaker
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id] = {
                'is_open': False,
                'failure_count': 0,
                'last_failure': 0
            }
            self.performance_metrics['agent_restarts'] += 1
            logger.info(f"Attempted restart for agent: {agent_id}")
        else:
             logger.warning(f"Agent {agent_id} not found in circuit breakers for restart attempt.")

    async def _record_failure(self, agent_id: str, error: Exception):
        """Record agent failure"""
        if agent_id in self.agent_health:
            self.agent_health[agent_id]['failure_count'] += 1

        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id]['failure_count'] += 1
            self.circuit_breakers[agent_id]['last_failure'] = time.time()

    def _record_success(self, agent_id: str):
        """Record agent success"""
        if agent_id in self.agent_health:
            # Ensure success_count is always at least 1 for health score calculation
            self.agent_health[agent_id]['success_count'] = max(1, self.agent_health[agent_id].get('success_count', 0) + 1)

    def _is_relevant(self, interests: List[str], agent_id: str, state: Dict[str, Any]) -> bool:
        """Check if state is relevant to agent interests"""
        if 'all' in interests:
            return True

        # Check if any interest category is present in state keys or state data keys
        for interest in interests:
            if interest in state:  # Direct interest match
                return True
            # Check if any key within the state dictionary matches the interest
            for key in state.keys():
                if interest in key:
                    return True
        return False

    def _filter_data_by_interests(self, categorized_data: Dict[str, Any], interests: List[str]) -> Dict[str, Any]:
        """Filter data based on agent interests"""
        if 'all' in interests:
            # Return all categorized data plus raw market data if 'market_data' is of interest
            filtered = categorized_data.copy()
            if 'market_data' in interests and 'market_data' in categorized_data:
                 filtered['market_data'] = categorized_data['market_data']
            return filtered

        filtered_data = {}
        for interest in interests:
            if interest in categorized_data:
                filtered_data[interest] = categorized_data[interest]
            # Also include raw market data if 'market_data' is an interest
            if interest == 'market_data' and 'market_data' in categorized_data:
                filtered_data['market_data'] = categorized_data['market_data']
        return filtered_data

    async def _notify_interested_agents(self, source_agent_id: str, state_delta: Dict[str, Any]):
        """Notify other agents about state changes"""
        for agent_id, interests in self.subscriptions.items():
            if agent_id != source_agent_id and self._is_relevant(interests, agent_id, state_delta):
                try:
                    notification = {
                        'source_agent': source_agent_id,
                        'state_delta': state_delta,
                        'timestamp': time.time()
                    }
                    await self.update_queues[agent_id].put(notification)
                except asyncio.QueueFull:
                    logger.warning(f"Notification queue full for agent {agent_id}")

    async def _update_system_health(self):
        """Update system-wide health metrics"""
        total_agents = len(self.agent_states)
        healthy_agents = 0
        for agent_id in self.agent_states:
             is_open = self._is_circuit_open(agent_id)
             if not is_open:
                 healthy_agents += 1

        health_score = healthy_agents / total_agents if total_agents > 0 else 1.0

        async with self.lock:
            self.global_state['system_health'] = {
                'total_agents': total_agents,
                'healthy_agents': healthy_agents,
                'health_score': health_score,
                'performance_metrics': self.performance_metrics,
                'last_update': time.time()
            }

    async def get_state(self, key: str) -> Any:
        """Retrieve state value with lock"""
        async with self.lock:
            return self.global_state.get(key)

    async def update_state(self, key: str, value: Any) -> None:
        """Update global state with thread-safe access"""
        async with self.lock:
            if isinstance(self.global_state.get(key), list) and isinstance(value, list):
                self.global_state[key].extend(value)
            else:
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
        exchange_config = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        if use_testnet:
             exchange_config['urls'] = {'api': ccxt.binance().urls['test']}
             logger.warning("Using Binance testnet.")

        if dry_run:
            logger.warning("Using dry_run mode. Exchange interactions will be simulated.")
            # For simulation, we might not need actual API keys if using a mock exchange.
            # However, ccxt might still require them for initialization, even if not used for live trading.
            # If using ccxt directly in dry_run, it often relies on simulated order execution.
            exchange = ccxt.binance(exchange_config) # Placeholder for simulated exchange
        else:
            if not secrets_manager.validate_credentials():
                raise ValueError("API credentials missing for live trading. Please set EXHANGE_API_KEY, EXCHANGE_API_SECRET, EXCHANGE_API_PASSPHRASE.")
            exchange_config.update(secrets_manager.get_exchange_config())
            exchange = ccxt.binance(exchange_config) # Use real exchange for live trading

        # Set the exchange for the emergency controller
        emergency_controller.set_exchange(exchange)

        scanner = MomentumScanner(exchange=exchange)
        strategy_trainer = StrategyTrainerAgent()
        trade_analyzer = TradeAnalyzerAgent()
        arch_ctrl = ARCH_CTRL()

        # Instantiate agents that need to be attached to the SyncBus
        ego_processor = EgoProcessor()
        fear_analyzer = FearAnalyzer()
        self_awareness_agent = SelfAwarenessAgent()
        decision_mirror = DecisionMirror()
        execution_daemon = ExecutionDaemon(exchange=exchange, dry_run=dry_run)
        reflection_core = ReflectionCore()
        mirror_mind_meta_agent = MirrorMindMetaAgent()
        meta_agent = MetaAgent()
        risk_sentinel = RiskSentinel()
        # Import and create comprehensive optimizer
        # Ensure mirror_optimizer.py exists and contains ComprehensiveMirrorOptimizer
        try:
            from mirror_optimizer import ComprehensiveMirrorOptimizer
        except ImportError:
            logger.error("Could not import ComprehensiveMirrorOptimizer. Please ensure mirror_optimizer.py exists and is in the Python path.")
            # Provide a fallback or raise an error
            class ComprehensiveMirrorOptimizer: # Mock class if import fails
                def __init__(self, components):
                    logger.warning("Using mock ComprehensiveMirrorOptimizer.")
                def optimize(self, agent_config, data, target_metric):
                    return {"best_params": {}, "best_score": 0.0}

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
            'risk_sentinel': risk_sentinel
        }

        # Create comprehensive optimizer
        comprehensive_optimizer = ComprehensiveMirrorOptimizer(all_components)


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
        sync_bus.attach('comprehensive_optimizer', comprehensive_optimizer)


        # Register strategies (assuming these agents exist)
        strategy_trainer.register_strategy("UT_BOT", UTSignalAgent())
        strategy_trainer.register_strategy("GRADIENT_TREND", GradientTrendAgent())
        strategy_trainer.register_strategy("VBSR", SupportResistanceAgent())

        # Initialize Oracle and Imagination integration if enabled
        oracle_imagination_integration = None
        if enable_oracle or enable_bayesian or enable_imagination:
            try:
                from oracle_imagination_integration import integrate_oracle_and_imagination

                oracle_imagination_integration = await integrate_oracle_and_imagination(
                    sync_bus=sync_bus,
                    scanner=scanner,
                    strategy_trainer=strategy_trainer,
                    execution_daemon=execution_daemon,
                    trade_analyzer=trade_analyzer,
                    arch_ctrl=arch_ctrl,
                    enable_bayesian=enable_bayesian,
                    enable_imagination=enable_imagination
                )

                logger.info("✨ Oracle & Imagination integration complete")

            except Exception as e:
                logger.error(f"Failed to integrate Oracle & Imagination: {e}")
                logger.warning("Continuing without enhancement engines")

        logger.info("MirrorCore-X system created with enhanced SyncBus and intelligence layers")

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
            'console_monitor': console_monitor,
            'command_interface': command_interface,
            'emergency_controller': emergency_controller,
            'heartbeat_manager': heartbeat_manager,
            'audit_logger': audit_logger,
            'mirror_audit': mirror_audit,
            'secrets_manager': secrets_manager,
            'oracle_imagination': oracle_imagination_integration
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

# --- Main Execution ---
if __name__ == "__main__":
    async def main():
        # Set dry_run to True for simulation, False for live trading
        # Set use_testnet to True to use the exchange's test network
        sync_bus, components = await create_mirrorcore_system(dry_run=True, use_testnet=False)

        console_monitor = components['console_monitor']
        command_interface = components['command_interface']
        data_pipeline = components['data_pipeline']
        exchange = components['exchange']
        emergency_controller = components['emergency_controller']
        heartbeat_manager = components['heartbeat_manager']
        mirror_audit = components['mirror_audit']
        secrets_manager = components['secrets_manager']

        # Mock market data generation for simulation
        async def mock_market_data_generator():
            symbol = "BTC/USDT"
            price = 30000.0
            while True:
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

                # Display grid periodically
                if sync_bus.tick_count % 10 == 0:
                    await console_monitor.display_agent_grid()

                # Example: Check emergency conditions periodically
                if sync_bus.tick_count % 60 == 0: # Check every minute
                    # Fetch current PnL and position value (simplified)
                    performance = await sync_bus.get_state('system_performance')
                    current_pnl = performance.get('pnl', 0.0)
                    # Placeholder for current position value - would need to come from ExecutionDaemon or similar
                    current_positions_value = 5000.0 # Example value
                    # Placeholder for current latency - would need actual measurement
                    current_latency = np.random.uniform(50, 300)

                    await emergency_controller.check_health(current_pnl, current_positions_value, current_latency)

                    # Log a heartbeat event
                    await mirror_audit.log_system_event("heartbeat", "System is operational.", {"tick": sync_bus.tick_count})


                await asyncio.sleep(0.5) # Simulate time passing between ticks

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down MirrorCore-X system")
        if heartbeat_manager:
             heartbeat_manager.stop_heartbeat()
        await mirror_audit.log_system_event("system_shutdown", "MirrorCore-X system shutting down due to KeyboardInterrupt.")
    finally:
        # Cleanup
        if 'exchange' in components and components['exchange']:
             if hasattr(components['exchange'], 'close'):
                 asyncio.run(components['exchange'].close())
        logger.info("MirrorCore-X system stopped.")