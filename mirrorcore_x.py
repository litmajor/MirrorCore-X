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
async def create_mirrorcore_system(dry_run: bool = True) -> Tuple[HighPerformanceSyncBus, Any]:
    """Create enhanced MirrorCore-X system with high-performance SyncBus"""

    sync_bus = HighPerformanceSyncBus()
    data_pipeline = DataPipeline(sync_bus)
    console_monitor = ConsoleMonitor(sync_bus)
    command_interface = CommandInterface(sync_bus)

    try:
        exchange_config = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        if dry_run:
            # Use a mock or sandbox URL if available, otherwise simulate.
            # For simplicity, ccxt might handle dry_run internally or require specific setup.
            # Here, we'll just log the mode and use a placeholder.
            logger.warning("Using dry_run mode. Exchange interactions will be simulated.")
            # Example: If using a mock exchange library:
            # from mock_exchange import MockExchange
            # exchange = MockExchange(config=exchange_config)
            # For now, just use ccxt with a note it's simulated.
            exchange = ccxt.binance(exchange_config) # Placeholder for simulated exchange
        else:
            exchange = ccxt.binance(exchange_config) # Use real exchange for live trading


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
        from mirror_optimizer import ComprehensiveMirrorOptimizer

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

        logger.info("MirrorCore-X system created with enhanced SyncBus")

        return sync_bus, {
            'data_pipeline': data_pipeline,
            'console_monitor': console_monitor,
            'command_interface': command_interface,
            'exchange': exchange,
            # Make other key components accessible if needed
            'scanner': scanner,
            'arch_ctrl': arch_ctrl,
            'execution_daemon': execution_daemon,
            'strategy_trainer': strategy_trainer,
            'trade_analyzer': trade_analyzer,
            'comprehensive_optimizer': comprehensive_optimizer,
            'ego_processor': ego_processor,
            'fear_analyzer': fear_analyzer,
            'self_awareness': self_awareness_agent,
            'decision_mirror': decision_mirror,
            'reflection_core': reflection_core,
            'mirror_mind_meta': mirror_mind_meta_agent,
            'meta_agent': meta_agent,
            'risk_sentinel': risk_sentinel
        }

    except Exception as e:
        logger.error(f"Failed to create MirrorCore system: {e}")
        raise

# --- Mock Agent Implementations (for demonstration purposes) ---
# These should be replaced with actual agent implementations

class EgoProcessor(MirrorAgent):
    def __init__(self):
        super().__init__('EgoProcessor', data_interests=['market_data', 'fear_level', 'trade_log'])
        self.confidence = 0.5
        self.stress = 0.0
        self.fear = 0.0 # Add fear to ego state if relevant
        self.pnl = 0.0
        self.last_profile = None

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        market_data = data.get('market_data', {})
        fear_level = data.get('fear_level', self.fear)
        trade_log = data.get('trade_log', []) # Assume trade_log is passed here

        # Simulate psychological updates based on inputs
        if market_data:
            # Example: Higher volatility might increase stress
            self.stress = max(0.0, min(1.0, self.stress + (market_data.get('volatility', 0) - 0.05) * 0.5))
            # Example: Positive PnL might increase confidence, negative decrease
            if trade_log:
                current_pnl = sum(t.get('pnl', 0) for t in trade_log[-5:]) # Sum PnL from last 5 trades
                self.pnl = current_pnl
                if current_pnl > 0.01 * self.pnl: # Small PnL increase boosts confidence
                    self.confidence = min(1.0, self.confidence + 0.02)
                elif current_pnl < -0.01 * self.pnl: # Small PnL decrease lowers confidence
                    self.confidence = max(0.0, self.confidence - 0.02)

        # Update fear level based on external input or internal logic
        self.fear = fear_level

        # Simple emotional state based on confidence and stress
        if self.confidence > 0.7 and self.stress < 0.3:
            emotional_state = "CONFIDENT"
        elif self.stress > 0.6:
            emotional_state = "STRESSED"
        elif self.confidence < 0.3:
            emotional_state = "UNCERTAIN"
        else:
            emotional_state = "NORMAL"

        self.last_profile = PsychProfile(
            emotional_state=emotional_state,
            confidence_level=self.confidence,
            stress_level=self.stress,
            recent_pnl=self.pnl
        )

        return {
            'ego_state': {
                'confidence': self.confidence,
                'stress': self.stress,
                'fear': self.fear,
                'emotional_state': emotional_state
            },
            'psych_profile': self.last_profile
        }

class FearAnalyzer(MirrorAgent):
    def __init__(self):
        super().__init__('FearAnalyzer', data_interests=['market_data', 'ego_state'])
        self.volatility_threshold = 0.05
        self.fear_level = 0.0
        self.decay_rate = 0.1

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        market_data = data.get('market_data', {})
        ego_state = data.get('ego_state', {})

        volatility = market_data.get('volatility', 0)
        # Influence fear by ego's stress level
        ego_stress = ego_state.get('stress', 0.0)
        adjusted_volatility_risk = (volatility / self.volatility_threshold) * (1 + ego_stress * 0.5)

        # Update fear level
        self.fear_level = (1 - self.decay_rate) * self.fear_level + self.decay_rate * min(1.0, adjusted_volatility_risk)

        return {
            'fear_level': self.fear_level,
            'fear_metrics': {'volatility_risk': adjusted_volatility_risk} # Example metric
        }

class SelfAwarenessAgent(MirrorAgent):
    def __init__(self):
        super().__init__('SelfAwarenessAgent', data_interests=['ego_state', 'fear_level', 'execution_result', 'market_data'])
        self.awareness_state = {"consistency": 0.8, "drift": 0.1, "trust": 0.8, "deviations": 0}
        self.behavioral_baseline = {"avg_risk_tolerance": 0.5}

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ego_state = data.get('ego_state', {})
        fear_level = data.get('fear_level', 0.0)
        execution_result = data.get('execution_result', {})
        market_data = data.get('market_data', {})

        # Simulate updates to self-awareness state
        current_risk_tolerance = 1.0 - fear_level * (1.0 - ego_state.get('confidence', 0.5))
        self.awareness_state['drift'] = max(0.0, min(1.0, self.awareness_state['drift'] + (abs(current_risk_tolerance - self.behavioral_baseline['avg_risk_tolerance']) - 0.1) * 0.1))
        self.awareness_state['trust'] = max(0.0, min(1.0, self.awareness_state['trust'] + (0.1 - self.awareness_state['drift']) * 0.05))

        # Count deviations (simplified)
        deviations = 0
        if current_risk_tolerance < 0.2 or current_risk_tolerance > 0.8:
            deviations +=1
        if ego_state.get('stress', 0) > 0.7:
             deviations +=1
        self.awareness_state['deviations'] = deviations


        return {
            'self_state': self.awareness_state,
            'behavioral_baseline': self.behavioral_baseline # Include baseline if needed
        }

class DecisionMirror(MirrorAgent):
     def __init__(self):
        super().__init__('DecisionMirror', data_interests=['market_data', 'scanner_data', 'ego_state', 'fear_level', 'system_health'])
        self.selected_scenario = None
        self.decision_count = 0

     async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        market_data = data.get('market_data', {})
        scanner_data = data.get('scanner_data', {}) # Assuming scanner provides this
        ego_state = data.get('ego_state', {})
        fear_level = data.get('fear_level', 0.0)
        system_health = data.get('system_health', {})

        # Simplified decision logic
        if market_data and scanner_data:
            price = market_data.get('price', 0)
            momentum = scanner_data.get('momentum_7d', 0)
            confidence = ego_state.get('confidence', 0.5)
            risk_factor = fear_level # Use fear level as a proxy for risk aversion

            decision_strength = (confidence * (1.0 - risk_factor)) * abs(momentum)

            if decision_strength > 0.1:
                direction = "long" if momentum > 0 else "short"
                self.selected_scenario = TradeRecord( # Using TradeRecord as a simple scenario object
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    entry=price, # Entry price is a placeholder for decision scenario
                    pnl=0.0, # PnL is 0 for a decision scenario
                    strategy=f"{direction}_{momentum:.2f}" # Example strategy identifier
                )
                self.decision_count += 1
            else:
                self.selected_scenario = None

        return {
            'selected_scenario': self.selected_scenario,
            'decision_count': self.decision_count
        }


class ExecutionDaemon(MirrorAgent):
    DRY_RUN = True  # Set to False to enable live trading

    def __init__(self, exchange=None, capital=1000, risk_pct=0.01, dry_run=True):
        super().__init__('ExecutionDaemon', data_interests=['selected_scenario', 'market_data', 'arch_ctrl', 'risk_sentinel', 'meta_weights'])
        self.exchange = exchange
        self.capital = capital
        self.risk_pct = risk_pct
        self.trade_log = []
        self.virtual_balance = capital
        self.open_positions = {} # Stores {symbol: {"entry": price, "size": size, "entry_time": timestamp}}
        self.DRY_RUN = dry_run
        self.execution_count = 0
        self.active_positions = 0 # Track number of open positions

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        selected_scenario = data.get('selected_scenario')
        market_data = data.get('market_data', {})
        arch_ctrl = data.get('arch_ctrl', ARCH_CTRL()) # Use mock if not provided
        risk_sentinel_status = data.get('risk_sentinel', {}).get('risk_block', False)
        meta_weights = data.get('meta_weights', {}) # Get weights from MetaAgent

        action = "none"
        trades_executed = []
        closed_trades = []

        # Check if execution is blocked by risk sentinel or scenario is invalid
        if risk_sentinel_status or not selected_scenario:
            # If blocked, try to close existing positions if safe
            if not self.DRY_RUN and self.exchange and self.open_positions:
                 await self._close_all_positions(data)
            action = "blocked" # Indicate why execution didn't happen
            return {"execution_result": {"action": action, "trades": [], "closed_trades": [], "position_size_factor": 0.0}, "active_positions": self.active_positions}

        symbol = selected_scenario.symbol
        direction = selected_scenario.strategy.split('_')[0] # e.g., "long" from "long_0.123"
        momentum = float(selected_scenario.strategy.split('_')[1]) if len(selected_scenario.strategy.split('_')) > 1 else 0.0
        entry_price = market_data.get('price', selected_scenario.entry) # Use current market price if available

        # Apply MetaAgent weights (e.g., reduce execution if decision weight is low)
        decision_weight = meta_weights.get('decision_weight', 1.0)
        execution_weight = meta_weights.get('execution_weight', 1.0) # Could be used for risk management

        # Adjust position size based on scenario strength, risk, and meta-weights
        risk_amount = self.capital * self.risk_pct * decision_weight # Scale risk by decision confidence/weight
        position_size = (risk_amount / entry_price) * execution_weight # Scale size by execution weight

        # Emotional gating from ARCH_CTRL
        if arch_ctrl.stress > 0.7 or arch_ctrl.fear > 0.3:
            logger.warning(f"Emotional gating active: Stress={arch_ctrl.stress:.2f}, Fear={arch_ctrl.fear:.2f}. Blocking trade.")
            action = "gated"
            return {"execution_result": {"action": action, "trades": [], "closed_trades": [], "position_size_factor": 0.0}, "active_positions": self.active_positions}

        # Execute trade logic
        if self.DRY_RUN:
            logger.info(f"[🧪 DRY-RUN] Executing: {direction.upper()} {symbol} | Size: {position_size:.4f} | Price: {entry_price:.2f} | Decision Weight: {decision_weight:.2f}")
            # Simulate position and balance
            self.open_positions[symbol] = {"entry": entry_price, "size": position_size, "entry_time": time.time(), "direction": direction}
            self.virtual_balance -= entry_price * position_size if direction == "long" else entry_price * position_size # Simplified balance update
            self.active_positions = len(self.open_positions)
            action = "opened"
            trades_executed.append({
                "symbol": symbol, "entry": entry_price, "size": position_size,
                "risk": risk_amount, "direction": direction, "timestamp": time.time()
            })
            self.execution_count += 1

        else:
             # LIVE TRADING LOGIC (requires actual exchange integration)
             if self.exchange:
                 try:
                     order = await self.exchange.create_market_order(symbol, direction, position_size)
                     logger.info(f"Live Order Placed: {order}")
                     self.open_positions[symbol] = {"entry": order['price'], "size": order['amount'], "entry_time": time.time(), "direction": direction}
                     self.active_positions = len(self.open_positions)
                     action = "opened"
                     trades_executed.append({
                         "symbol": symbol, "entry": order['price'], "size": order['amount'],
                         "risk": risk_amount, "direction": direction, "timestamp": time.time()
                     })
                     self.execution_count += 1
                 except Exception as e:
                     logger.error(f"Live order execution failed for {symbol}: {e}")
                     action = "failed"


        # Simulate closing positions and PnL (for DRY_RUN)
        if self.DRY_RUN and self.open_positions:
            # Randomly close some positions to simulate trading activity
            symbols_to_close = list(self.open_positions.keys())
            np.random.shuffle(symbols_to_close)
            for sym in symbols_to_close[:max(0, len(symbols_to_close) // 2)]: # Close ~half
                if sym in self.open_positions:
                    pos = self.open_positions[sym]
                    exit_price = pos["entry"] * (1 + np.random.normal(0, 0.015)) # Simulate exit price with slight variation
                    profit = (exit_price - pos["entry"]) * pos["size"] if pos["direction"] == "long" else (pos["entry"] - exit_price) * pos["size"]
                    self.virtual_balance += exit_price * pos["size"]
                    logger.info(f"[🧾 DRY-RUN PROFIT] {sym}: ${profit:.2f} | New Balance: ${self.virtual_balance:.2f}")
                    closed_trades.append({
                        "symbol": sym, "entry": pos["entry"], "exit": exit_price,
                        "size": pos["size"], "pnl": profit
                    })
                    del self.open_positions[sym]
            self.active_positions = len(self.open_positions)


        return {
            "execution_result": {"action": action, "trades": trades_executed, "closed_trades": closed_trades, "position_size_factor": position_size},
            "trade_log": self.trade_log, # This should be updated by ReflectionCore typically
            "virtual_balance": self.virtual_balance,
            "active_positions": self.active_positions
        }

    async def _close_all_positions(self, data: Dict[str, Any]):
         """Attempt to close all open positions, especially during emergency stops or risk blocks."""
         logger.warning(f"Attempting to close all open positions due to risk block or emergency stop.")
         if self.DRY_RUN:
             self.virtual_balance += sum(pos['entry'] * pos['size'] for pos in self.open_positions.values()) # Add back capital
             self.open_positions = {}
             self.active_positions = 0
             logger.info("[DRY-RUN] All simulated positions closed.")
         elif self.exchange and self.open_positions:
             # Implement live closing logic here
             for symbol, pos in list(self.open_positions.items()):
                 try:
                     # Fetch current price to determine exit price
                     ticker = await self.exchange.fetch_ticker(symbol)
                     exit_price = ticker['last']
                     # Simple market close order
                     order = await self.exchange.create_market_order(symbol, 'sell' if pos['direction'] == 'long' else 'buy', pos['size'])
                     logger.info(f"Live order to close {symbol} placed: {order}")
                     # Update balance and remove position after confirmation (simplified)
                     profit = (order['price'] - pos['entry']) * pos['size'] if pos['direction'] == 'long' else (pos['entry'] - order['price']) * pos['size']
                     # Update self.capital or balance tracking accordingly
                     del self.open_positions[symbol]
                 except Exception as e:
                     logger.error(f"Failed to close live position for {symbol}: {e}")
             self.active_positions = len(self.open_positions)


class ReflectionCore(MirrorAgent):
    def __init__(self):
        super().__init__('ReflectionCore', data_interests=['execution_result', 'market_data'])
        self.trade_log = []
        self.pnl_curve = []
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.trades_count = 0
        self.wins = 0
        self.losses = 0

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        execution_result = data.get('execution_result', {})
        market_data = data.get('market_data', {}) # Needed for calculating PnL if exit price isn't in result

        trades = execution_result.get('trades', [])
        closed_trades = execution_result.get('closed_trades', [])

        for trade in trades: # Log opened trades
            if trade.get('symbol') and trade.get('entry') and trade.get('size'):
                self.trade_log.append({
                    "symbol": trade['symbol'],
                    "entry": trade['entry'],
                    "exit": None, # Not yet exited
                    "pnl": 0.0, # PnL is 0 initially
                    "strategy": trade.get('strategy', 'UNSPECIFIED'),
                    "timestamp": trade.get('timestamp', time.time())
                })

        for closed_trade in closed_trades: # Process closed trades
            symbol = closed_trade.get('symbol')
            if symbol and symbol in [t['symbol'] for t in self.trade_log if t['exit'] is None]: # Find the corresponding open trade
                # Find the specific trade log entry to update
                for log_entry in self.trade_log:
                    if log_entry['symbol'] == symbol and log_entry['exit'] is None:
                        log_entry['exit'] = closed_trade.get('exit', market_data.get('price')) # Use exit price from result or current market price
                        log_entry['pnl'] = closed_trade.get('pnl', 0.0)
                        self.total_pnl += log_entry['pnl']

                        # Update win/loss streaks
                        if log_entry['pnl'] > 0:
                            self.wins += 1
                            self.win_streak += 1
                            self.loss_streak = 0
                        else:
                            self.losses += 1
                            self.loss_streak += 1
                            self.win_streak = 0
                        break # Update only the first matching open trade

        self.trades_count = self.wins + self.losses
        if self.trades_count > 0:
            self.win_rate = self.wins / self.trades_count
        else:
            self.win_rate = 0.0

        # Update PnL curve (can be simplified to just total PnL)
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

class MirrorMindMetaAgent(MirrorAgent):
    def __init__(self):
        super().__init__('MirrorMindMetaAgent', data_interests=['ego_state', 'fear_level', 'self_state', 'decision_count', 'execution_count'])
        self.history = []
        self.session_grades = []

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ego_state = data.get('ego_state', {"confidence": 0.5, "stress": 0.5, "fear": 0.0})
        fear_state = data.get('fear_level', 0.0) # Assuming fear_level is directly passed
        self_state = data.get('self_state', {"drift": 0.0, "trust": 0.8, "deviations": 0})
        decision_count = data.get('decision_count', 0)
        execution_count = data.get('execution_count', 0)

        grade, insights = self.generate_insights(ego_state, fear_state, self_state, decision_count, execution_count)
        self.session_grades.append(grade)

        # Log insights and grade
        logger.info(f"[{self.name}] Session Grade: {grade} | Insights: {insights}")

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


class MetaAgent(MirrorAgent):
    def __init__(self):
        super().__init__('MetaAgent', data_interests=['fear_level', 'psych_profile', 'execution_result'])

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fear_level = data.get('fear_level', 0.0)
        psych_profile = data.get('psych_profile', {})
        confidence = psych_profile.get('confidence_level', 0.5)
        execution_result = data.get('execution_result', {})
        action = execution_result.get('action', 'none')

        # Adjust weights based on fear and confidence
        # Higher fear reduces execution weight, higher confidence boosts decision weight
        execution_weight = max(0.0, 1.0 - fear_level * 1.2) # Fear significantly impacts execution
        decision_weight = confidence * (1.0 - fear_level * 0.5) # Confidence boosts decision, but fear dampens it

        # If no execution happened, slightly reduce decision weight? (Optional logic)
        if action == 'none' or action == 'blocked' or action == 'gated':
             decision_weight *= 0.95

        logger.info(f"[{self.name}] Adjusting weights — Execution: {execution_weight:.2f}, Decision: {decision_weight:.2f}")

        return {
            'meta_weights': {
                'execution_weight': execution_weight,
                'decision_weight': decision_weight
            }
        }

class RiskSentinel(MirrorAgent):
    def __init__(self, max_drawdown_pct: float = -0.10, max_open_positions: int = 5):
        # Initialize with percentage-based drawdown
        super().__init__('RiskSentinel', data_interests=['performance_metrics', 'active_positions', 'execution_result'])
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.risk_block = False # State variable to track if trading is blocked

    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        performance = data.get('performance_metrics', {})
        active_positions = data.get('active_positions', 0)
        execution_action = data.get('execution_result', {}).get('action', 'none')

        total_pnl = performance.get('total_pnl', 0.0)
        # Assuming initial capital is available or can be derived (e.g., from ExecutionDaemon)
        # For simplicity, let's assume a starting capital of 1000 if not available
        initial_capital = 1000 # Replace with actual initial capital if available
        current_drawdown = total_pnl / initial_capital if initial_capital else 0.0

        # Check drawdown
        if current_drawdown <= self.max_drawdown_pct:
            logger.warning(f"[{self.name}] Max drawdown ({self.max_drawdown_pct*100:.1f}%) reached! PnL: ${total_pnl:.2f}. Blocking new trades.")
            self.risk_block = True
        else:
             # If conditions improve, potentially unblock
             if self.risk_block and current_drawdown > self.max_drawdown_pct * 0.8: # Unblock if drawdown improves by 20%
                 self.risk_block = False
                 logger.info(f"[{self.name}] Drawdown improved. Risk block lifted.")

        # Check open positions limit
        if active_positions >= self.max_open_positions:
            logger.warning(f"[{self.name}] Position limit ({self.max_open_positions}) reached! Active positions: {active_positions}. Blocking new trades.")
            self.risk_block = True
        # else: # If positions drop below limit, allow unblocking if not already unblocked by drawdown
             # if self.risk_block and active_positions < self.max_open_positions:
                 # self.risk_block = False # Allow unblocking if limit is no longer exceeded

        # If execution was blocked, ensure the block status is reflected
        if execution_action in ["blocked", "gated", "failed"]:
            self.risk_block = True # Ensure block is active if execution failed for any reason


        return {'risk_block': self.risk_block, 'active_positions': active_positions}


# --- System Creation Function ---
async def create_mirrorcore_system(dry_run: bool = True) -> Tuple[HighPerformanceSyncBus, Any]:
    """Create enhanced MirrorCore-X system with high-performance SyncBus"""

    sync_bus = HighPerformanceSyncBus()
    data_pipeline = DataPipeline(sync_bus)
    console_monitor = ConsoleMonitor(sync_bus)
    command_interface = CommandInterface(sync_bus)

    try:
        exchange_config = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        if dry_run:
            # Use a mock or sandbox URL if available, otherwise simulate.
            # For simplicity, ccxt might handle dry_run internally or require specific setup.
            # Here, we'll just log the mode and use a placeholder.
            logger.warning("Using dry_run mode. Exchange interactions will be simulated.")
            # Example: If using a mock exchange library:
            # from mock_exchange import MockExchange
            # exchange = MockExchange(config=exchange_config)
            # For now, just use ccxt with a note it's simulated.
            exchange = ccxt.binance(exchange_config) # Placeholder for simulated exchange
        else:
            exchange = ccxt.binance(exchange_config) # Use real exchange for live trading


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
        from mirror_optimizer import ComprehensiveMirrorOptimizer

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

        logger.info("MirrorCore-X system created with enhanced SyncBus")

        return sync_bus, {
            'data_pipeline': data_pipeline,
            'console_monitor': console_monitor,
            'command_interface': command_interface,
            'exchange': exchange,
            # Make other key components accessible if needed
            'scanner': scanner,
            'arch_ctrl': arch_ctrl,
            'execution_daemon': execution_daemon,
            'strategy_trainer': strategy_trainer,
            'trade_analyzer': trade_analyzer,
            'comprehensive_optimizer': comprehensive_optimizer,
            'ego_processor': ego_processor,
            'fear_analyzer': fear_analyzer,
            'self_awareness': self_awareness_agent,
            'decision_mirror': decision_mirror,
            'reflection_core': reflection_core,
            'mirror_mind_meta': mirror_mind_meta_agent,
            'meta_agent': meta_agent,
            'risk_sentinel': risk_sentinel
        }

    except Exception as e:
        logger.error(f"Failed to create MirrorCore system: {e}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    async def main():
        # Set dry_run to True for simulation, False for live trading
        sync_bus, components = await create_mirrorcore_system(dry_run=True)

        console_monitor = components['console_monitor']
        command_interface = components['command_interface']
        data_pipeline = components['data_pipeline']
        exchange = components['exchange']

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

                await asyncio.sleep(0.5) # Simulate time passing between ticks

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down MirrorCore-X system")
    finally:
        # Cleanup
        if 'exchange' in components and components['exchange']:
             if hasattr(components['exchange'], 'close'):
                 asyncio.run(components['exchange'].close())
        logger.info("MirrorCore-X system stopped.")