
# strategy_trainer_agent.py
from collections import defaultdict
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import asyncio
import time

logger = logging.getLogger(__name__)

class StrategyTrainerAgent:
    """Enhanced StrategyTrainerAgent compatible with high-performance SyncBus"""
    
    def __init__(self, min_weight: float = 0.1, max_weight: float = 1.0, 
                 lookback_window: int = 10, pnl_scale_factor: float = 0.1, 
                 name: str = "StrategyTrainerAgent"):
        self.name = name
        self.strategies = {}  # name -> strategy module (must have .evaluate())
        self.performance = defaultdict(list)  # name -> [pnl, pnl, ...]
        self.signal_history = defaultdict(list)  # name -> [signals]
        self.live_weights = {}  # strategy -> 0.0–1.0 confidence
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.lookback_window = lookback_window
        self.pnl_scale_factor = pnl_scale_factor
        
        # Enhanced SyncBus compatibility
        self.data_interests = ['technical', 'market_data']
        self.is_paused = False
        self.command_queue = []
        self.last_update = time.time()
        
        self.logger = logging.getLogger(__name__)
        
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced update method compatible with SyncBus architecture"""
        # Process commands first
        await self._process_commands()
        
        if self.is_paused:
            return {'status': 'paused'}
        
        try:
            # Handle different data formats from enhanced SyncBus
            market_df = None
            
            # Extract market data from various possible formats
            if 'market_data_df' in data:
                market_df = data['market_data_df']
            elif 'technical' in data and 'market_data' in data['technical']:
                # Convert technical data to DataFrame format
                import pandas as pd
                tech_data = data['technical']
                market_df = pd.DataFrame([tech_data])
            elif 'market_data' in data:
                # Handle list of market data
                if isinstance(data['market_data'], list) and data['market_data']:
                    import pandas as pd
                    market_df = pd.DataFrame(data['market_data'])
            
            if market_df is not None and not market_df.empty:
                signals = self.evaluate(market_df)
                self.last_update = time.time()
                return {
                    f"{self.name}_signals": signals,
                    'status': 'active',
                    'confidence': self._calculate_overall_confidence(),
                    'last_update': self.last_update,
                    'strategy_count': len(self.strategies),
                    'best_strategy': self.get_best_strategy()
                }
            
            return {'status': 'no_data'}
            
        except Exception as e:
            self.logger.error(f"StrategyTrainerAgent update failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _process_commands(self):
        """Process commands from SyncBus"""
        while self.command_queue:
            command_msg = self.command_queue.pop(0)
            command = command_msg.get('command')
            
            if command == 'pause':
                self.is_paused = True
                self.logger.info(f"StrategyTrainer paused")
            elif command == 'resume':
                self.is_paused = False
                self.logger.info(f"StrategyTrainer resumed")
            elif command == 'emergency_stop':
                self.is_paused = True
                # Reset all strategy weights to minimum for safety
                for strategy in self.live_weights:
                    self.live_weights[strategy] = self.min_weight
                self.logger.warning(f"StrategyTrainer emergency stopped")
            elif command == 'reset_performance':
                # Reset all performance metrics
                self.performance.clear()
                self.signal_history.clear()
                for strategy in self.live_weights:
                    self.live_weights[strategy] = 0.5
                self.logger.info("StrategyTrainer performance reset")
        
    def register_strategy(self, name: str, strategy_module: Any) -> None:
        """Register a strategy with the trainer."""
        if not hasattr(strategy_module, 'evaluate'):
            raise ValueError(f"Strategy {name} must have an 'evaluate' method")
        
        self.strategies[name] = strategy_module
        self.live_weights[name] = 0.5  # start with neutral weight
        self.logger.info(f"Registered strategy: {name}")
    
    def learn_new_strategy(self, name: str, strategy_class: type, **kwargs) -> None:
        """Dynamically construct and register a new strategy from a class definition."""
        try:
            instance = strategy_class(**kwargs)
            self.register_strategy(name, instance)
            self.logger.info(f"Learned new strategy: {name}")
        except Exception as e:
            self.logger.error(f"Failed to learn strategy {name}: {e}")
            raise
    
    def evaluate(self, df) -> Dict[str, Any]:
        """Enhanced evaluate method with better error handling"""
        results = {}
        
        for name, module in self.strategies.items():
            try:
                signal_output = module.evaluate(df)
                
                # Handle different return types from evaluate()
                if isinstance(signal_output, (list, tuple)):
                    signal = signal_output[-1] if signal_output else 0
                elif isinstance(signal_output, np.ndarray):
                    signal = float(signal_output[-1]) if len(signal_output) > 0 else 0.0
                elif isinstance(signal_output, (int, float)):
                    signal = float(signal_output)
                else:
                    # Try to convert to float, default to 0
                    try:
                        signal = float(signal_output)
                    except (ValueError, TypeError):
                        signal = 0.0
                
                self.signal_history[name].append(signal)
                
                # Keep signal history manageable
                if len(self.signal_history[name]) > 1000:
                    self.signal_history[name] = self.signal_history[name][-1000:]
                
                results[name] = {
                    'signal': signal,
                    'weight': self.live_weights.get(name, 0.5),
                    'confidence': self._calculate_strategy_confidence(name),
                    'recent_performance': np.mean(self.performance[name][-10:]) if self.performance[name] else 0.0
                }
                
            except Exception as e:
                self.logger.error(f"Strategy {name} evaluation failed: {e}")
                results[name] = {
                    'signal': 0.0,
                    'weight': self.min_weight,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def update_performance(self, name: str, pnl: float) -> None:
        """Called by ExecutionDaemon or ReflectionCore with realized profit/loss"""
        if name not in self.strategies:
            self.logger.warning(f"Unknown strategy {name} for performance update")
            return
        
        self.performance[name].append(pnl)
        
        # Keep performance history manageable
        if len(self.performance[name]) > 1000:
            self.performance[name] = self.performance[name][-1000:]
        
        self._update_weight(name)
        
    def _update_weight(self, name: str) -> None:
        """Enhanced weight update with better metrics"""
        recent_pnl = self.performance[name][-self.lookback_window:]
        
        if not recent_pnl:
            return
        
        # Calculate performance metrics
        avg_pnl = np.mean(recent_pnl)
        win_rate = sum(1 for pnl in recent_pnl if pnl > 0) / len(recent_pnl)
        volatility = np.std(recent_pnl) if len(recent_pnl) > 1 else 0
        
        # Sharpe-like ratio (scaled)
        sharpe = (avg_pnl / volatility) if volatility > 0 else avg_pnl
        
        # Enhanced performance score with multiple factors
        performance_score = (
            (avg_pnl * self.pnl_scale_factor) + 
            (win_rate - 0.5) * 0.5 +
            np.clip(sharpe * 0.1, -0.2, 0.2)  # Bounded sharpe contribution
        )
        
        # Apply bounds with smoother transitions
        current_weight = self.live_weights.get(name, 0.5)
        weight_change = performance_score * 0.1  # Limit weight change per update
        new_weight = current_weight + weight_change
        
        # Apply final bounds
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        self.live_weights[name] = new_weight
        
        self.logger.debug(f"Updated weight for {name}: {new_weight:.3f} "
                         f"(avg_pnl: {avg_pnl:.4f}, win_rate: {win_rate:.3f}, sharpe: {sharpe:.3f})")
    
    def _calculate_strategy_confidence(self, name: str) -> float:
        """Calculate confidence score for a strategy"""
        if name not in self.performance or not self.performance[name]:
            return 0.5
        
        recent_pnl = self.performance[name][-self.lookback_window:]
        if len(recent_pnl) < 3:
            return 0.5
        
        # Confidence based on consistency and performance
        win_rate = sum(1 for pnl in recent_pnl if pnl > 0) / len(recent_pnl)
        avg_pnl = np.mean(recent_pnl)
        consistency = 1.0 - (np.std(recent_pnl) / (abs(np.mean(recent_pnl)) + 1e-6))
        
        confidence = (win_rate * 0.4) + (min(avg_pnl, 0.1) * 5 * 0.3) + (max(0, consistency) * 0.3)
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall system confidence"""
        if not self.live_weights:
            return 0.0
        
        weights = list(self.live_weights.values())
        confidences = [self._calculate_strategy_confidence(name) for name in self.live_weights.keys()]
        
        # Weighted average of confidences
        if confidences:
            overall_confidence = np.average(confidences, weights=weights)
            return float(np.clip(overall_confidence, 0.0, 1.0))
        
        return 0.0
    
    def get_best_strategy(self) -> Optional[str]:
        """Return the strategy with the highest adjusted weight."""
        if not self.live_weights:
            return None
        
        # Filter strategies with sufficient trading history
        valid_strategies = {
            name: weight for name, weight in self.live_weights.items()
            if len(self.performance[name]) >= 3  # Minimum trades required
        }
        
        if not valid_strategies:
            # If no strategy has enough history, return the one with highest weight
            return max(self.live_weights.items(), key=lambda x: x[1])[0]
        
        return max(valid_strategies.items(), key=lambda x: x[1])[0]
    
    def get_weighted_signal(self, signals: Dict[str, Any]) -> float:
        """Combine signals from all strategies using their weights."""
        if not signals:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, signal_data in signals.items():
            if isinstance(signal_data, dict):
                signal = signal_data.get('signal', 0.0)
                weight = signal_data.get('weight', self.live_weights.get(name, 0.5))
            else:
                signal = float(signal_data) if signal_data else 0.0
                weight = self.live_weights.get(name, 0.5)
            
            weighted_sum += signal * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def grade_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Enhanced strategy grading with more metrics"""
        grades = {}
        
        for name, pnl_list in self.performance.items():
            if not pnl_list:
                grades[name] = {
                    "average_pnl": 0.0,
                    "trades": 0,
                    "weight": round(self.live_weights.get(name, 0.5), 3),
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "confidence": 0.5,
                    "grade": "N/A"
                }
                continue
            
            recent_pnl = pnl_list[-self.lookback_window:]
            avg_pnl = np.mean(recent_pnl)
            win_rate = sum(1 for pnl in recent_pnl if pnl > 0) / len(recent_pnl)
            total_pnl = sum(pnl_list)
            confidence = self._calculate_strategy_confidence(name)
            
            # Assign letter grade
            if avg_pnl > 0.05 and win_rate > 0.6:
                grade = "A"
            elif avg_pnl > 0.02 and win_rate > 0.5:
                grade = "B"
            elif avg_pnl > 0 and win_rate > 0.4:
                grade = "C"
            elif avg_pnl > -0.02:
                grade = "D"
            else:
                grade = "F"
            
            grades[name] = {
                "average_pnl": round(avg_pnl, 4),
                "trades": len(pnl_list),
                "weight": round(self.live_weights.get(name, 0.5), 3),
                "win_rate": round(win_rate, 3),
                "total_pnl": round(total_pnl, 4),
                "confidence": round(confidence, 3),
                "grade": grade
            }
        
        return grades
    
    def reset_strategy_performance(self, name: str) -> None:
        """Reset performance history for a specific strategy."""
        if name in self.strategies:
            self.performance[name].clear()
            self.signal_history[name].clear()
            self.live_weights[name] = 0.5
            self.logger.info(f"Reset performance for strategy: {name}")
    
    def remove_strategy(self, name: str) -> None:
        """Remove a strategy from the trainer."""
        if name in self.strategies:
            del self.strategies[name]
            if name in self.performance:
                del self.performance[name]
            if name in self.signal_history:
                del self.signal_history[name]
            if name in self.live_weights:
                del self.live_weights[name]
            self.logger.info(f"Removed strategy: {name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status for monitoring"""
        return {
            'name': self.name,
            'is_paused': self.is_paused,
            'strategy_count': len(self.strategies),
            'total_trades': sum(len(pnl_list) for pnl_list in self.performance.values()),
            'best_strategy': self.get_best_strategy(),
            'overall_confidence': self._calculate_overall_confidence(),
            'last_update': self.last_update,
            'data_interests': self.data_interests
        }

# Enhanced Strategy Agent Wrappers with SyncBus compatibility
class StrategyAgent:
    """Enhanced base strategy agent with SyncBus compatibility"""
    
    def __init__(self, name: str, strategy_module: Any):
        self.name = name
        self.module = strategy_module
        self.data_interests = ['technical', 'market_data']
        self.is_paused = False
        self.command_queue = []
    
    def evaluate(self, df):
        if self.is_paused:
            return 0.0
        return self.module.evaluate(df)
    
    async def update(self, data):
        await self._process_commands()
        
        if self.is_paused:
            return {'status': 'paused'}
        
        df = data.get('market_data_df') or data.get('market_data')
        
        if hasattr(df, 'price') and hasattr(df, 'high') and hasattr(df, 'low') and hasattr(df, 'open'):
            import pandas as pd
            df = pd.DataFrame({
                'open': [df.open],
                'high': [df.high],
                'low': [df.low],
                'close': [df.price]
            })
        
        if df is not None:
            signals = self.evaluate(df)
            signal = signals[-1] if isinstance(signals, list) and signals else 'HOLD'
            return {
                f"{self.name}_signal": signal,
                'status': 'active',
                'confidence': 0.5  # Default confidence
            }
        
        return {'status': 'no_data'}
    
    async def _process_commands(self):
        """Process commands from SyncBus"""
        while self.command_queue:
            command_msg = self.command_queue.pop(0)
            command = command_msg.get('command')
            
            if command == 'pause':
                self.is_paused = True
            elif command == 'resume':
                self.is_paused = False
            elif command == 'emergency_stop':
                self.is_paused = True

class UTSignalAgent(StrategyAgent):
    def __init__(self, **kwargs):
        from strategies.ut_bot import UTBotStrategy
        super().__init__("UT_BOT", UTBotStrategy(**kwargs))

class GradientTrendAgent(StrategyAgent):
    def __init__(self, **kwargs):
        from strategies.gradient_trend_filter import GradientTrendFilter
        super().__init__("GRADIENT_TREND", GradientTrendFilter(**kwargs))

class SupportResistanceAgent(StrategyAgent):
    def __init__(self, **kwargs):
        from strategies.volume_sr_agent import VolumeSupportResistance
        super().__init__("VBSR", VolumeSupportResistance(**kwargs))

# Global trainer instance with enhanced capabilities
trainer = StrategyTrainerAgent(
    min_weight=0.1,
    max_weight=1.0,
    lookback_window=20,
    pnl_scale_factor=0.1
)

# Register additional strategies if available
try:
    from additional_strategies import register_additional_strategies
    register_additional_strategies(trainer)
except ImportError:
    logger.info("Additional strategies not available")

# Example usage
if __name__ == "__main__":
    # Test the enhanced strategy trainer
    async def test_trainer():
        trainer = StrategyTrainerAgent()
        
        # Register strategies
        ut_agent = UTSignalAgent()
        trainer.register_strategy("UT_BOT", ut_agent)
        
        # Test update with various data formats
        test_data = {
            'technical': {
                'price': 50000,
                'volume': 1000,
                'momentum': 0.05
            }
        }
        
        result = await trainer.update(test_data)
        print(f"Update result: {result}")
        
        # Test command processing
        trainer.command_queue.append({'command': 'pause'})
        result = await trainer.update(test_data)
        print(f"Paused result: {result}")
        
        trainer.command_queue.append({'command': 'resume'})
        result = await trainer.update(test_data)
        print(f"Resumed result: {result}")
        
        # Get status
        status = trainer.get_status()
        print(f"Status: {status}")
    
    import asyncio
    asyncio.run(test_trainer())
