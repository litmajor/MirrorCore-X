# strategy_trainer_agent.py
from collections import defaultdict
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple



class StrategyTrainerAgent:
    def update(self, data):
        # Integrate with main signal flow: evaluate all registered strategies
        df = data.get('market_data_df') or data.get('market_data')
        if df is not None:
            signals = self.evaluate(df)
            return {f"{self.name}_signals": signals}
        return {}
    def __init__(self, min_weight: float = 0.1, max_weight: float = 1.0, 
                 lookback_window: int = 10, pnl_scale_factor: float = 0.1, name: str = "StrategyTrainerAgent"):
        self.name = name
        self.strategies = {}  # name -> strategy module (must have .evaluate())
        self.performance = defaultdict(list)  # name -> [pnl, pnl, ...]
        self.signal_history = defaultdict(list)  # name -> [signals]
        self.live_weights = {}  # strategy -> 0.0–1.0 confidence
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.lookback_window = lookback_window
        self.pnl_scale_factor = pnl_scale_factor
        self.logger = logging.getLogger(__name__)
        
    def register_strategy(self, name: str, strategy_module: Any) -> None:
        """Register a strategy with the trainer."""
        if not hasattr(strategy_module, 'evaluate'):
            raise ValueError(f"Strategy {name} must have an 'evaluate' method")
        
        self.strategies[name] = strategy_module
        self.live_weights[name] = 0.5  # start with neutral weight
        self.logger.info(f"Registered strategy: {name}")
    
    def learn_new_strategy(self, name: str, strategy_class: type, **kwargs) -> None:
        """
        Dynamically construct and register a new strategy from a class definition.
        """
        try:
            instance = strategy_class(**kwargs)
            self.register_strategy(name, instance)
            self.logger.info(f"Learned new strategy: {name}")
        except Exception as e:
            self.logger.error(f"Failed to learn strategy {name}: {e}")
            raise
    
    def evaluate(self, df) -> Dict[str, Any]:
        """
        For each strategy, evaluate the current market and collect signals.
        Returns a dict of {strategy_name: signal}
        """
        results = {}
        
        for name, module in self.strategies.items():
            try:
                signal_output = module.evaluate(df)
                # Fix: If df is a list, pass each item to evaluate and collect signals
                if isinstance(df, list):
                    signals = [module.evaluate(item) for item in df]
                    signal = signals[-1] if signals else 0
                else:
                    # Handle different return types from evaluate()
                    if isinstance(signal_output, (list, tuple)):
                        signal = signal_output[-1]  # Get the latest signal
                    elif isinstance(signal_output, np.ndarray):
                        signal = signal_output[-1]
                    else:
                        signal = signal_output  # Assume it's a scalar
                self.signal_history[name].append(signal)
                results[name] = signal
            except Exception as e:
                self.logger.error(f"Strategy {name} evaluation failed: {e}")
                results[name] = 0  # Neutral signal on error
        return results
    
    def update_performance(self, name: str, pnl: float) -> None:
        """
        Called by ExecutionDaemon or ReflectionCore with realized profit/loss
        """
        if name not in self.strategies:
            self.logger.warning(f"Unknown strategy {name} for performance update")
            return
        
        self.performance[name].append(pnl)
        self._update_weight(name)
        
    def _update_weight(self, name: str) -> None:
        """Update the weight for a strategy based on recent performance."""
        recent_pnl = self.performance[name][-self.lookback_window:]
        
        if not recent_pnl:
            return
        
        # Calculate performance metrics
        avg_pnl = np.mean(recent_pnl)
        win_rate = sum(1 for pnl in recent_pnl if pnl > 0) / len(recent_pnl)
        volatility = np.std(recent_pnl) if len(recent_pnl) > 1 else 0
        
        # Sharpe-like ratio (scaled)
        sharpe = (avg_pnl / volatility) if volatility > 0 else avg_pnl
        
        # Combine metrics for weight calculation
        performance_score = (avg_pnl * self.pnl_scale_factor) + (win_rate - 0.5) * 0.5
        
        # Apply bounds
        new_weight = float(max(self.min_weight, float(min(self.max_weight, float(0.5 + performance_score)))))
        self.live_weights[name] = new_weight
        
        self.logger.debug(f"Updated weight for {name}: {new_weight:.3f} "
                         f"(avg_pnl: {avg_pnl:.4f}, win_rate: {win_rate:.3f})")
    
    def get_best_strategy(self) -> Optional[str]:
        """
        Return the strategy with the highest adjusted weight.
        """
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
    
    def get_weighted_signal(self, signals: Dict[str, float]) -> float:
        """
        Combine signals from all strategies using their weights.
        """
        if not signals:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, signal in signals.items():
            weight = self.live_weights.get(name, 0.5)
            weighted_sum += signal * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def grade_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Output a grade/score per strategy.
        """
        grades = {}
        
        for name, pnl_list in self.performance.items():
            if not pnl_list:
                grades[name] = {
                    "average_pnl": 0.0,
                    "trades": 0,
                    "weight": round(self.live_weights.get(name, 0), 3),
                    "win_rate": 0.0,
                    "total_pnl": 0.0
                }
                continue
            
            recent_pnl = pnl_list[-self.lookback_window:]
            avg_pnl = np.mean(recent_pnl)
            win_rate = sum(1 for pnl in recent_pnl if pnl > 0) / len(recent_pnl)
            total_pnl = sum(pnl_list)
            
            grades[name] = {
                "average_pnl": round(avg_pnl, 4),
                "trades": len(pnl_list),
                "weight": round(self.live_weights.get(name, 0), 3),
                "win_rate": round(win_rate, 3),
                "total_pnl": round(total_pnl, 4)
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
            del self.performance[name]
            del self.signal_history[name]
            del self.live_weights[name]
            self.logger.info(f"Removed strategy: {name}")


# StrategyAgent wrappers
class StrategyAgent:
    def __init__(self, name: str, strategy_module: Any):
        self.name = name
        self.module = strategy_module
    
    def evaluate(self, df):
        return self.module.evaluate(df)


class UTSignalAgent(StrategyAgent):
    def __init__(self, **kwargs):
        from strategies.ut_bot import UTBotStrategy
        super().__init__("UT_BOT", UTBotStrategy(**kwargs))
    def update(self, data):
        df = data.get('market_data_df') or data.get('market_data')
        # If df is a MarketData object, convert to DataFrame
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
            print(f"[UT_BOT] Output signal: {signal} | Inputs: {df.tail(1).to_dict()}")
            return {f"{self.name}_signal": signal}
        return {}


class GradientTrendAgent(StrategyAgent):
    def __init__(self, **kwargs):
        from strategies.gradient_trend_filter import GradientTrendFilter
        super().__init__("GRADIENT_TREND", GradientTrendFilter(**kwargs))
    def update(self, data):
        df = data.get('market_data_df') or data.get('market_data')
        # If df is a MarketData object, convert to DataFrame
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
            signal = signals[-1] if isinstance(signals, list) and signals else 'NONE'
            print(f"[GRADIENT_TREND] Output signal: {signal} | Inputs: {df.tail(1).to_dict()}")
            return {f"{self.name}_signal": signal}
        return {}


class SupportResistanceAgent(StrategyAgent):
    def __init__(self, **kwargs):
        from strategies.volume_sr_agent import VolumeSupportResistance
        super().__init__("VBSR", VolumeSupportResistance(**kwargs))
    def update(self, data):
        df = data.get('market_data_df') or data.get('market_data')
        if df is not None:
            signal = self.evaluate(df)
            return {f"{self.name}_signal": signal}
        return {}


# Always-initialized trainer instance
trainer = StrategyTrainerAgent(
    min_weight=0.1,
    max_weight=1.0,
    lookback_window=20,
    pnl_scale_factor=0.1
)

from additional_strategies import register_additional_strategies
register_additional_strategies(trainer)

# Example usage
if __name__ == "__main__":
    # Register strategies
    ut_agent = UTSignalAgent()
    trainer.register_strategy("UT_BOT", ut_agent)
    
    # Simulate some trading
    import pandas as pd
    df = pd.DataFrame()  # Your market data here
    
    # Get signals
    signals = trainer.evaluate(df)
    
    # Update performance (would come from your execution system)
    trainer.update_performance("UT_BOT", 0.05)  # 5% profit
    
    # Get best strategy
    best = trainer.get_best_strategy()
    print(f"Best strategy: {best}")
    
    # Grade all strategies
    grades = trainer.grade_strategies()
    print("Strategy grades:", grades)
