"""
Compatibility adapters to make the new strategies work with your StrategyTrainerAgent
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from strategy_trainer_agent import (
    UTSignalAgent, GradientTrendAgent, SupportResistanceAgent
)


# Import the new strategies 
from additional_strategies import (
     MeanReversionAgent, MomentumBreakoutAgent, VolatilityRegimeAgent,
     PairsTradingAgent, AnomalyDetectionAgent, SentimentMomentumAgent,
     RegimeChangeAgent,
 )

class StrategyAdapter:
    """
    Adapter to make new strategies compatible with StrategyTrainerAgent
    Converts between update(data) and evaluate(df) interfaces
    """
    
    def __init__(self, strategy_agent):
        self.strategy_agent = strategy_agent
        self.name = strategy_agent.name
        
    def evaluate(self, df) -> Union[float, str]:
        """
        Convert DataFrame to the format expected by new strategies
        and return a simple signal value for StrategyTrainer
        """
        try:
            # Convert DataFrame to the data format expected by new strategies
            if isinstance(df, pd.DataFrame):
                # Ensure required columns exist
                required_cols = ['close', 'high', 'low', 'open', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    # Fill missing columns with close price or zero
                    for col in missing_cols:
                        if col == 'volume':
                            df[col] = 1000  # Default volume
                        elif col in ['high', 'low', 'open']:
                            df[col] = df['close']  # Use close as fallback
                
                # Add symbol and timestamp if missing
                if 'symbol' not in df.columns:
                    df['symbol'] = 'BTC/USDT'  # Default symbol
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.Timestamp.now().timestamp()
                    
                data = {'market_data_df': df}
            else:
                data = {'market_data_df': df}
            
            # Call the strategy's update method
            result = self.strategy_agent.update(data)
            
            # Extract signal from result and convert to simple format
            signals = result.get(f"{self.strategy_agent.name}_signals", {})
            
            if not signals:
                return 0.0  # Neutral signal
                
            # Get the first signal (for single symbol) or aggregate multiple signals
            if isinstance(signals, dict):
                signal_values = []
                for symbol, signal_data in signals.items():
                    if isinstance(signal_data, dict):
                        signal = signal_data.get('signal', 'Hold')
                        confidence = signal_data.get('confidence', 0.5)
                        
                        # Convert string signals to numerical values
                        if signal in ['Strong Buy']:
                            signal_values.append(1.0 * confidence)
                        elif signal in ['Buy']:
                            signal_values.append(0.7 * confidence)
                        elif signal in ['Strong Sell']:
                            signal_values.append(-1.0 * confidence)
                        elif signal in ['Sell']:
                            signal_values.append(-0.7 * confidence)
                        else:  # Hold
                            signal_values.append(0.0)
                    else:
                        signal_values.append(0.0)
                
                # Return average signal across all symbols
                return float(np.mean(signal_values)) if signal_values else 0.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"StrategyAdapter error for {self.name}: {e}")
            return 0.0  # Return neutral on error

# Wrapper classes that inherit from your StrategyAgent pattern
class MeanReversionAgentWrapper:
    """Wrapper for MeanReversionAgent compatible with StrategyTrainer"""
    
    def __init__(self, **kwargs):
        from additional_strategies import MeanReversionAgent  # Import your strategy
        self.strategy = MeanReversionAgent(**kwargs)
        self.adapter = StrategyAdapter(self.strategy)
        self.name = "MEAN_REVERSION"
    
    def evaluate(self, df):
        return self.adapter.evaluate(df)
    
    def update(self, data):
        return self.strategy.update(data)

class MomentumBreakoutAgentWrapper:
    """Wrapper for MomentumBreakoutAgent compatible with StrategyTrainer"""
    
    def __init__(self, **kwargs):
        from additional_strategies import MomentumBreakoutAgent
        self.strategy = MomentumBreakoutAgent(**kwargs)
        self.adapter = StrategyAdapter(self.strategy)
        self.name = "MOMENTUM_BREAKOUT"
    
    def evaluate(self, df):
        return self.adapter.evaluate(df)
    
    def update(self, data):
        return self.strategy.update(data)

class VolatilityRegimeAgentWrapper:
    """Wrapper for VolatilityRegimeAgent compatible with StrategyTrainer"""
    
    def __init__(self, **kwargs):
        from additional_strategies import VolatilityRegimeAgent
        self.strategy = VolatilityRegimeAgent(**kwargs)
        self.adapter = StrategyAdapter(self.strategy)
        self.name = "VOLATILITY_REGIME"
    
    def evaluate(self, df):
        return self.adapter.evaluate(df)
    
    def update(self, data):
        return self.strategy.update(data)

class PairsTradingAgentWrapper:
    """Wrapper for PairsTradingAgent compatible with StrategyTrainer"""
    
    def __init__(self, **kwargs):
        from additional_strategies import PairsTradingAgent
        self.strategy = PairsTradingAgent(**kwargs)
        self.adapter = StrategyAdapter(self.strategy)
        self.name = "PAIRS_TRADING"
    
    def evaluate(self, df):
        return self.adapter.evaluate(df)
    
    def update(self, data):
        return self.strategy.update(data)

class AnomalyDetectionAgentWrapper:
    """Wrapper for AnomalyDetectionAgent compatible with StrategyTrainer"""
    
    def __init__(self, **kwargs):
        from additional_strategies import AnomalyDetectionAgent
        self.strategy = AnomalyDetectionAgent(**kwargs)
        self.adapter = StrategyAdapter(self.strategy)
        self.name = "ANOMALY_DETECTION"
    
    def evaluate(self, df):
        return self.adapter.evaluate(df)
    
    def update(self, data):
        return self.strategy.update(data)

class SentimentMomentumAgentWrapper:
    """Wrapper for SentimentMomentumAgent compatible with StrategyTrainer"""
    
    def __init__(self, **kwargs):
        from additional_strategies import SentimentMomentumAgent
        self.strategy = SentimentMomentumAgent(**kwargs)
        self.adapter = StrategyAdapter(self.strategy)
        self.name = "SENTIMENT_MOMENTUM"
    
    def evaluate(self, df):
        return self.adapter.evaluate(df)
    
    def update(self, data):
        return self.strategy.update(data)

class RegimeChangeAgentWrapper:
    """Wrapper for RegimeChangeAgent compatible with StrategyTrainer"""
    
    def __init__(self, **kwargs):
        from additional_strategies import RegimeChangeAgent
        self.strategy = RegimeChangeAgent(**kwargs)
        self.adapter = StrategyAdapter(self.strategy)
        self.name = "REGIME_CHANGE"
    
    def evaluate(self, df):
        return self.adapter.evaluate(df)
    
    def update(self, data):
        return self.strategy.update(data)

# Updated registration function
def register_all_strategies_with_trainer(strategy_trainer):
    """
    Register all strategies (original + new) with StrategyTrainerAgent
    This ensures they all go through the performance tracking and weighting system
    """
    try:
        # Register original strategies (your existing pattern)
        strategy_trainer.register_strategy("UT_BOT", UTSignalAgent())
        strategy_trainer.register_strategy("GRADIENT_TREND", GradientTrendAgent())
        strategy_trainer.register_strategy("VBSR", SupportResistanceAgent())
        
        # Register new strategies with wrappers
        strategy_trainer.register_strategy("MEAN_REVERSION", MeanReversionAgentWrapper())
        strategy_trainer.register_strategy("MOMENTUM_BREAKOUT", MomentumBreakoutAgentWrapper())
        strategy_trainer.register_strategy("VOLATILITY_REGIME", VolatilityRegimeAgentWrapper())
        strategy_trainer.register_strategy("PAIRS_TRADING", PairsTradingAgentWrapper())
        strategy_trainer.register_strategy("ANOMALY_DETECTION", AnomalyDetectionAgentWrapper())
        strategy_trainer.register_strategy("SENTIMENT_MOMENTUM", SentimentMomentumAgentWrapper())
        strategy_trainer.register_strategy("REGIME_CHANGE", RegimeChangeAgentWrapper())
        
        print("Successfully registered all 10 strategies with StrategyTrainer")
        return True
        
    except Exception as e:
        print(f"Failed to register strategies: {e}")
        return False

# Alternative: Direct SyncBus registration (bypasses StrategyTrainer)
def register_strategies_direct_to_syncbus(sync_bus):
    """
    Register new strategies directly to SyncBus as independent agents
    This bypasses StrategyTrainer but loses performance tracking
    """
    try:
        from additional_strategies import (
            MeanReversionAgent, MomentumBreakoutAgent, VolatilityRegimeAgent,
            PairsTradingAgent, AnomalyDetectionAgent, SentimentMomentumAgent,
            RegimeChangeAgent
        )
        
        # Attach directly to SyncBus
        sync_bus.attach("mean_reversion", MeanReversionAgent())
        sync_bus.attach("momentum_breakout", MomentumBreakoutAgent())
        sync_bus.attach("volatility_regime", VolatilityRegimeAgent())
        sync_bus.attach("pairs_trading", PairsTradingAgent())
        sync_bus.attach("anomaly_detection", AnomalyDetectionAgent())
        sync_bus.attach("sentiment_momentum", SentimentMomentumAgent())
        sync_bus.attach("regime_change", RegimeChangeAgent())
        
        print("Successfully registered 7 new strategies directly to SyncBus")
        return True
        
    except Exception as e:
        print(f"Failed to register strategies to SyncBus: {e}")
    
