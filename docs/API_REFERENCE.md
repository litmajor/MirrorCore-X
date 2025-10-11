
# API Reference

## Scanner Module (`scanner.py`)

### MomentumScanner

Main scanning class for market analysis.

```python
scanner = MomentumScanner(
    exchange,                    # CCXT exchange instance
    config=None,                 # Optional TradingConfig
    market_type='crypto',        # 'crypto' or 'forex'
    quote_currency='USDT',       # Quote currency filter
    min_volume_usd=500_000,      # Minimum daily volume
    top_n=50                     # Number of top signals
)
```

#### Methods

##### `scan_market(timeframe='daily', top_n=None) -> pd.DataFrame`
Scan market and generate signals.

**Parameters**:
- `timeframe` (str): 'scalping', 'short', 'medium', 'daily', 'weekly'
- `top_n` (int, optional): Override default top_n

**Returns**: DataFrame with columns:
- symbol, signal, state, composite_score, confidence_score
- momentum_short, momentum_long, rsi, macd
- Enhanced features: reversion_probability, cluster_validated, etc.

##### `scan_multi_timeframe(timeframes=None, save_results=True) -> pd.DataFrame`
Scan multiple timeframes for confluence.

**Parameters**:
- `timeframes` (list): List of timeframes (default: ['1h', '4h', '1d'])
- `save_results` (bool): Auto-save to CSV

**Returns**: DataFrame with additional columns:
- multi_tf_confirmed, confluence_signal, confluence_zone

##### `get_strong_signals(timeframe='daily', min_score=70.0) -> pd.DataFrame`
Filter for high-quality signals.

**Parameters**:
- `timeframe` (str): Filter by timeframe
- `min_score` (float): Minimum composite score

**Returns**: Filtered DataFrame

##### `get_reversion_candidates(timeframe='daily') -> pd.DataFrame`
Get mean reversion opportunities.

**Returns**: DataFrame sorted by reversion_probability

##### `get_cluster_validated_momentum(timeframe='daily') -> pd.DataFrame`
Get momentum signals validated by clustering.

**Returns**: DataFrame filtered for cluster_validated=True

### TechnicalIndicators

Static methods for indicator calculation.

```python
# RSI
rsi = TechnicalIndicators.calculate_rsi(prices, period=14)

# MACD
macd = TechnicalIndicators.calculate_macd(prices, fast=12, slow=26, signal=9)

# Momentum
momentum = TechnicalIndicators.calculate_momentum(prices, period=10)

# Fibonacci Levels
fib_data = TechnicalIndicators.fib_levels(df, lookback=55)

# Volume Profile
volume_hist, poc_price = TechnicalIndicators.calculate_volume_profile(df, bins=50)

# Composite Score
score = TechnicalIndicators.calculate_composite_score(
    momentum_short, momentum_long, rsi, macd,
    trend_score, volume_ratio, ichimoku_bullish,
    fib_confluence, weights=None
)
```

### SignalClassifier

Signal classification methods.

```python
# Momentum Signal
signal = SignalClassifier.classify_momentum_signal(
    momentum_short, momentum_long, rsi, macd,
    thresholds={'momentum_short': 0.01, 'rsi_min': 33, ...},
    additional_indicators={'ichimoku_bullish': True, ...}
)
# Returns: 'Strong Buy', 'Buy', 'Weak Buy', 'Neutral', 'Weak Sell', 'Sell', 'Strong Sell'

# State Classification
state = SignalClassifier.classify_state(
    mom1d, mom7d, mom30d, rsi, macd, bb_pos, vol_ratio
)
# Returns: 'BULL_EARLY', 'BULL_STRONG', 'BULL_PARABOLIC', etc.

# Legacy Signal
legacy = SignalClassifier.classify_legacy(
    mom7d, mom30d, rsi, macd, bb_position, volume_ratio
)
# Returns: 'Consistent Uptrend', 'New Spike', 'Topping Out', etc.
```

## RL Trading System (`rl_trading_system.py`)

### RLTradingAgent

Main RL agent wrapper.

```python
from rl_trading_system import RLTradingAgent, RLConfig

config = RLConfig(
    lookback_window=20,
    learning_rate=3e-4,
    total_timesteps=100000
)

agent = RLTradingAgent(algorithm='PPO', config=config)
```

#### Methods

##### `train(scanner_results, save_path='rl_trading_model', verbose=1) -> dict`
Train the RL agent.

**Parameters**:
- `scanner_results` (pd.DataFrame): Training data from scanner
- `save_path` (str): Model save location
- `verbose` (int): 0=silent, 1=progress bars

**Returns**: Training results dict

##### `evaluate(test_data, episodes=10) -> dict`
Evaluate trained model.

**Returns**:
```python
{
    'mean_reward': float,
    'std_reward': float,
    'mean_return': float,
    'sharpe_ratio': float,
    'mean_win_rate': float
}
```

##### `predict(observation, deterministic=True) -> tuple`
Predict action for observation.

**Parameters**:
- `observation` (np.ndarray): Shape (20, 21)
- `deterministic` (bool): Use deterministic policy

**Returns**: (action, None)

### MetaController

Blend rule-based and RL signals.

```python
from rl_trading_system import MetaController

meta = MetaController(strategy='confidence_blend')
# Strategies: 'rule_only', 'rl_only', 'confidence_blend', 'regime_adaptive'

decision = meta.decide(
    rule_decision='Strong Buy',
    rl_action=np.array([0.8]),
    confidence_score=0.75,
    market_regime='trending'
)
```

**Returns**:
```python
{
    'final_position': float,      # -1.0 to 1.0
    'rule_position': float,
    'rl_position': float,
    'confidence_score': float,
    'method': str,                 # 'rl_weighted', 'balanced', etc.
    'market_regime': str
}
```

### IntegratedTradingSystem

Full system integration.

```python
from rl_trading_system import IntegratedTradingSystem

system = IntegratedTradingSystem(scanner, rl_agent, meta_controller)

# Generate signals
signals = await system.generate_signals(timeframe='daily')

# Execute trading session
session = await system.execute_trading_session(
    timeframe='daily',
    dry_run=True  # Paper trading
)

# Get performance
report = system.get_performance_report()
```

## Strategy Trainer (`strategy_trainer_agent.py`)

### StrategyTrainerAgent

Manage multiple strategies.

```python
from strategy_trainer_agent import StrategyTrainerAgent

trainer = StrategyTrainerAgent(
    min_weight=0.1,
    max_weight=1.0,
    lookback_window=20
)

# Register strategy
trainer.register_strategy("MY_STRATEGY", strategy_instance)

# Evaluate on data
signals = trainer.evaluate(market_df)

# Update performance
trainer.update_performance("MY_STRATEGY", pnl=150.0)

# Get grades
grades = trainer.grade_strategies()
# Returns: {'STRATEGY_NAME': {'grade': 'A', 'avg_pnl': 0.05, ...}, ...}
```

## Trade Analyzer (`trade_analyzer_agent.py`)

### TradeAnalyzerAgent

Track and analyze trades.

```python
from trade_analyzer_agent import TradeAnalyzerAgent

analyzer = TradeAnalyzerAgent()

# Record trade
analyzer.record_trade({
    'symbol': 'BTC/USDT',
    'entry': 50000,
    'exit': 51000,
    'pnl': 100,
    'strategy': 'momentum',
    'timestamp': time.time()
})

# Get summary
analyzer.summary(top_n=5)

# Performance metrics
analyzer.performance_metrics()

# Export
analyzer.export_to_csv('trades.csv')

# Get drawdown stats
dd_stats = analyzer.get_drawdown_stats()
```

## Additional Strategies (`additional_strategies.py`)

### Strategy Classes

All inherit from `BaseStrategyAgent` with `evaluate(df)` method.

```python
from additional_strategies import (
    MeanReversionAgent,
    MomentumBreakoutAgent,
    VolatilityRegimeAgent,
    PairsTradingAgent,
    AnomalyDetectionAgent,
    SentimentMomentumAgent,
    RegimeChangeAgent
)

# Initialize
mean_rev = MeanReversionAgent(bb_period=20, rsi_period=14)

# Update with data
result = mean_rev.update({'market_data_df': df})
# Returns: {'MEAN_REVERSION_signals': {symbol: {'signal': 'Buy', 'confidence': 0.8}}}

# Or evaluate directly
score = mean_rev.evaluate(df)
# Returns: float signal score
```

### Ensemble Function

```python
from additional_strategies import ensemble_signal

consensus = ensemble_signal(
    agents_outputs,                    # Dict of agent outputs
    weights=None,                      # Optional static weights
    meta_model=None,                   # Optional sklearn model
    majority_vote=False,               # Use majority voting
    bayesian_averaging=False,          # Bayesian model averaging
    min_agree_count=None,              # Minimum agreeing agents
    require_signals={'AGENT': 'Buy'},  # Required confirmations
    explainability=True                # Include explanations
)
```

**Returns**:
```python
{
    'symbol': {
        'consensus': float,
        'direction': 'Buy'|'Sell'|'Hold',
        'explain': {...}  # If explainability=True
    }
}
```

## Configuration (`config_manager.py`)

### TradingConfig

```python
from scanner import TradingConfig

config = TradingConfig(
    timeframes={
        'daily': '1d',
        'weekly': '1w'
    },
    backtest_periods={
        'daily': 7,
        'weekly': 4
    },
    momentum_periods={
        'crypto': {
            'daily': {'short': 7, 'long': 30}
        }
    },
    signal_thresholds={
        'crypto': {
            'daily': {
                'momentum_short': 0.06,
                'rsi_min': 33,
                'rsi_max': 65
            }
        }
    },
    lookback_window=68,
    rsi_threshold=33.0,
    volume_multiplier=3.2
)
```

### RLConfig

```python
from rl_trading_system import RLConfig

rl_config = RLConfig(
    lookback_window=20,
    max_position_size=1.0,
    transaction_cost=0.003,
    slippage=0.0005,
    reward_scaling=3.0,
    learning_rate=3e-4,
    total_timesteps=100000,
    eval_freq=5000,
    save_freq=10000
)
```

## Data Types

### Scanner Output DataFrame

```python
pd.DataFrame({
    'symbol': str,
    'signal': str,              # Traditional signal
    'state': str,               # Market state
    'legacy_signal': str,       # Legacy label
    'composite_score': float,   # 0-100
    'confidence_score': float,  # 0.0-1.0
    'momentum_short': float,
    'momentum_long': float,
    'rsi': float,
    'macd': float,
    'volume_ratio': float,
    
    # Enhanced features
    'enhanced_momentum_score': float,
    'cluster_validated': bool,
    'momentum_strength': str,   # 'weak', 'moderate', 'strong'
    'reversion_probability': float,
    'reversion_candidate': bool,
    'volatility_regime': str,   # 'low', 'normal', 'high'
    'trend_regime': str,        # 'sideways', 'up', 'down', etc.
    'cluster_detected': bool,
    'trend_formation_signal': bool
})
```

### Trading Signal Format

```python
{
    'symbol': 'BTC/USDT',
    'signal_type': 'Strong Buy',
    'rule_position': 1.0,
    'rl_position': 0.8,
    'final_position': 0.9,
    'confidence': 0.75,
    'method': 'rl_weighted',
    'price': 50000.0,
    'composite_score': 85.5,
    'timestamp': datetime,
    'dry_run': bool
}
```

## Error Handling

All async methods should be wrapped in try-except:

```python
try:
    results = await scanner.scan_market()
except Exception as e:
    logger.error(f"Scan failed: {e}")
    # Handle error
```

Common exceptions:
- `ccxt.NetworkError`: Exchange connectivity issues
- `ccxt.RateLimitExceeded`: API rate limit hit
- `ValueError`: Invalid configuration
- `KeyError`: Missing required data fields
