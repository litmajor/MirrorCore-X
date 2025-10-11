
# MirrorCore-X System Architecture

## Overview

MirrorCore-X is a multi-agent reinforcement learning trading system with modular architecture supporting:
- Real-time market scanning
- Traditional + AI-driven signal generation
- Multi-strategy ensemble
- Adaptive risk management
- Live/paper trading execution

## System Components

### 1. Core Scanning Engine

**Module**: `scanner.py`

**Components**:
- `MomentumScanner`: Main scanning class
- `MarketDataFetcher`: Data retrieval and caching
- `TechnicalIndicators`: Indicator calculation
- `SignalClassifier`: Signal classification

**Responsibilities**:
- Fetch market data from exchanges (CCXT) or adapters (yfinance)
- Calculate 30+ technical indicators
- Generate traditional trading signals
- Perform multi-timeframe analysis
- Detect market regimes and patterns

**Outputs**:
- `scan_results` DataFrame with 40+ features per symbol
- CSV exports for historical analysis

### 2. Reinforcement Learning System

**Module**: `rl_trading_system.py`

**Components**:
- `RLTradingAgent`: Main RL wrapper
- `TradingEnvironment`: Gymnasium-compatible env
- `SignalEncoder`: Feature preprocessing
- `MetaController`: Rule-based + RL blending

**Algorithms Supported**:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- A2C (Advantage Actor-Critic)

**Training Pipeline**:
1. Prepare data from scanner results
2. Encode features (normalize to [-1, 1])
3. Create trading environment
4. Train RL model (PPO default)
5. Evaluate on test set
6. Save model + encoder

**Observation Space**: (20, 21) matrix
- 20 timesteps lookback
- 21 features: price, volume, momentum, RSI, MACD, etc.

**Action Space**: Continuous [-1.0, 1.0]
- -1.0: Max short position
- 0.0: Neutral (no position)
- +1.0: Max long position

### 3. Strategy Management

**Module**: `strategy_trainer_agent.py`

**Components**:
- `StrategyTrainerAgent`: Strategy orchestrator
- Strategy wrappers: `UTSignalAgent`, `GradientTrendAgent`, etc.

**Responsibilities**:
- Register multiple strategies
- Track performance per strategy
- Dynamically adjust strategy weights
- Grade strategies (A-F based on PnL, win rate, Sharpe)

**Weight Adjustment Algorithm**:
```python
performance_score = (
    (avg_pnl * 0.1) +
    (win_rate - 0.5) * 0.5 +
    clip(sharpe * 0.1, -0.2, 0.2)
)
new_weight = current_weight + (performance_score * 0.1)
new_weight = clip(new_weight, min_weight, max_weight)
```

### 4. Additional Strategies Ensemble

**Module**: `additional_strategies.py`

**7 Complementary Strategies**:
1. **Mean Reversion**: Bollinger Bands + RSI z-score
2. **Momentum Breakout**: ATR-based with volume confirmation
3. **Volatility Regime**: Adaptive strategy selection
4. **Pairs Trading**: Statistical arbitrage
5. **Anomaly Detection**: Isolation Forest ML
6. **Sentiment Momentum**: Price-volume sentiment
7. **Regime Change**: Hidden Markov Model concepts

**Ensemble Methods**:
- Weighted voting
- Majority voting with tie-breaks
- Stacked ensemble (meta-model)
- Dynamic weighting
- Bayesian model averaging
- Regime-switching
- Correlation filtering

### 5. Trading Execution

**Module**: `mirrorcore_x.py` (main integrated system)

**Components**:
- `IntegratedTradingSystem`: Coordinates all agents
- `ExecutionDaemon`: (planned) Trade execution
- `TradeAnalyzerAgent`: Performance tracking

**Flow**:
1. Generate signals from scanner + RL
2. Meta-controller blends decisions
3. Filter actionable signals (position > 0.1)
4. Execute trades (real or paper)
5. Record in trading history
6. Update strategy performance

### 6. Risk Management

**Module**: `risk_management.py`

**Features**:
- Position sizing
- Stop-loss management
- Drawdown protection
- Portfolio heat limits
- Correlation-based diversification

### 7. Analytics & Monitoring

**Modules**: 
- `trade_analyzer_agent.py`: Trade performance
- `dashboard_server.py`: Real-time monitoring
- `weekly_report.py`: Automated reporting

**Metrics Tracked**:
- Win rate, profit factor, Sharpe ratio
- Maximum drawdown, recovery time
- Strategy-level performance
- Symbol-level profitability

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Market Data Sources                      │
│  (CCXT Exchanges, yfinance, Alternative Data)               │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              MarketDataFetcher (scanner.py)                  │
│  - Async data retrieval                                     │
│  - Caching (5 min TTL)                                      │
│  - Rate limiting                                             │
│  - Circuit breaker                                           │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│           TechnicalIndicators (scanner.py)                   │
│  - RSI, MACD, Bollinger Bands                               │
│  - Ichimoku, VWAP, EMAs                                     │
│  - Volume Profile, Fibonacci                                │
│  - ATR, ADX, OBV                                            │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              SignalClassifier (scanner.py)                   │
│  - Traditional signals (Buy/Sell/Neutral)                   │
│  - State classification (BULL_STRONG, etc.)                 │
│  - Legacy labels (Consistent Uptrend, etc.)                 │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
        ┌──────────────┴──────────────┐
        ↓                              ↓
┌──────────────────┐         ┌──────────────────────┐
│  Enhanced        │         │  Additional          │
│  Analysis        │         │  Strategies          │
│  (scanner.py)    │         │  (7 agents)          │
│  - Momentum      │         │  - Mean Reversion    │
│  - Reversion     │         │  - Breakout          │
│  - Regime        │         │  - Pairs Trading     │
│  - Clustering    │         │  - etc.              │
└────────┬─────────┘         └──────────┬───────────┘
         ↓                              ↓
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  scanner.scan_results        │
         │  (DataFrame with all signals)│
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  SignalEncoder               │
         │  (rl_trading_system.py)      │
         │  - Normalize features        │
         │  - Create observation        │
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  RLTradingAgent              │
         │  (rl_trading_system.py)      │
         │  - PPO/SAC/A2C model         │
         │  - Predict position          │
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  MetaController              │
         │  (rl_trading_system.py)      │
         │  - Blend rule + RL signals   │
         │  - Regime adaptation         │
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  IntegratedTradingSystem     │
         │  (mirrorcore_x.py)           │
         │  - Filter actionable signals │
         │  - Execute trades            │
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  TradeAnalyzerAgent          │
         │  (trade_analyzer_agent.py)   │
         │  - Record trades             │
         │  - Calculate metrics         │
         │  - Generate reports          │
         └──────────────────────────────┘
```

## Module Dependencies

```
scanner.py
  ├── ccxt.async_support (market data)
  ├── ta (technical indicators)
  ├── pandas, numpy (data processing)
  └── services/yfinance_adapter.py (forex data)

rl_trading_system.py
  ├── scanner.py (data source)
  ├── stable_baselines3 (RL algorithms)
  ├── gymnasium (environment)
  └── sklearn (preprocessing)

additional_strategies.py
  ├── scanner.py (data source)
  ├── talib (indicators)
  ├── sklearn.ensemble (IsolationForest)
  └── scipy.stats (statistical methods)

strategy_trainer_agent.py
  ├── strategies/ (UT Bot, Gradient, VBSR)
  └── additional_strategies.py (7 agents)

mirrorcore_x.py
  ├── scanner.py
  ├── rl_trading_system.py
  ├── strategy_trainer_agent.py
  ├── trade_analyzer_agent.py
  └── risk_management.py
```

## Configuration System

**Module**: `config_manager.py`

**Key Configurations**:

### TradingConfig
```python
@dataclass
class TradingConfig:
    timeframes: Dict[str, str]           # Timeframe mappings
    backtest_periods: Dict[str, int]     # Lookback periods
    momentum_periods: Dict[...]          # Momentum calculations
    signal_thresholds: Dict[...]         # Signal generation
    trade_durations: Dict[str, int]      # Expected hold times
    lookback_window: int = 68            # Historical data window
    rsi_threshold: float = 33.0          # RSI oversold
    volume_multiplier: float = 3.2       # Volume surge
```

### RLConfig
```python
@dataclass
class RLConfig:
    lookback_window: int = 20
    max_position_size: float = 1.0
    transaction_cost: float = 0.003
    slippage: float = 0.0005
    reward_scaling: float = 3.0
    learning_rate: float = 3e-4
    total_timesteps: int = 100000
```

## Performance Optimization

### Caching Strategy
- Market data: 5-minute TTL
- Technical indicators: Computed once per scan
- RL observations: Reused during evaluation

### Async Operations
- Market data fetching (parallel)
- Multi-symbol processing (concurrent)
- Rate limiting with semaphores

### Resource Management
- Dynamic concurrent request limits (CPU-based)
- Circuit breaker for API failures
- Exponential backoff for retries

## Extensibility Points

### Adding New Strategies
```python
# 1. Create strategy class with evaluate() method
class MyStrategy(BaseStrategyAgent):
    def evaluate(self, df):
        # Return signal score
        return signal_value

# 2. Register with trainer
trainer.register_strategy("MY_STRATEGY", MyStrategy())
```

### Adding New Indicators
```python
# In scanner.py TechnicalIndicators class
@staticmethod
def calculate_my_indicator(df: pd.DataFrame) -> pd.Series:
    # Compute indicator
    return indicator_series

# Use in add_all_indicators()
df['my_indicator'] = TechnicalIndicators.calculate_my_indicator(df)
```

### Custom Signal Classification
```python
# In scanner.py SignalClassifier class
@staticmethod
def classify_my_signal(indicators: dict) -> str:
    # Custom logic
    return signal_label
```

## Deployment Architecture

### Development Mode
```
Replit IDE → Python Process → CCXT Test API
              ↓
         scan_results.csv
              ↓
         Local Storage
```

### Production Mode
```
Replit Deployment → Python Process → CCXT Production API
                     ↓
                  Real Trades
                     ↓
              Trade Database
                     ↓
              Analytics Dashboard
```

## Security & Secrets

**Module**: `secrets_manager.py`

**Managed Secrets**:
- Exchange API keys
- Database credentials
- Third-party API tokens

**Access Pattern**:
```python
from secrets_manager import get_secret
api_key = get_secret('EXCHANGE_API_KEY')
```

## Monitoring & Observability

### Logging Levels
- DEBUG: Detailed execution traces
- INFO: Key events (trades, signals)
- WARNING: Degraded performance
- ERROR: Failures requiring attention

### Key Metrics
- Scanner: Symbols scanned, signals generated, scan duration
- RL: Episode rewards, win rate, Sharpe ratio
- Execution: Trade count, success rate, latency
- Portfolio: PnL, drawdown, exposure

## Scaling Considerations

### Horizontal Scaling
- Multiple scanner instances (different exchanges)
- Distributed strategy evaluation
- Load-balanced signal generation

### Vertical Scaling
- Increased concurrent requests (CPU cores)
- Larger lookback windows (memory)
- More frequent scans (I/O optimization)

## Testing Strategy

### Unit Tests
- Individual indicator calculations
- Signal classification logic
- Strategy evaluation methods

### Integration Tests
- Scanner → RL pipeline
- Multi-agent coordination
- Trade execution flow

### Backtests
- Historical signal quality
- Strategy performance
- Risk metric validation

## Roadmap

### Completed ✅
- Multi-timeframe scanning
- RL integration (PPO/SAC/A2C)
- 7 additional strategies
- Meta-controller blending
- Enhanced signal features

### In Progress 🚧
- Real-time execution daemon
- Live dashboard (WebSocket)
- Automated reporting

### Planned 📋
- Multi-exchange scanning
- Options/futures support
- Advanced portfolio optimization
- Distributed backtesting
