
# Trading Strategies Documentation

## Overview

MirrorCore-X includes 10+ trading strategies across multiple categories:
- **Core Strategies** (3): UT Bot, Gradient Trend, Volume SR
- **Additional Strategies** (7): Mean Reversion, Breakout, Pairs Trading, etc.
- **Meta Strategies**: RL-based, Ensemble blending

## Core Strategies

### 1. UT Bot Strategy

**Module**: `strategies/ut_bot.py`

**Description**: Chandelier Exit-based trend following with ATR volatility adjustment.

**Key Features**:
- Dynamic stop-loss using ATR
- Trend reversal detection
- Configurable sensitivity

**Parameters**:
```python
UTBotStrategy(
    atr_period=10,        # ATR calculation period
    atr_multiplier=1.0,   # Stop distance multiplier
    sensitivity=1         # Signal sensitivity
)
```

**Signals**:
- `BUY`: Price crosses above trailing stop
- `SELL`: Price crosses below trailing stop
- `HOLD`: No clear signal

**Best For**: Trending markets, medium-term trades

### 2. Gradient Trend Filter

**Module**: `strategies/gradient_trend_filter.py`

**Description**: Multi-timeframe trend detection using price gradients.

**Key Features**:
- Trend strength measurement
- Multi-timeframe confirmation
- Gradient acceleration detection

**Parameters**:
```python
GradientTrendFilter(
    short_window=10,
    long_window=30,
    threshold=0.02
)
```

**Signals**:
- `BUY`: Strong upward gradient
- `SELL`: Strong downward gradient
- `HOLD`: Weak or mixed gradients

**Best For**: Identifying established trends

### 3. Volume Support/Resistance

**Module**: `strategies/volume_sr_agent.py`

**Description**: Support/resistance levels based on volume profile.

**Key Features**:
- High-volume node detection
- Dynamic level adjustment
- Breakout confirmation

**Parameters**:
```python
VolumeSupportResistance(
    lookback_period=50,
    volume_threshold=1.5,
    level_tolerance=0.02
)
```

**Signals**:
- `BUY`: Bounce off support with volume
- `SELL`: Rejection at resistance with volume
- `HOLD`: Between levels

**Best For**: Range-bound markets, key level trading

## Additional Strategies

### 4. Mean Reversion

**Module**: `additional_strategies.py` → `MeanReversionAgent`

**Description**: Bollinger Bands + RSI-based reversion trading.

**Key Features**:
- Z-score deviation detection
- RSI extreme confirmation
- Statistical mean calculation

**Parameters**:
```python
MeanReversionAgent(
    bb_period=20,
    bb_std=2.0,
    rsi_period=14,
    zscore_threshold=2.0
)
```

**Entry Signals**:
- `Strong Buy`: Z-score < -2.0 AND RSI < 30 (oversold)
- `Strong Sell`: Z-score > 2.0 AND RSI > 70 (overbought)
- `Hold`: |Z-score| < 0.5 (near mean)

**Exit Signals**: Price returns to mean (Z-score ≈ 0)

**Risk**: Trending markets (gets caught wrong side)

### 5. Momentum Breakout

**Module**: `additional_strategies.py` → `MomentumBreakoutAgent`

**Description**: ATR-based breakout detection with volume confirmation.

**Key Features**:
- High/low breakout identification
- Volume surge confirmation (>1.5x avg)
- ATR-adjusted stops

**Parameters**:
```python
MomentumBreakoutAgent(
    atr_period=14,
    breakout_multiplier=2.0,
    volume_threshold=1.5,
    lookback_period=20
)
```

**Entry Signals**:
- `Strong Buy`: Price > 20-day high AND volume > 1.5x avg AND momentum > 2%
- `Strong Sell`: Price < 20-day low AND volume > 1.5x avg AND momentum < -2%

**Best For**: Volatile breakout markets

### 6. Volatility Regime

**Module**: `additional_strategies.py` → `VolatilityRegimeAgent`

**Description**: Adapts strategy based on market volatility state.

**Key Features**:
- Volatility percentile ranking
- Regime classification (HIGH/MEDIUM/LOW)
- Strategy switching

**Parameters**:
```python
VolatilityRegimeAgent(
    vol_window=20,
    regime_threshold=1.5
)
```

**Strategy Logic**:
- **High Volatility** (>80th percentile): Trend following (MACD)
- **Low Volatility** (<20th percentile): Mean reversion (contrarian)
- **Medium Volatility**: Neutral/Hold

**Signals**:
- Depends on regime + MACD/RSI

**Best For**: Adaptive trading across market conditions

### 7. Pairs Trading

**Module**: `additional_strategies.py` → `PairsTradingAgent`

**Description**: Statistical arbitrage between correlated pairs.

**Key Features**:
- Co-integration detection
- Spread z-score trading
- Market-neutral approach

**Parameters**:
```python
PairsTradingAgent(
    lookback_period=60,
    zscore_entry=2.0,
    zscore_exit=0.5,
    min_correlation=0.7
)
```

**Entry Logic**:
- Find pairs with correlation > 0.7
- Calculate spread = price1 - price2
- Z-score = (spread - mean) / std

**Signals**:
- `Long A, Short B`: Z-score > 2.0 (spread too high)
- `Short A, Long B`: Z-score < -2.0 (spread too low)
- `Exit`: |Z-score| < 0.5

**Best For**: Market-neutral strategies, low correlation markets

### 8. Anomaly Detection

**Module**: `additional_strategies.py` → `AnomalyDetectionAgent`

**Description**: Machine learning-based anomaly detection using Isolation Forest.

**Key Features**:
- Unsupervised learning
- Price/volume pattern recognition
- Contrarian approach

**Parameters**:
```python
AnomalyDetectionAgent(
    contamination=0.1,    # Expected anomaly %
    n_estimators=100      # Forest size
)
```

**Features Used**:
- Price change, volatility, momentum
- Volume change, volume ratio
- RSI, Bollinger position
- Spread proxy, price impact

**Signals**:
- `Buy`: Strong negative anomaly (price likely oversold)
- `Sell`: Strong positive anomaly (price likely overbought)
- `Hold`: Normal behavior

**Best For**: Catching reversals, flash crashes

### 9. Sentiment Momentum

**Module**: `additional_strategies.py` → `SentimentMomentumAgent`

**Description**: Combines price momentum with implied market sentiment.

**Key Features**:
- Price-volume sentiment proxy
- Multi-timeframe momentum
- Sentiment-weighted positions

**Parameters**:
```python
SentimentMomentumAgent(
    short_period=5,
    long_period=20,
    sentiment_weight=0.3
)
```

**Sentiment Calculation**:
```python
sentiment = price_change * log(volume / avg_volume)
# Positive: Strong move with high volume (bullish)
# Negative: Strong move down with high volume (bearish)
```

**Signals**:
- `Strong Buy`: Bullish momentum + positive sentiment
- `Strong Sell`: Bearish momentum + negative sentiment

**Best For**: News-driven markets, high-volume events

### 10. Regime Change Detection

**Module**: `additional_strategies.py` → `RegimeChangeAgent`

**Description**: Detects structural breaks in market behavior using HMM concepts.

**Key Features**:
- Volatility clustering detection
- Trend/range regime classification
- Early warning for shifts

**Parameters**:
```python
RegimeChangeAgent(
    window_size=50,
    sensitivity=2.0
)
```

**Regime Types**:
- `HIGH_VOLATILITY`: Vol > 80th percentile
- `LOW_VOLATILITY`: Vol < 20th percentile
- `TRENDING`: Trend strength > 70th percentile
- `RANGING`: Otherwise

**Signals** (on regime change):
- `TRENDING`: Follow trend direction
- `HIGH_VOLATILITY`: Hold/wait for direction
- `RANGING`: Mean reversion

**Best For**: Adaptive strategy switching

## Meta Strategies

### 11. Reinforcement Learning (RL)

**Module**: `rl_trading_system.py`

**Description**: PPO/SAC/A2C deep RL agents trained on historical signals.

**Key Features**:
- Learns optimal position sizing
- Adapts to changing markets
- Risk-aware decision making

**Training**:
```python
agent = RLTradingAgent(algorithm='PPO')
agent.train(scanner_results, total_timesteps=100000)
```

**Action Space**: Continuous [-1.0, 1.0] position size

**Reward Function**:
```python
reward = (
    step_pnl / initial_balance +
    -drawdown_penalty +
    -risk_penalty
) * reward_scaling
```

**Best For**: Complex market dynamics, portfolio optimization

### 12. Ensemble Blending

**Module**: `additional_strategies.py` → `ensemble_signal()`

**Description**: Combines multiple strategy signals using various methods.

**Methods**:
- **Weighted Voting**: Static or dynamic weights
- **Majority Voting**: Democratic consensus
- **Bayesian Averaging**: Probabilistic blending
- **Stacked Ensemble**: Meta-model on strategy outputs
- **Regime-Switching**: Different weights per regime

**Example**:
```python
consensus = ensemble_signal(
    agents_outputs,
    weights={'MEAN_REVERSION': 1.5, 'MOMENTUM': 1.2},
    majority_vote=False,
    min_agree_count=3  # Require 3 strategies agreeing
)
```

**Best For**: Diversification, reducing false signals

## Strategy Comparison

| Strategy | Type | Timeframe | Win Rate* | Sharpe* | Best Market |
|----------|------|-----------|-----------|---------|-------------|
| UT Bot | Trend | Medium | 55-60% | 1.2 | Trending |
| Gradient Trend | Trend | Long | 50-55% | 1.0 | Strong trends |
| Volume SR | S/R | Short | 60-65% | 1.5 | Range-bound |
| Mean Reversion | Reversion | Short | 65-70% | 1.8 | Ranging |
| Momentum Breakout | Breakout | Short | 45-50% | 0.8 | Volatile |
| Volatility Regime | Adaptive | Any | 55-60% | 1.3 | Mixed |
| Pairs Trading | Arbitrage | Medium | 60-65% | 1.4 | Stable |
| Anomaly Detection | ML | Short | 50-55% | 0.9 | Event-driven |
| Sentiment Momentum | Hybrid | Short | 55-60% | 1.1 | News-driven |
| Regime Change | Adaptive | Long | 50-55% | 1.0 | Transitional |
| RL Agent | Adaptive | Any | 60-70% | 1.6 | All markets |
| Ensemble | Meta | Any | 65-75% | 1.9 | All markets |

*Estimated based on typical backtests

## Usage Recommendations

### Market Conditions

**Trending Markets**:
- Primary: UT Bot, Gradient Trend, RL Agent
- Avoid: Mean Reversion, Pairs Trading

**Range-Bound Markets**:
- Primary: Mean Reversion, Volume SR, Pairs Trading
- Avoid: Momentum Breakout, Trend strategies

**Volatile Markets**:
- Primary: Volatility Regime, Anomaly Detection
- Avoid: Mean Reversion (whipsaws)

**Mixed/Uncertain**:
- Primary: Ensemble, RL Agent, Regime Change
- Use adaptive strategies

### Risk Profiles

**Conservative**:
- Mean Reversion (defined risk)
- Pairs Trading (market-neutral)
- Volume SR (clear levels)

**Moderate**:
- UT Bot
- Gradient Trend
- Ensemble (diversified)

**Aggressive**:
- Momentum Breakout
- RL Agent (max learning)
- Sentiment Momentum

### Timeframe Matching

**Scalping (1m-5m)**:
- Volume SR
- Mean Reversion (tight stops)
- Anomaly Detection

**Intraday (5m-1h)**:
- Momentum Breakout
- Sentiment Momentum
- UT Bot (short-term)

**Swing (1h-1d)**:
- Gradient Trend
- Volatility Regime
- RL Agent

**Position (1d+)**:
- Pairs Trading
- Regime Change
- Ensemble

## Combining Strategies

### Complementary Pairs

1. **Trend + Reversion**:
   - UT Bot (trend) + Mean Reversion (counter-trend)
   - Trade trend continuation OR exhaustion reversals

2. **Momentum + Support**:
   - Momentum Breakout + Volume SR
   - Breakouts at key levels only

3. **RL + Traditional**:
   - RL Agent + any strategy via Meta-Controller
   - RL learns to weight traditional signals

### Strategy Stacking

```python
# 1. Get primary signal
primary = gradient_trend.evaluate(df)

# 2. Confirm with secondary
if primary == 'BUY':
    confirm = volume_sr.evaluate(df)
    if confirm == 'BUY':
        execute_trade()

# 3. Meta-confirmation
ensemble = ensemble_signal(all_strategies)
if ensemble['consensus'] > 0.7:
    execute_trade()
```

## Performance Optimization

### Strategy-Specific Tips

**Mean Reversion**:
- Use shorter BB periods (14-18) for crypto
- Increase zscore threshold (2.5-3.0) for fewer signals
- Exit at mean (zscore=0) or opposite extreme

**Momentum Breakout**:
- Filter by ADX > 25 (strong trend)
- Require volume > 2x average for validity
- Use ATR for stop-loss placement

**RL Agent**:
- Train on ≥50,000 timesteps
- Use PPO for stable convergence
- Retrain monthly on recent data

**Ensemble**:
- Weight by recent Sharpe ratio
- Exclude strategies in drawdown (>10%)
- Require ≥3 strategies agreeing

## Common Pitfalls

1. **Over-Optimization**: Fitting to historical data
   - Solution: Walk-forward testing, out-of-sample validation

2. **Strategy Drift**: Performance degrades over time
   - Solution: Regular retraining, adaptive weighting

3. **Conflicting Signals**: Strategies disagree
   - Solution: Use ensemble consensus, meta-controller

4. **Regime Mismatch**: Wrong strategy for market
   - Solution: Volatility/Regime adaptive switching

5. **Overfitting Indicators**: Too many parameters
   - Solution: Simplify, use robust defaults

## Backtesting Strategies

```python
from vectorized_backtest import vectorized_backtest

# Backtest single strategy
results = vectorized_backtest(
    df=historical_data,
    strategy_func=mean_reversion_agent.evaluate,
    initial_capital=10000
)

# Compare multiple strategies
for strategy in [ut_bot, mean_rev, ensemble]:
    results = vectorized_backtest(df, strategy.evaluate)
    print(f"{strategy.name}: Sharpe={results['sharpe']:.2f}")
```

See `BACKTEST_GUIDE.md` for full backtesting documentation.
