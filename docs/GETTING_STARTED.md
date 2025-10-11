
# Getting Started with MirrorCore-X

## Quick Start (5 Minutes)

### 1. Scan the Market
```python
import asyncio
import ccxt.async_support as ccxt
from scanner import MomentumScanner

# Initialize exchange
exchange = ccxt.binance({'enableRateLimit': True})

# Create scanner
scanner = MomentumScanner(
    exchange=exchange,
    market_type='crypto',
    quote_currency='USDT',
    min_volume_usd=500_000,
    top_n=20
)

# Scan for daily signals
async def quick_scan():
    results = await scanner.scan_market(timeframe='daily')
    print(f"Found {len(results)} signals")
    print(results[['symbol', 'signal', 'composite_score']].head())
    await exchange.close()

asyncio.run(quick_scan())
```

### 2. Get Strong Signals
```python
# Filter high-quality signals
strong_signals = scanner.get_strong_signals(min_score=70.0)
print(f"\n{len(strong_signals)} strong signals:")
print(strong_signals[['symbol', 'signal', 'confidence_score', 'composite_score']])
```

### 3. Check Mean Reversion Opportunities
```python
# Find overbought/oversold candidates
reversion = scanner.get_reversion_candidates()
print(f"\n{len(reversion)} mean reversion candidates:")
print(reversion[['symbol', 'reversion_probability', 'signal']])
```

## Full Trading System Setup

### Step 1: Train RL Agent
```python
from rl_trading_system import RLTrainingPipeline

# Initialize pipeline
pipeline = RLTrainingPipeline(scanner)

# Run full training (includes optimization)
async def train_system():
    results = await pipeline.run_full_pipeline(
        training_timeframe='daily',
        algorithm='PPO',
        total_timesteps=50000,
        n_opt_trials=10
    )
    
    print(f"Training complete: {results['training_results']}")
    print(f"Evaluation: {results['evaluation_results']}")
    print(f"Test session: {results['test_session']['trades_executed']} trades")

asyncio.run(train_system())
```

### Step 2: Execute Trading Session
```python
from rl_trading_system import IntegratedTradingSystem, MetaController

# Get trained components
integrated_system = pipeline.integrated_system

# Run dry-run trading session
async def trade_session():
    session = await integrated_system.execute_trading_session(
        timeframe='daily',
        dry_run=True  # Set False for real trading
    )
    
    print(f"Session status: {session['status']}")
    print(f"Signals: {session['total_signals']}")
    print(f"Executed: {session['trades_executed']} trades")

asyncio.run(trade_session())
```

### Step 3: Analyze Performance
```python
# Get performance report
report = integrated_system.get_performance_report()
print(f"\nPerformance Report:")
print(f"Total trades: {report['total_trades']}")
print(f"Execution rate: {report['execution_rate']:.2%}")
print(f"Average confidence: {report['average_confidence']:.3f}")
print(f"Method distribution: {report['method_distribution']}")
```

## Understanding Output

### Scanner Results
```python
# DataFrame columns explained
print(scanner.scan_results.columns.tolist())

# Key columns:
# - symbol: Trading pair (e.g., 'BTC/USDT')
# - signal: Traditional signal ('Strong Buy', 'Buy', etc.)
# - state: Market state ('BULL_STRONG', 'NEUTRAL', etc.)
# - composite_score: Overall quality score (0-100)
# - confidence_score: Confidence in signal (0.0-1.0)
# - momentum_short: Short-term momentum
# - momentum_long: Long-term momentum
# - rsi: Relative Strength Index
# - macd: MACD indicator
# - reversion_probability: Mean reversion likelihood
# - cluster_validated: Momentum confirmed by patterns
```

### Trading Signals
```python
# Integrated system signals
signals_df = await integrated_system.generate_signals()

# Additional columns:
# - rl_position: RL agent position (-1.0 to 1.0)
# - final_position: Meta-controller final decision
# - method: Decision method used
# - rule_position: Traditional signal position
# - market_regime: Detected regime
```

## Common Use Cases

### Case 1: Daily Momentum Trading
```python
# Scan for strong momentum
scanner = MomentumScanner(exchange, market_type='crypto')
results = await scanner.scan_market(timeframe='daily')

# Filter momentum signals
momentum = results[
    (results['signal'].isin(['Strong Buy', 'Buy'])) &
    (results['momentum_strength'] == 'strong') &
    (results['cluster_validated'] == True)
]

print(f"Found {len(momentum)} validated momentum opportunities")
```

### Case 2: Mean Reversion Strategy
```python
# Find extreme RSI with reversal probability
reversion = scanner.get_reversion_candidates()

# High-confidence reversions
high_prob = reversion[reversion['reversion_probability'] > 0.75]

print(f"High-probability reversions: {len(high_prob)}")
print(high_prob[['symbol', 'rsi', 'reversion_probability', 'signal']])
```

### Case 3: Multi-Timeframe Confirmation
```python
# Scan multiple timeframes
mtf_results = scanner.scan_multi_timeframe(
    timeframes=['1h', '4h', '1d'],
    save_results=True
)

# Strong confluence signals
strong_confluence = mtf_results[
    (mtf_results['strong_confluence'] == True) &
    (mtf_results['composite_score'] > 75)
]

print(f"Strong multi-timeframe signals: {len(strong_confluence)}")
```

### Case 4: Ensemble Strategy
```python
from additional_strategies import run_all_strategies_and_ensemble

# Run 7 additional strategies + ensemble
consensus = run_all_strategies_and_ensemble(
    scanner.scan_results,
    weights={
        'MEAN_REVERSION': 1.5,
        'MOMENTUM_BREAKOUT': 1.2,
        'SENTIMENT_MOMENTUM': 1.0
    }
)

# Filter consensus trades
strong_consensus = {
    sym: data for sym, data in consensus.items()
    if abs(data['consensus']) > 0.7
}

print(f"Ensemble consensus: {len(strong_consensus)} symbols")
```

## Configuration Tips

### For Conservative Trading
```python
from scanner import TradingConfig

conservative_config = TradingConfig(
    signal_thresholds={
        'crypto': {
            'daily': {
                'momentum_short': 0.08,  # Higher threshold
                'rsi_min': 40,
                'rsi_max': 60
            }
        }
    },
    lookback_window=100,  # More historical data
    rsi_threshold=25.0    # Stricter oversold
)

scanner = MomentumScanner(exchange, config=conservative_config)
```

### For Aggressive Trading
```python
aggressive_config = TradingConfig(
    signal_thresholds={
        'crypto': {
            'daily': {
                'momentum_short': 0.02,  # Lower threshold
                'rsi_min': 25,
                'rsi_max': 75
            }
        }
    },
    lookback_window=30,   # Less historical data
    volume_multiplier=2.0  # Lower volume requirement
)
```

### For Forex Markets
```python
forex_scanner = MomentumScanner(
    exchange=ccxt.oanda(),  # Or use yfinance adapter
    market_type='forex',
    quote_currency='USD',
    min_volume_usd=1_000_000
)

# Forex config automatically loaded
results = await forex_scanner.scan_market(timeframe='daily')
```

## Next Steps

1. **Read Full Documentation**: See `docs/SIGNAL_SYSTEM.md` for signal details
2. **Explore Strategies**: Check `docs/STRATEGIES.md` for all available strategies
3. **Customize Config**: Adjust `TradingConfig` for your risk profile
4. **Backtest**: Use `vectorized_backtest.py` for historical validation
5. **Deploy**: Set up continuous scanning with cron/scheduler

## Troubleshooting

### No Signals Generated
```python
# Check minimum volume filter
scanner.min_volume_usd = 100_000  # Lower threshold

# Check scan results
print(f"Scanned {len(scanner.scan_results)} symbols")
print(scanner.scan_results['average_volume_usd'].describe())
```

### RL Agent Not Working
```python
# Verify training
print(f"RL Agent trained: {pipeline.rl_agent.is_trained}")

# Check model path
print(f"Model saved: models/rl_ppo_model.zip exists")
```

### Exchange Connection Issues
```python
# Test exchange connectivity
try:
    markets = await exchange.load_markets()
    print(f"Connected: {len(markets)} markets available")
except Exception as e:
    print(f"Connection error: {e}")
```

## Support & Resources

- **Full API Docs**: See `docs/API_REFERENCE.md`
- **Architecture**: See `docs/ARCHITECTURE.md`
- **Signal System**: See `docs/SIGNAL_SYSTEM.md`
- **Strategies**: See `docs/STRATEGIES.md`
- **Code Examples**: See `examples/` directory

## Eventlet + Flask / SocketIO (important)

If you're running the dashboard or any Flask + SocketIO server using `eventlet`, the monkey-patching must occur before importing Flask, SQLAlchemy, or any other module that touches sockets or threads. Put the following at the very top of your main entrypoint (for example `dashboard_server.py`) as the first lines in the file:

```python
import eventlet
eventlet.monkey_patch()

# then import the rest of your app
from flask import Flask
from flask_socketio import SocketIO
```

Failing to do this can cause runtime errors such as `RuntimeError: Working outside of application context` and repeated monkey-patch exceptions as the server restarts.
