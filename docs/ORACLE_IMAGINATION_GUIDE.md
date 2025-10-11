
# Oracle & Imagination Integration Guide

## Overview

The Oracle & Imagination integration adds advanced decision-making and strategy testing capabilities to MirrorCore-X:

1. **Trading Oracle Engine**: Enhanced directive generation with RL integration
2. **Bayesian Oracle**: Probabilistic belief system for strategy ranking
3. **Imagination Engine**: Counterfactual scenario testing and robustness analysis

## Architecture

```
MirrorCore-X System
├── Trading Oracle Engine
│   ├── Traditional Strategy Signals
│   ├── RL Predictions (optional)
│   ├── Meta Controller (blending)
│   └── Enhanced Features (momentum, reversion, etc.)
│
├── Bayesian Oracle
│   ├── Belief Tracking per Strategy
│   ├── Regime-Aware Probabilities
│   └── Confidence Intervals
│
└── Imagination Engine
    ├── Scenario Generator (8 scenario types)
    ├── Counterfactual Simulator
    ├── Robustness Scorer
    └── Meta Learner (parameter optimization)
```

## Quick Start

### Basic Integration

```python
from mirrorcore_x import create_mirrorcore_system

# Create system with all enhancements
sync_bus, components = await create_mirrorcore_system(
    dry_run=True,
    enable_oracle=True,
    enable_bayesian=True,
    enable_imagination=True
)

oracle_imagination = components['oracle_imagination']
```

### Running Enhanced Cycles

```python
# Run enhanced cycle
results = await oracle_imagination.run_enhanced_cycle()

# Get directives
directives = results['oracle_directives']
for directive in directives:
    print(f"{directive['symbol']}: {directive['action']} "
          f"@ {directive['confidence']:.2%} confidence")
```

### Accessing Bayesian Recommendations

```python
if 'bayesian_recommendations' in results:
    bayesian = results['bayesian_recommendations']
    
    # Top strategies
    for strategy in bayesian['top_strategies']:
        print(f"{strategy['name']}: {strategy['probability']:.2%}")
        print(f"  {strategy['explanation']}")
```

### Forcing Imagination Analysis

```python
# Force immediate scenario analysis
imagination_results = await oracle_imagination.force_imagination_analysis()

# Check robustness scores
summary = imagination_results['summary']
print(f"Best strategy: {summary['best_strategy']['name']}")
print(f"Robustness score: {summary['best_strategy']['score']:.3f}")
```

## Features

### Trading Oracle Engine

**Enhanced Directive Generation:**
- Integrates multiple signal sources
- RL prediction blending via Meta Controller
- Advanced scanner features (momentum, reversion, clustering)
- Adaptive confidence thresholds
- Volatility regime awareness

**Directive Format:**
```python
{
    'symbol': 'BTC/USDT',
    'action': 'buy',  # or 'sell', 'hold'
    'amount': 0.05,
    'price': 50000.0,
    'confidence': 0.85,
    'strategy': 'enhanced_integrated',
    'method': 'meta_blend',  # or 'rule_based', 'rl_weighted'
    
    # Enhanced metadata
    'enhanced_momentum_score': 0.042,
    'reversion_probability': 0.15,
    'cluster_validated': True,
    'volatility_regime': 'normal'
}
```

### Bayesian Oracle

**Probabilistic Strategy Ranking:**
- Regime-aware belief tracking
- Confidence intervals for predictions
- Uncertainty quantification
- Automated regime change adaptation

**Accessing Beliefs:**
```python
if oracle_imagination.bayesian_integration:
    beliefs = oracle_imagination.bayesian_integration.export_beliefs()
    print(beliefs)
```

### Imagination Engine

**Counterfactual Scenario Types:**
1. Breakout Continuation
2. False Breakout Reversal
3. Consolidation Range
4. Volatility Spike
5. Gradual Trend
6. Gap and Go
7. Whipsaw Chop
8. News Shock

**Robustness Metrics:**
- Average PnL across scenarios
- Survival rate (avoiding catastrophic loss)
- Stress resistance
- Maximum drawdown
- Scenario success rate

**Running Analysis:**
```python
# Automatic (every hour)
results = await oracle_imagination.run_enhanced_cycle()

# Manual/forced
imagination_results = await oracle_imagination.force_imagination_analysis()

# Access results
summary = imagination_results['summary']
vulnerabilities = imagination_results['vulnerabilities']
optimizations = imagination_results['optimizations']
```

## Configuration

### System Creation Options

```python
sync_bus, components = await create_mirrorcore_system(
    dry_run=True,              # Simulated trading
    use_testnet=True,          # Use testnet exchange
    enable_oracle=True,        # Enable Trading Oracle
    enable_bayesian=True,      # Enable Bayesian beliefs
    enable_imagination=True    # Enable scenario testing
)
```

### Oracle Configuration

```python
oracle_engine = oracle_imagination.oracle_engine

# Adjust parameters
oracle_engine.confidence_threshold = 0.75
oracle_engine.prediction_horizon = 10
oracle_engine.risk_weight = 0.4
oracle_engine.rl_weight = 0.4  # If RL available
```

### Imagination Configuration

```python
imagination_engine = oracle_imagination.imagination_engine

# Adjust scenario generation
imagination_engine.config['num_scenarios'] = 150
imagination_engine.config['scenario_length'] = 75
imagination_engine.config['robustness_threshold'] = 0.7
```

## Monitoring & Analysis

### Status Checking

```python
status = oracle_imagination.get_status()
print(f"Oracle Active: {status['oracle_active']}")
print(f"Bayesian Active: {status['bayesian_active']}")
print(f"Imagination Active: {status['imagination_active']}")
```

### Exporting Analysis

```python
# Export comprehensive analysis
filepath = await oracle_imagination.export_analysis('analysis.json')
print(f"Analysis saved to: {filepath}")
```

### Real-time Monitoring

```python
# In main loop
for i in range(100):
    await sync_bus.tick()
    
    # Enhanced cycle every 5 ticks
    if i % 5 == 0:
        results = await oracle_imagination.run_enhanced_cycle()
        
        # Log key metrics
        directives = results['oracle_directives']
        print(f"Tick {i}: {len(directives)} directives")
```

## Best Practices

1. **Initialization Order**: Always initialize the enhancement systems after the base system is created

2. **Data Requirements**: Ensure at least 100 data points before running Imagination analysis

3. **Performance**: Imagination analysis is CPU-intensive; run it periodically (default: hourly)

4. **Bayesian Updates**: Feed strategy performance data regularly for accurate belief updates

5. **Error Handling**: Wrap enhanced cycles in try-except blocks for production use

## Troubleshooting

### Oracle Not Generating Directives
- Check if scanner_data is available in SyncBus
- Verify confidence_threshold isn't too high
- Ensure strategies are registered in strategy_trainer

### Imagination Analysis Failing
- Verify sufficient market data (100+ points)
- Check strategy_trainer has registered strategies
- Reduce num_scenarios if performance is an issue

### Bayesian Integration Issues
- Ensure pandas is installed
- Check that market data has required columns
- Verify strategy names match across components

## Advanced Usage

### Custom Scenario Generation

```python
# Access scenario generator directly
scenario_gen = oracle_imagination.imagination_engine.scenario_generator

# Generate specific scenario type
custom_scenarios = scenario_gen.generate_scenarios(
    num_scenarios=20,
    scenario_length=100,
    oracle_bias=False  # No Oracle influence
)
```

### Parameter Optimization

```python
# Get optimization suggestions
imagination_results = await oracle_imagination.force_imagination_analysis()
optimizations = imagination_results.get('optimizations', [])

for opt in optimizations:
    if opt.confidence > 0.8:
        print(f"High-confidence optimization for {opt.strategy_name}:")
        print(f"  {opt.parameter_name}: {opt.current_value} → {opt.suggested_value}")
```

## API Reference

See [API_REFERENCE.md](API_REFERENCE.md) for detailed method signatures and parameters.
