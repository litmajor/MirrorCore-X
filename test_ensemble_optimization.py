
"""
Test Mathematical Ensemble Optimization Integration
"""

import asyncio
import pandas as pd
import numpy as np
from strategy_trainer_agent import StrategyTrainerAgent
from ensemble_integration import EnhancedEnsembleManager
from mirrorcore_x import HighPerformanceSyncBus
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mathematical_optimization():
    print("=" * 80)
    print("🔬 MATHEMATICAL ENSEMBLE OPTIMIZATION TEST")
    print("=" * 80)
    
    # Create SyncBus
    sync_bus = HighPerformanceSyncBus()
    
    # Create strategy trainer
    trainer = StrategyTrainerAgent(
        min_weight=0.05,
        max_weight=0.30,
        lookback_window=20
    )
    
    # Register some test strategies
    from strategy_trainer_agent import UTSignalAgent, GradientTrendAgent, SupportResistanceAgent
    trainer.register_strategy("UT_BOT", UTSignalAgent())
    trainer.register_strategy("GRADIENT_TREND", GradientTrendAgent())
    trainer.register_strategy("VBSR", SupportResistanceAgent())
    
    # Add advanced strategies
    try:
        from additional_strategies import (
            MeanReversionAgent, MomentumBreakoutAgent, 
            VolatilityRegimeAgent, AnomalyDetectionAgent
        )
        trainer.register_strategy("MEAN_REVERSION", MeanReversionAgent())
        trainer.register_strategy("MOMENTUM_BREAKOUT", MomentumBreakoutAgent())
        trainer.register_strategy("VOLATILITY_REGIME", VolatilityRegimeAgent())
        trainer.register_strategy("ANOMALY_DETECTION", AnomalyDetectionAgent())
        print(f"✅ Registered {len(trainer.strategies)} strategies")
    except ImportError:
        print("⚠️  Advanced strategies not available, using core strategies only")
    
    # Simulate performance history
    print("\n📊 Simulating strategy performance...")
    np.random.seed(42)
    for strategy_name in trainer.strategies.keys():
        # Generate realistic PnL history
        base_return = np.random.uniform(-0.02, 0.05)
        volatility = np.random.uniform(0.01, 0.03)
        
        for i in range(50):
            pnl = np.random.normal(base_return, volatility) * 100
            trainer.update_performance(strategy_name, pnl)
    
    print(f"   Generated 50 trades per strategy")
    
    # Create mock market data
    market_data = pd.DataFrame({
        'price': np.cumsum(np.random.randn(100) * 0.01) + 100,
        'volume': np.random.uniform(1000, 5000, 100),
        'volatility': np.random.uniform(0.01, 0.05, 100),
        'trend_score': np.random.uniform(-10, 10, 100)
    })
    
    # Test 1: Basic optimization
    print("\n🧮 Test 1: Mathematical Optimization")
    optimized_weights = trainer.optimize_ensemble_weights(market_data)
    
    print("\n  Optimized Weights:")
    for name, weight in sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:20s}: {weight*100:6.2f}%")
    
    # Test 2: Optimization report
    print("\n📈 Test 2: Optimization Report")
    report = trainer.get_optimization_report()
    
    print(f"  Regime: {report.get('current_regime', 'N/A')}")
    print(f"  Optimization Count: {report.get('optimization_count', 0)}")
    print(f"  Average Sharpe: {report.get('average_sharpe', 0):.3f}")
    
    if 'latest_optimization' in report:
        latest = report['latest_optimization']
        print(f"\n  Latest Optimization:")
        print(f"    Expected Return: {latest['expected_return']*100:.3f}%")
        print(f"    Expected Volatility: {latest['expected_volatility']*100:.3f}%")
        print(f"    Sharpe Ratio: {latest['sharpe_ratio']:.3f}")
    
    # Test 3: Ensemble integration
    print("\n🎯 Test 3: Enhanced Ensemble Manager Integration")
    ensemble = EnhancedEnsembleManager(trainer, sync_bus, risk_profile='moderate')
    
    # Prepare scanner data
    scanner_data = []
    for _, row in market_data.tail(20).iterrows():
        scanner_data.append({
            'symbol': 'BTC/USDT',
            'price': row['price'],
            'volume': row['volume'],
            'volatility': row['volatility'],
            'trend_score': row['trend_score']
        })
    
    result = await ensemble.update({'scanner_data': scanner_data})
    
    print(f"\n  Ensemble Regime: {result.get('regime', 'N/A')}")
    print(f"  Active Strategies: {len(result.get('weights', {}))}")
    
    if 'optimization_report' in result:
        opt = result['optimization_report']
        print(f"  Mathematical Optimization: {opt.get('enabled', False)}")
    
    # Test 4: Different market regimes
    print("\n🌊 Test 4: Regime-Aware Optimization")
    
    regimes_data = {
        'trending': pd.DataFrame({
            'price': np.cumsum(np.random.randn(50) * 0.02 + 0.01) + 100,
            'volume': np.random.uniform(2000, 6000, 50),
            'volatility': np.random.uniform(0.01, 0.03, 50),
            'trend_score': np.random.uniform(5, 10, 50)
        }),
        'volatile': pd.DataFrame({
            'price': np.cumsum(np.random.randn(50) * 0.05) + 100,
            'volume': np.random.uniform(3000, 8000, 50),
            'volatility': np.random.uniform(0.05, 0.10, 50),
            'trend_score': np.random.uniform(-5, 5, 50)
        }),
        'ranging': pd.DataFrame({
            'price': 100 + np.sin(np.linspace(0, 4*np.pi, 50)) * 2,
            'volume': np.random.uniform(1000, 3000, 50),
            'volatility': np.random.uniform(0.005, 0.015, 50),
            'trend_score': np.random.uniform(-2, 2, 50)
        })
    }
    
    for regime_name, regime_data in regimes_data.items():
        weights = trainer.optimize_ensemble_weights(regime_data)
        report = trainer.get_optimization_report()
        
        print(f"\n  {regime_name.upper()} Regime:")
        print(f"    Detected: {report.get('current_regime', 'N/A')}")
        print(f"    Top 3 Strategies:")
        
        top_3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, weight in top_3:
            print(f"      {name:20s}: {weight*100:6.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ Mathematical ensemble optimization test completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_mathematical_optimization())
