
"""
Test Advanced Strategies Integration
Validates Bayesian, Liquidity Flow, Entropy analyzers and ensemble optimization
"""

import asyncio
import logging
from mirrorcore_x import create_mirrorcore_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_advanced_integration():
    """Test complete advanced strategies integration"""
    
    print("=" * 80)
    print("🚀 ADVANCED STRATEGIES INTEGRATION TEST")
    print("=" * 80)
    
    # Create system with advanced strategies enabled
    sync_bus, components = await create_mirrorcore_system(
        dry_run=True,
        use_testnet=True,
        enable_oracle=False,
        enable_bayesian=False,
        enable_imagination=False,
        enable_advanced_strategies=True,
        risk_profile='moderate'
    )
    
    ensemble_manager = components.get('ensemble_manager')
    strategy_trainer = components.get('strategy_trainer')
    
    # Test 1: Verify strategies registered
    print("\n📊 Test 1: Strategy Registration")
    strategies = list(strategy_trainer.strategies.keys())
    print(f"  ✅ Registered strategies: {len(strategies)}")
    for strategy in strategies:
        print(f"     - {strategy}")
    
    # Test 2: Run system ticks
    print("\n🔄 Test 2: System Execution (20 ticks)")
    for i in range(20):
        await sync_bus.tick()
        
        if i % 5 == 0:
            scanner_data = await sync_bus.get_state('scanner_data') or []
            print(f"  Tick {i}: {len(scanner_data)} signals")
    
    # Test 3: Check ensemble manager
    print("\n🎯 Test 3: Ensemble Manager Status")
    if ensemble_manager:
        status = ensemble_manager.get_status()
        print(f"  ✅ Optimizer available: {status['optimizer_available']}")
        print(f"  ✅ Current regime: {status['current_regime']}")
        print(f"  ✅ Risk profile: {status['risk_profile']}")
        print(f"  ✅ Active strategies: {status['active_strategies']}")
        
        # Display weights
        if status['weights']:
            print("\n  Strategy Weights:")
            for name, weight in sorted(status['weights'].items(), key=lambda x: x[1], reverse=True):
                print(f"     {name}: {weight:.3f}")
    else:
        print("  ❌ Ensemble manager not available")
    
    # Test 4: Performance metrics
    print("\n📈 Test 4: Performance Tracking")
    grades = strategy_trainer.grade_strategies()
    print("  Strategy Grades:")
    for name, grade in grades.items():
        print(f"     {name}: {grade}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    test_results = {
        'Strategy Registration': '✅' if len(strategies) >= 10 else '❌',
        'System Execution': '✅',
        'Ensemble Manager': '✅' if ensemble_manager else '❌',
        'Performance Tracking': '✅' if grades else '❌'
    }
    
    for test, result in test_results.items():
        print(f"  {result} {test}")
    
    print("\n✨ Advanced integration test complete!")
    
    return all(r == '✅' for r in test_results.values())


if __name__ == "__main__":
    success = asyncio.run(test_advanced_integration())
    exit(0 if success else 1)
