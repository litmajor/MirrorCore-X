
"""
Test Oracle & Imagination Integration

Demonstrates the full capabilities of the integrated system
"""

import asyncio
import logging
from mirrorcore_x import create_mirrorcore_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_oracle_imagination():
    """Test the Oracle and Imagination integration"""
    
    print("=" * 80)
    print("🌌 TESTING ORACLE & IMAGINATION INTEGRATION")
    print("=" * 80)
    
    # Create system with all enhancements
    sync_bus, components = await create_mirrorcore_system(
        dry_run=True,
        use_testnet=True,
        enable_oracle=True,
        enable_bayesian=True,
        enable_imagination=True
    )
    
    oracle_imagination = components.get('oracle_imagination')
    
    if not oracle_imagination:
        print("❌ Oracle & Imagination integration not available")
        return
    
    print("\n✅ System initialized with all enhancements")
    print(f"Status: {oracle_imagination.get_status()}")
    
    # Run a few ticks to generate data
    print("\n📊 Running initial ticks to generate market data...")
    for i in range(10):
        await sync_bus.tick()
        await asyncio.sleep(0.1)
    
    # Run enhanced cycle
    print("\n🎯 Running enhanced trading cycle...")
    results = await oracle_imagination.run_enhanced_cycle()
    
    print("\n📈 Enhanced Cycle Results:")
    print(f"  Oracle Directives: {len(results.get('oracle_directives', []))}")
    
    # Display directives
    for directive in results.get('oracle_directives', [])[:5]:
        print(f"\n  📋 Directive for {directive['symbol']}:")
        print(f"     Action: {directive['action']}")
        print(f"     Amount: {directive['amount']:.4f}")
        print(f"     Confidence: {directive['confidence']:.2%}")
        print(f"     Method: {directive.get('method', 'N/A')}")
        if 'enhanced_momentum_score' in directive:
            print(f"     Enhanced Momentum: {directive['enhanced_momentum_score']:.4f}")
        if 'volatility_regime' in directive:
            print(f"     Volatility Regime: {directive['volatility_regime']}")
    
    # Check Bayesian recommendations
    if 'bayesian_recommendations' in results and results['bayesian_recommendations']:
        print("\n🧠 Bayesian Recommendations:")
        bayesian = results['bayesian_recommendations']
        if 'top_strategies' in bayesian:
            for strat in bayesian['top_strategies'][:3]:
                print(f"  • {strat['name']}: {strat['probability']:.2%} confidence")
                if 'explanation' in strat:
                    print(f"    {strat['explanation']}")
    
    # Check Imagination insights
    if 'imagination_insights' in results:
        imagination = results['imagination_insights']
        print(f"\n🌌 Imagination Analysis:")
        print(f"  Status: {imagination.get('status', 'N/A')}")
        
        if 'summary' in imagination:
            summary = imagination['summary']
            print(f"  Strategies Analyzed: {summary.get('strategies_analyzed', 0)}")
            print(f"  Average Robustness: {summary.get('average_robustness', 0):.3f}")
            
            if 'best_strategy' in summary and summary['best_strategy']:
                best = summary['best_strategy']
                print(f"  Best Strategy: {best.get('name', 'N/A')} (score: {best.get('score', 0):.3f})")
            
            if 'recommendations' in summary:
                print("\n  Recommendations:")
                for rec in summary['recommendations']:
                    print(f"    • {rec}")
    
    # Force imagination analysis
    print("\n🔄 Forcing immediate Imagination analysis...")
    imagination_results = await oracle_imagination.force_imagination_analysis()
    
    if imagination_results.get('status') == 'Analysis completed':
        print(f"✅ Analysis complete: {imagination_results.get('scenarios_tested', 0)} scenarios tested")
    
    # Export analysis
    export_path = await oracle_imagination.export_analysis('test_enhanced_analysis.json')
    print(f"\n💾 Analysis exported to: {export_path}")
    
    # Run a few more ticks with the enhanced system
    print("\n🔄 Running enhanced system for 20 ticks...")
    for i in range(20):
        await sync_bus.tick()
        
        # Run enhanced cycle every 5 ticks
        if i % 5 == 0:
            cycle_results = await oracle_imagination.run_enhanced_cycle()
            directive_count = len(cycle_results.get('oracle_directives', []))
            print(f"  Tick {i}: {directive_count} directives generated")
        
        await asyncio.sleep(0.05)
    
    # Final status
    print("\n" + "=" * 80)
    print("📊 FINAL STATUS")
    print("=" * 80)
    
    status = oracle_imagination.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n✨ Test complete!")

async def integrate_into_main_system():
    """Integrate Oracle & Imagination into main MirrorCore system"""
    from mirrorcore_x import create_mirrorcore_system
    
    print("\n" + "="*80)
    print("🔗 INTEGRATING ORACLE & IMAGINATION INTO MAIN SYSTEM")
    print("="*80)
    
    # Create main system with all enhancements
    sync_bus, components = await create_mirrorcore_system(
        dry_run=True,
        use_testnet=True,
        enable_oracle=True,
        enable_bayesian=True,
        enable_imagination=True
    )
    
    oracle_imagination = components.get('oracle_imagination')
    
    if oracle_imagination:
        print("✅ Oracle & Imagination successfully integrated into main system")
        
        # Run live system test
        print("\n🔄 Running live system integration test...")
        for i in range(30):
            await sync_bus.tick()
            
            if i % 10 == 0:
                results = await oracle_imagination.run_enhanced_cycle()
                print(f"  Tick {i}: {len(results.get('oracle_directives', []))} directives generated")
        
        # Export final analysis
        export_path = await oracle_imagination.export_analysis('integrated_system_analysis.json')
        print(f"\n✅ Integration complete! Analysis saved to: {export_path}")
        
        return True
    else:
        print("❌ Integration failed - Oracle & Imagination not available")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--integrate':
        asyncio.run(integrate_into_main_system())
    else:
        asyncio.run(test_oracle_imagination())
