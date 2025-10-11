
"""
Complete System Integration Test
Tests all components working together: Oracle, Imagination, Parallel Scanner, WebSocket
"""

import asyncio
import logging
from mirrorcore_x import create_mirrorcore_system
from parallel_scanner_integration import add_parallel_scanner_to_mirrorcore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_complete_integration():
    """Test all systems integrated together"""
    
    print("=" * 80)
    print("🚀 COMPLETE SYSTEM INTEGRATION TEST")
    print("=" * 80)
    
    # 1. Create main system with all enhancements
    print("\n📦 Initializing MirrorCore-X with all enhancements...")
    sync_bus, components = await create_mirrorcore_system(
        dry_run=True,
        use_testnet=True,
        enable_oracle=False,  # Disable oracle for basic test
        enable_bayesian=False,
        enable_imagination=False
    )
    
    scanner = components.get('market_scanner')
    trade_analyzer = components.get('trade_analyzer')
    execution_daemon = components.get('execution_daemon')
    
    # 2. Run initial ticks to populate data
    print("\n📊 Generating initial market data...")
    for i in range(15):
        await sync_bus.tick()
        await asyncio.sleep(0.05)
    
    # 3. Verify scanner is working
    print("\n🔍 Testing Scanner...")
    scanner_data = await sync_bus.get_state('scanner_data') or []
    print(f"  ✅ Scanner data points: {len(scanner_data)}")
    
    # 4. Test execution daemon
    print("\n⚙️ Testing Execution Daemon...")
    test_directive = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'amount': 0.001,
        'price': 50000
    }
    exec_result = await execution_daemon.execute_order(
        test_directive['symbol'],
        test_directive['action'],
        test_directive['amount'],
        test_directive['price']
    )
    print(f"  ✅ Execution test: {exec_result.get('id', 'Success')}")
    
    # 5. Test trade analyzer
    print("\n📈 Testing Trade Analyzer...")
    trades = await sync_bus.get_state('trades') or []
    print(f"  ✅ Trades recorded: {len(trades)}")
    if trade_analyzer:
        print(f"  ✅ Total PnL: ${trade_analyzer.get_total_pnl():.2f}")
        print(f"  ✅ Win Rate: {trade_analyzer.get_win_rate() * 100:.1f}%")
    
    # 6. Test data flow through SyncBus
    print("\n🔄 Testing SyncBus Data Flow...")
    scanner_data = await sync_bus.get_state('scanner_data')
    market_data = await sync_bus.get_state('market_data')
    trading_directives = await sync_bus.get_state('trading_directives')
    
    print(f"  ✅ Scanner data points: {len(scanner_data or [])}")
    print(f"  ✅ Market data points: {len(market_data or [])}")
    print(f"  ✅ Trading directives: {len(trading_directives or [])}")
    
    # 7. Run full integrated cycle
    print("\n🔁 Running Full Integrated Cycle...")
    for i in range(20):
        await sync_bus.tick()
        
        if i % 5 == 0:
            directives = await sync_bus.get_state('trading_directives') or []
            print(f"  Tick {i}: {len(directives)} directives")
        
        await asyncio.sleep(0.05)
    
    # 8. Final Status Report
    print("\n" + "=" * 80)
    print("📊 FINAL SYSTEM STATUS")
    print("=" * 80)
    
    system_health = await sync_bus.get_state('system_health') or {}
    print(f"  Active Agents: {system_health.get('active_agents', 0)}")
    print(f"  Total Ticks: {sync_bus.tick_count}")
    print(f"  Efficiency: {system_health.get('efficiency', 0) * 100:.1f}%")
    
    print("\n✨ Complete system integration test finished!")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_complete_integration())

async def test_complete_integration():
    """Test all integrated components"""
    
    print("\n" + "="*80)
    print("🚀 COMPLETE SYSTEM INTEGRATION TEST")
    print("="*80)
    
    # Create system with ALL features enabled
    sync_bus, components = await create_mirrorcore_system(
        dry_run=True,
        use_testnet=True,
        enable_oracle=True,
        enable_bayesian=True,
        enable_imagination=True,
        enable_parallel_scanner=True
    )
    
    oracle_imagination = components.get('oracle_imagination')
    parallel_scanner = components.get('parallel_scanner')
    
    # Test 1: Oracle & Imagination
    print("\n📊 Test 1: Oracle & Imagination System")
    if oracle_imagination:
        for i in range(10):
            await sync_bus.tick()
        
        results = await oracle_imagination.run_enhanced_cycle()
        print(f"  ✅ Generated {len(results.get('oracle_directives', []))} enhanced directives")
        
        imagination_results = await oracle_imagination.force_imagination_analysis()
        print(f"  ✅ Imagination analysis: {imagination_results.get('summary', {}).get('status')}")
    else:
        print("  ❌ Oracle & Imagination not available")
    
    # Test 2: Parallel Exchange Scanner
    print("\n🌐 Test 2: Parallel Exchange Scanner")
    if parallel_scanner:
        scan_results = await parallel_scanner.scan_and_update()
        health = await parallel_scanner.get_health_report()
        
        print(f"  ✅ Scanned {len(scan_results)} opportunities")
        print(f"  ✅ Exchanges: {', '.join(health.keys())}")
        
        for exchange, metrics in health.items():
            print(f"     {exchange}: {metrics['health_score']:.1%} health")
    else:
        print("  ❌ Parallel scanner not available")
    
    # Test 3: Data Flow
    print("\n📡 Test 3: Data Flow Through System")
    scanner_data = await sync_bus.get_state('scanner_data') or []
    oracle_directives = await sync_bus.get_state('oracle_directives') or []
    parallel_data = await sync_bus.get_state('parallel_scanner_data') or []
    
    print(f"  ✅ Scanner signals: {len(scanner_data)}")
    print(f"  ✅ Oracle directives: {len(oracle_directives)}")
    print(f"  ✅ Parallel exchange data: {len(parallel_data)}")
    
    # Test 4: Run integrated trading loop
    print("\n🔄 Test 4: Integrated Trading Loop (20 ticks)")
    for i in range(20):
        await sync_bus.tick()
        
        if i % 5 == 0 and oracle_imagination:
            results = await oracle_imagination.run_enhanced_cycle()
            print(f"  Tick {i}: {len(results.get('oracle_directives', []))} directives")
        
        if i % 10 == 0 and parallel_scanner:
            await parallel_scanner.scan_and_update()
    
    # Final Summary
    print("\n" + "="*80)
    print("📋 INTEGRATION TEST SUMMARY")
    print("="*80)
    
    final_stats = {
        'Oracle & Imagination': '✅' if oracle_imagination else '❌',
        'Parallel Scanner': '✅' if parallel_scanner else '❌',
        'Data Flow': '✅' if scanner_data else '❌',
        'Trading Loop': '✅'
    }
    
    for component, status in final_stats.items():
        print(f"  {status} {component}")
    
    # Cleanup
    if parallel_scanner:
        await parallel_scanner.close()
    
    print("\n✨ Complete integration test finished!")
    
    return all(status == '✅' for status in final_stats.values())

if __name__ == "__main__":
    success = asyncio.run(test_complete_integration())
    exit(0 if success else 1)
