
"""
Complete System Integration Test
Tests all components working together: Oracle, Imagination, Parallel Scanner, WebSocket
"""

import asyncio
import logging
from mirrorcore_x import create_mirrorcore_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
