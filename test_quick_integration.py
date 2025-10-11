
import asyncio
import logging
from api import system_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_test():
    """Quick integration test"""
    print("\n🧪 Quick Integration Test")
    print("=" * 50)
    
    # Wait for system startup
    await asyncio.sleep(3)
    
    sync_bus = system_state.get('sync_bus')
    components = system_state.get('components')
    
    if not sync_bus or not components:
        print("❌ System not initialized")
        return False
    
    # Test 1: Scanner data
    scanner_data = await sync_bus.get_state('scanner_data') or []
    print(f"✅ Scanner: {len(scanner_data)} data points")
    
    # Test 2: Market data
    market_data = await sync_bus.get_state('market_data') or []
    print(f"✅ Market: {len(market_data)} data points")
    
    # Test 3: Components
    trade_analyzer = components.get('trade_analyzer')
    print(f"✅ Trade Analyzer: {'Available' if trade_analyzer else 'Missing'}")
    
    # Test 4: WebSocket clients
    from api import active_websockets
    print(f"✅ WebSocket Clients: {len(active_websockets)}")
    
    # Test 5: Parallel scanner
    parallel_scanner = system_state.get('parallel_scanner')
    if parallel_scanner:
        health = await parallel_scanner.get_health_report()
        print(f"✅ Parallel Scanner: {len(health)} exchanges")
    else:
        print("⚠️ Parallel Scanner: Not enabled")
    
    print("\n✨ Quick test complete!")
    return True

if __name__ == "__main__":
    asyncio.run(quick_test())
