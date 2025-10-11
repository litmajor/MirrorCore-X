
"""
Complete System Startup Script
Launches FastAPI server with integrated trading system
"""

import uvicorn
import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def startup_with_parallel_scanner():
    """Initialize system with parallel scanner before starting server"""
    from parallel_scanner_integration import add_parallel_scanner_to_mirrorcore
    from api import system_state
    
    # Wait for system to initialize
    await asyncio.sleep(2)
    
    sync_bus = system_state.get('sync_bus')
    components = system_state.get('components')
    
    if sync_bus and components:
        scanner = components.get('market_scanner')
        
        # Enable parallel scanner
        parallel_scanner = await add_parallel_scanner_to_mirrorcore(
            sync_bus, scanner, enable=True
        )
        
        if parallel_scanner:
            system_state['parallel_scanner'] = parallel_scanner
            logging.info("✅ Parallel Exchange Scanner enabled")
        else:
            logging.warning("⚠️ Parallel Scanner failed to initialize")

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting MirrorCore-X Complete Trading System")
    print("=" * 80)
    print("\n📡 API Server: http://0.0.0.0:8000")
    print("🔌 WebSocket: ws://0.0.0.0:8000/ws")
    print("📊 API Docs: http://0.0.0.0:8000/docs")
    print("🌐 Parallel Scanner: ENABLED")
    print("\n" + "=" * 80 + "\n")
    
    # Start parallel scanner initialization in background
    asyncio.create_task(startup_with_parallel_scanner())
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
