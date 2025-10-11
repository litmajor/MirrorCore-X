
"""
Complete System Startup Script
Launches FastAPI server with integrated trading system
"""

import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting MirrorCore-X Complete Trading System")
    print("=" * 80)
    print("\n📡 API Server: http://0.0.0.0:8000")
    print("🔌 WebSocket: ws://0.0.0.0:8000/ws")
    print("📊 API Docs: http://0.0.0.0:8000/docs")
    print("\n" + "=" * 80 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
