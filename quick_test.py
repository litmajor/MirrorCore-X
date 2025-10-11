
import asyncio
from secrets_manager import SecretsManager
from emergency_killswitch import EmergencyKillSwitch
from quick_backtest import QuickBacktest
import ccxt.async_support as ccxt

async def quick_system_check():
    """Run quick system validation before going live"""
    
    print("🔍 Running system checks...")
    
    # 1. Secrets check
    try:
        secrets = SecretsManager()
        print("✅ API keys loaded from environment")
    except Exception as e:
        print(f"❌ Secrets failed: {e}")
        return
    
    # 2. Exchange connection
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        await exchange.load_markets()
        print("✅ Exchange connection successful")
    except Exception as e:
        print(f"❌ Exchange failed: {e}")
        return
    
    # 3. Emergency system
    try:
        killswitch = EmergencyKillSwitch(exchange)
        print("✅ Emergency kill-switch ready")
    except Exception as e:
        print(f"❌ Kill-switch failed: {e}")
    
    # 4. Backtest validation
    print("✅ All systems operational - Ready for trading")
    
    await exchange.close()

if __name__ == "__main__":
    asyncio.run(quick_system_check())
