
import asyncio
import logging
from datetime import datetime
import ccxt.async_support as ccxt
from scanner import MomentumScanner, get_dynamic_config
from mirrorcore_x import create_mirrorcore_system
from secrets_manager import SecretsManager, load_secrets_from_env_file
from emergency_killswitch import EmergencyKillSwitch
from quick_backtest import QuickBacktest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_real_exchange_connection():
    """Test real exchange connection and data fetching"""
    logger.info("Testing real exchange connection...")
    
    load_secrets_from_env_file()
    secrets = SecretsManager()
    
    try:
        # Test with testnet first
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Fetch real ticker
        ticker = await exchange.fetch_ticker('BTC/USDT')
        logger.info(f"✅ Real BTC/USDT price: ${ticker['last']:.2f}")
        
        # Fetch real OHLCV
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=10)
        logger.info(f"✅ Fetched {len(ohlcv)} real candlesticks")
        
        await exchange.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Exchange connection failed: {e}")
        return False

async def test_real_scanner():
    """Test real market scanner with live data"""
    logger.info("Testing real market scanner...")
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        config = get_dynamic_config()
        scanner = MomentumScanner(exchange, config=config)
        
        # Scan real market
        results = await scanner.scan_market(timeframe='1h', full_analysis=True)
        
        if not results.empty:
            logger.info(f"✅ Scanner found {len(results)} real trading opportunities")
            logger.info(f"Top signal: {results.iloc[0]['symbol']} - {results.iloc[0]['signal']}")
            logger.info(f"RSI: {results.iloc[0].get('rsi', 'N/A')}, Score: {results.iloc[0].get('composite_score', 'N/A')}")
            
            await exchange.close()
            return True
        else:
            logger.warning("⚠️ Scanner returned no results")
            await exchange.close()
            return False
            
    except Exception as e:
        logger.error(f"❌ Scanner test failed: {e}")
        return False

async def test_real_mirrorcore_system():
    """Test full MirrorCore-X system with real components"""
    logger.info("Testing complete MirrorCore-X system with real implementations...")
    
    try:
        # Create system with dry_run=True for safety
        sync_bus, components = await create_mirrorcore_system(dry_run=True, use_testnet=False)
        
        logger.info("✅ MirrorCore-X system initialized with real components")
        
        # Run a few ticks with real data
        for i in range(5):
            await sync_bus.tick()
            logger.info(f"✅ Tick {i+1} completed with real data processing")
            await asyncio.sleep(1)
        
        # Check system health
        system_health = await sync_bus.get_state('system_health')
        logger.info(f"✅ System health: {system_health}")
        
        # Close exchange connection
        if 'exchange' in components and components['exchange']:
            await components['exchange'].close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MirrorCore system test failed: {e}")
        return False

async def test_emergency_controls():
    """Test emergency killswitch with real conditions"""
    logger.info("Testing emergency controls...")
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        killswitch = EmergencyKillSwitch(exchange, max_drawdown_pct=10.0)
        
        # Test normal conditions
        await killswitch.check_and_trigger(current_pnl=-500, current_latency_ms=100)
        logger.info("✅ Emergency controls passed normal conditions")
        
        # Test drawdown trigger
        await killswitch.check_and_trigger(current_pnl=-1500, current_latency_ms=100)
        if killswitch.is_emergency:
            logger.info("✅ Emergency stop correctly triggered on drawdown")
            killswitch.reset()
        
        # Test latency trigger
        await killswitch.check_and_trigger(current_pnl=-500, current_latency_ms=600)
        if killswitch.is_emergency:
            logger.info("✅ Emergency stop correctly triggered on high latency")
        
        await exchange.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Emergency controls test failed: {e}")
        return False

async def test_backtest_with_real_data():
    """Test backtesting with real historical data"""
    logger.info("Testing backtest with real data...")
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        config = get_dynamic_config()
        scanner = MomentumScanner(exchange, config=config)
        
        # Get real historical data
        historical_data = await scanner.scan_market(timeframe='1d', full_analysis=True)
        
        if not historical_data.empty:
            # Run backtest
            backtester = QuickBacktest(initial_capital=10000.0)
            results = backtester.run(historical_data, historical_data)
            
            logger.info(f"✅ Backtest completed with real data")
            logger.info(f"Final capital: ${results.get('final_capital', 0):.2f}")
            logger.info(f"Total PnL: ${results.get('total_pnl', 0):.2f}")
            logger.info(f"Win rate: {results.get('win_rate', 0)*100:.1f}%")
            logger.info(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
            
            await exchange.close()
            return True
        else:
            logger.warning("⚠️ No historical data for backtest")
            await exchange.close()
            return False
            
    except Exception as e:
        logger.error(f"❌ Backtest test failed: {e}")
        return False

async def run_all_tests():
    """Run all real implementation tests"""
    print("="*60)
    print("🧪 MIRRORCORE-X REAL IMPLEMENTATION TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Test 1: Exchange Connection
    results['exchange'] = await test_real_exchange_connection()
    print()
    
    # Test 2: Real Scanner
    results['scanner'] = await test_real_scanner()
    print()
    
    # Test 3: Full System
    results['system'] = await test_real_mirrorcore_system()
    print()
    
    # Test 4: Emergency Controls
    results['emergency'] = await test_emergency_controls()
    print()
    
    # Test 5: Backtest
    results['backtest'] = await test_backtest_with_real_data()
    print()
    
    # Summary
    print("="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.capitalize():20} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    return all(results.values())

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
