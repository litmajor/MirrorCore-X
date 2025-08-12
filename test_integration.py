import asyncio
import logging
from mirrorcore_x import create_mirrorcore_system

logging.basicConfig(level=logging.INFO)

async def test_integration():
    try:
        sync_bus, trade_analyzer, scanner, exchange = await create_mirrorcore_system(dry_run=True)
        
        # Check if all strategies are registered
        strategy_trainer = sync_bus.agents.get('strategy_trainer')
        if strategy_trainer:
            print(f"Registered strategies: {list(strategy_trainer.strategies.keys())}")
            print(f"Strategy weights: {strategy_trainer.live_weights}")
        
        # Run a few ticks to test
        for i in range(5):
            await sync_bus.tick()
            print(f"Tick {i+1} completed")
        
        # Check strategy grades
        grades = strategy_trainer.grade_strategies()
        print("Strategy grades:", grades)
        
        print("✅ Integration test successful!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'exchange' in locals():
            await exchange.close()

if __name__ == "__main__":
    asyncio.run(test_integration())