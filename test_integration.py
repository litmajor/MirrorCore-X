import asyncio
import logging
from mirrorcore_x import create_mirrorcore_system

logging.basicConfig(level=logging.INFO)

async def test_integration():
    try:
        sync_bus, market_generator = create_mirrorcore_system(None)

        # Check if all strategies are registered (if applicable)
        strategy_trainer = getattr(sync_bus, 'agents', {}).get('strategy_trainer', None)
        if strategy_trainer:
            print(f"Registered strategies: {list(strategy_trainer.strategies.keys())}")
            print(f"Strategy weights: {strategy_trainer.live_weights}")

        # Run a few ticks to test (simulate market data)
        for i in range(5):
            # Generate a market tick and pass to sync_bus
            market_data = market_generator.generate_tick()
            sync_bus.tick(market_data)
            print(f"Tick {i+1} completed")

        # Check strategy grades (if applicable)
        if strategy_trainer:
            grades = strategy_trainer.grade_strategies()
            print("Strategy grades:", grades)

        print("✅ Integration test successful!")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # No async close needed for market_generator
        pass

if __name__ == "__main__":
    asyncio.run(test_integration())
