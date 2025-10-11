
import asyncio
import pandas as pd
from scanner import MomentumScanner
from rl_trading_system import RLTradingAgent, MetaController, IntegratedTradingSystem

async def get_final_trading_signals(timeframe='1h', top_n=10):
    """
    Get final actionable trading signals with full lifecycle
    
    Returns:
        DataFrame with final signals ready for execution
    """
    # Initialize scanner
    import ccxt.async_support as ccxt
    exchange = ccxt.binance({'enableRateLimit': True})
    
    scanner = MomentumScanner(
        exchange=exchange,
        market_type='crypto',
        quote_currency='USDT',
        min_volume_usd=500_000,
        top_n=50
    )
    
    try:
        # STEP 1: Get scanner signals
        print(f"📊 Scanning {timeframe} timeframe...")
        scan_results = await scanner.scan_market(timeframe=timeframe)
        
        if scan_results.empty:
            print("❌ No signals found")
            return pd.DataFrame()
        
        print(f"✅ Found {len(scan_results)} initial signals")
        
        # STEP 2: Filter by quality
        quality_signals = scan_results[
            (scan_results['confidence_score'] > 0.6) &
            (scan_results['composite_score'] > 50)
        ].copy()
        
        print(f"🔍 {len(quality_signals)} quality signals after filtering")
        
        # STEP 3: Add signal metadata
        quality_signals['timeframe'] = timeframe
        quality_signals['signal_strength'] = quality_signals.apply(
            lambda row: _calculate_signal_strength(row), axis=1
        )
        
        # STEP 4: Rank by strength
        final_signals = quality_signals.nlargest(top_n, 'signal_strength')
        
        # STEP 5: Add execution metadata
        final_signals['position_size_pct'] = final_signals['confidence_score'] * 0.1  # Max 10%
        final_signals['stop_loss_pct'] = 2.0  # 2% stop loss
        final_signals['take_profit_pct'] = 6.0  # 3:1 R/R ratio
        
        print(f"\n🎯 TOP {len(final_signals)} FINAL SIGNALS:")
        print("="*80)
        for idx, row in final_signals.iterrows():
            print(f"Symbol: {row['symbol']:<12} | Signal: {row['signal']:<12} | "
                  f"Score: {row['composite_score']:>5.1f} | "
                  f"Confidence: {row['confidence_score']:.2%} | "
                  f"Position: {row['position_size_pct']:.1%}")
        print("="*80)
        
        return final_signals[['symbol', 'signal', 'composite_score', 'confidence_score',
                             'position_size_pct', 'stop_loss_pct', 'take_profit_pct',
                             'price', 'timeframe']]
    
    finally:
        await exchange.close()

def _calculate_signal_strength(row):
    """Calculate overall signal strength"""
    base_score = row['composite_score'] / 100
    confidence_boost = row['confidence_score']
    
    # Bonus for validated signals
    cluster_bonus = 0.1 if row.get('cluster_validated', False) else 0
    regime_bonus = 0.05 if row.get('trend_formation_signal', False) else 0
    
    return base_score * confidence_boost + cluster_bonus + regime_bonus

async def get_integrated_signals_with_rl(timeframe='1h', use_rl=True):
    """
    Get final signals from integrated system (Scanner + RL + Meta-Controller)
    """
    import ccxt.async_support as ccxt
    exchange = ccxt.binance({'enableRateLimit': True})
    
    scanner = MomentumScanner(exchange=exchange)
    
    try:
        if use_rl:
            # Load trained RL agent
            rl_agent = RLTradingAgent(algorithm='PPO', model_path='models/rl_ppo_model')
            meta_controller = MetaController(strategy="confidence_blend")
            
            integrated_system = IntegratedTradingSystem(
                scanner=scanner,
                rl_agent=rl_agent,
                meta_controller=meta_controller
            )
            
            # Get integrated signals (rule-based + RL + meta-controller)
            signals = await integrated_system.generate_signals(timeframe=timeframe)
            
            print(f"\n🤖 INTEGRATED SIGNALS (Scanner + RL + Meta-Controller):")
            print("="*100)
            for idx, row in signals.head(10).iterrows():
                print(f"Symbol: {row['symbol']:<12} | "
                      f"Rule: {row['signal']:<12} | "
                      f"RL Position: {row['rl_position']:>6.2f} | "
                      f"Final Position: {row['final_position']:>6.2f} | "
                      f"Method: {row['method']:<15}")
            print("="*100)
            
            return signals
        else:
            # Scanner-only signals
            return await get_final_trading_signals(timeframe, top_n=10)
    
    finally:
        await exchange.close()

# Run example
if __name__ == "__main__":
    print("🚀 Getting Final Trading Signals...\n")
    
    # Option 1: Scanner-only signals (fast, simple)
    signals = asyncio.run(get_final_trading_signals(timeframe='1h', top_n=10))
    
    # Option 2: Integrated signals with RL (requires trained model)
    # signals = asyncio.run(get_integrated_signals_with_rl(timeframe='1h', use_rl=True))
