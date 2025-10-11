
"""
Strategy Backtest Comparison Tool
Tests all 19 strategies head-to-head with detailed performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import asyncio
from datetime import datetime
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class StrategyBacktestResult:
    strategy_name: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_win: float
    avg_loss: float
    recovery_factor: float
    equity_curve: List[float]

class StrategyBacktestComparison:
    """Compare all strategies using realistic backtesting"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.transaction_cost = 0.00075  # 0.075% per trade
        self.slippage = 0.0005  # 0.05% slippage
        
    async def compare_all_strategies(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest comparison for all 19 strategies"""
        
        strategies = self._get_all_strategies()
        results = []
        
        logger.info(f"Starting backtest comparison for {len(strategies)} strategies...")
        
        for strategy_name, strategy_agent in strategies.items():
            try:
                result = await self._backtest_strategy(
                    strategy_name, strategy_agent, market_data
                )
                results.append(result)
                logger.info(f"Completed: {strategy_name} - Sharpe: {result.sharpe_ratio:.2f}")
            except Exception as e:
                logger.error(f"Failed to backtest {strategy_name}: {e}")
        
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        # Generate comparison report
        comparison = self._generate_comparison_report(results)
        
        return comparison
    
    def _get_all_strategies(self) -> Dict[str, Any]:
        """Get all 19 strategies"""
        from strategy_trainer_agent import (
            UTSignalAgent, GradientTrendAgent, SupportResistanceAgent
        )
        from additional_strategies import (
            MeanReversionAgent, MomentumBreakoutAgent, VolatilityRegimeAgent,
            PairsTradingAgent, AnomalyDetectionAgent, SentimentMomentumAgent,
            RegimeChangeAgent
        )
        
        strategies = {
            # Core strategies (3)
            "UT_BOT": UTSignalAgent(),
            "GRADIENT_TREND": GradientTrendAgent(),
            "VOLUME_SR": SupportResistanceAgent(),
            
            # Additional strategies (7)
            "MEAN_REVERSION": MeanReversionAgent(),
            "MOMENTUM_BREAKOUT": MomentumBreakoutAgent(),
            "VOLATILITY_REGIME": VolatilityRegimeAgent(),
            "PAIRS_TRADING": PairsTradingAgent(),
            "ANOMALY_DETECTION": AnomalyDetectionAgent(),
            "SENTIMENT_MOMENTUM": SentimentMomentumAgent(),
            "REGIME_CHANGE": RegimeChangeAgent(),
        }
        
        # Add advanced strategies if available
        try:
            from advanced_strategies import (
                BayesianBeliefUpdater, LiquidityFlowTracker, MarketEntropyAnalyzer
            )
            strategies.update({
                "BAYESIAN_BELIEF": BayesianBeliefUpdater(),
                "LIQUIDITY_FLOW": LiquidityFlowTracker(),
                "MARKET_ENTROPY": MarketEntropyAnalyzer(),
            })
        except ImportError:
            logger.warning("Advanced strategies not available")
        
        return strategies
    
    async def _backtest_strategy(self, 
                                 strategy_name: str, 
                                 strategy_agent: Any, 
                                 market_data: pd.DataFrame) -> StrategyBacktestResult:
        """Run backtest for a single strategy"""
        
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        position = None
        
        for idx, row in market_data.iterrows():
            # Get strategy signal
            signal_data = {'market_data_df': pd.DataFrame([row])}
            signal_result = strategy_agent.update(signal_data)
            
            # Extract signal
            signal = self._extract_signal(signal_result)
            price = row['close']
            
            # Execute trades
            if signal in ['Strong Buy', 'Buy'] and position is None:
                # Enter long position
                shares = (capital * 0.95) / price
                entry_cost = shares * price * (1 + self.transaction_cost + self.slippage)
                
                if capital >= entry_cost:
                    position = {
                        'entry_price': price,
                        'shares': shares,
                        'entry_time': idx
                    }
                    capital -= entry_cost
                    
            elif signal in ['Strong Sell', 'Sell'] and position is not None:
                # Exit position
                exit_proceeds = position['shares'] * price * (1 - self.transaction_cost - self.slippage)
                pnl = exit_proceeds - (position['shares'] * position['entry_price'])
                
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'pnl': pnl,
                    'return': (price - position['entry_price']) / position['entry_price'] * 100
                })
                
                capital += exit_proceeds
                position = None
            
            # Update equity
            current_equity = capital
            if position:
                current_equity += position['shares'] * price
            
            equity_curve.append(current_equity)
        
        # Calculate metrics
        return self._calculate_metrics(strategy_name, equity_curve, trades)
    
    def _extract_signal(self, signal_result: Dict) -> str:
        """Extract signal from strategy result"""
        if not signal_result:
            return 'Hold'
        
        # Handle different signal formats
        for key, value in signal_result.items():
            if 'signals' in key.lower():
                if isinstance(value, dict):
                    for symbol_data in value.values():
                        if isinstance(symbol_data, dict):
                            return symbol_data.get('signal', 'Hold')
                        return 'Hold'
        
        return 'Hold'
    
    def _calculate_metrics(self, 
                          strategy_name: str, 
                          equity_curve: List[float],
                          trades: List[Dict]) -> StrategyBacktestResult:
        """Calculate performance metrics"""
        
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Total and annual returns
        total_return = (equity_array[-1] / equity_array[0] - 1) * 100
        years = len(equity_curve) / 252
        annual_return = ((equity_array[-1] / equity_array[0]) ** (1/years) - 1) * 100
        
        # Sharpe ratio
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 1
        sortino_ratio = (np.mean(returns) / downside_dev * np.sqrt(252)) if downside_dev > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100
        
        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            total_wins = sum([t['pnl'] for t in winning_trades])
            total_losses = abs(sum([t['pnl'] for t in losing_trades]))
            profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return StrategyBacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            recovery_factor=recovery_factor,
            equity_curve=equity_curve
        )
    
    def _generate_comparison_report(self, results: List[StrategyBacktestResult]) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        
        # Rankings by different metrics
        rankings = {
            'by_sharpe': sorted(results, key=lambda x: x.sharpe_ratio, reverse=True),
            'by_return': sorted(results, key=lambda x: x.total_return, reverse=True),
            'by_drawdown': sorted(results, key=lambda x: x.max_drawdown),
            'by_win_rate': sorted(results, key=lambda x: x.win_rate, reverse=True),
        }
        
        # Top performers
        top_5 = results[:5]
        
        # Summary statistics
        summary = {
            'total_strategies': len(results),
            'avg_sharpe': np.mean([r.sharpe_ratio for r in results]),
            'avg_return': np.mean([r.total_return for r in results]),
            'best_strategy': results[0].strategy_name if results else None,
            'best_sharpe': results[0].sharpe_ratio if results else 0,
        }
        
        return {
            'summary': summary,
            'rankings': {
                'by_sharpe': [
                    {
                        'rank': i+1,
                        'strategy': r.strategy_name,
                        'sharpe_ratio': r.sharpe_ratio,
                        'total_return': r.total_return,
                        'max_drawdown': r.max_drawdown,
                        'win_rate': r.win_rate
                    } for i, r in enumerate(rankings['by_sharpe'])
                ],
                'by_return': [
                    {
                        'rank': i+1,
                        'strategy': r.strategy_name,
                        'total_return': r.total_return,
                        'sharpe_ratio': r.sharpe_ratio
                    } for i, r in enumerate(rankings['by_return'][:10])
                ],
                'by_drawdown': [
                    {
                        'rank': i+1,
                        'strategy': r.strategy_name,
                        'max_drawdown': r.max_drawdown,
                        'sharpe_ratio': r.sharpe_ratio
                    } for i, r in enumerate(rankings['by_drawdown'][:10])
                ]
            },
            'top_performers': [
                {
                    'strategy': r.strategy_name,
                    'total_return': r.total_return,
                    'annual_return': r.annual_return,
                    'sharpe_ratio': r.sharpe_ratio,
                    'sortino_ratio': r.sortino_ratio,
                    'max_drawdown': r.max_drawdown,
                    'calmar_ratio': r.calmar_ratio,
                    'win_rate': r.win_rate,
                    'profit_factor': r.profit_factor,
                    'total_trades': r.total_trades,
                    'recovery_factor': r.recovery_factor
                } for r in top_5
            ],
            'all_results': [
                {
                    'strategy': r.strategy_name,
                    'total_return': r.total_return,
                    'sharpe_ratio': r.sharpe_ratio,
                    'max_drawdown': r.max_drawdown,
                    'win_rate': r.win_rate,
                    'total_trades': r.total_trades,
                    'equity_curve': r.equity_curve[-50:]  # Last 50 points
                } for r in results
            ],
            'timestamp': datetime.now().isoformat()
        }

# Test function
async def run_comparison_test():
    """Test the comparison tool"""
    import ccxt.async_support as ccxt
    
    # Get test data
    exchange = ccxt.binance({'enableRateLimit': True})
    
    try:
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=500)
        await exchange.close()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Run comparison
        comparator = StrategyBacktestComparison()
        results = await comparator.compare_all_strategies(df)
        
        # Print results
        print("\n=== STRATEGY BACKTEST COMPARISON ===")
        print(f"\nTotal Strategies: {results['summary']['total_strategies']}")
        print(f"Best Strategy: {results['summary']['best_strategy']}")
        print(f"Best Sharpe: {results['summary']['best_sharpe']:.2f}")
        
        print("\n=== TOP 5 PERFORMERS (by Sharpe) ===")
        for r in results['rankings']['by_sharpe'][:5]:
            print(f"{r['rank']}. {r['strategy']:20s} | Sharpe: {r['sharpe_ratio']:6.2f} | Return: {r['total_return']:7.2f}% | DD: {r['max_drawdown']:6.2f}%")
        
        # Save results
        with open('strategy_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nResults saved to strategy_comparison_results.json")
        
    except Exception as e:
        logger.error(f"Comparison test failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_comparison_test())
