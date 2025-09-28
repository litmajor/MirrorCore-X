
"""
High-Performance Vectorized Backtesting System for MirrorCore-X
Implements realistic fee modeling, slippage simulation, and regime analysis
"""

import numpy as np
import pandas as pd
import numba
from numba import jit, float64, int64, boolean
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class RegimeType(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"

@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    transaction_cost_pct: float = 0.075  # 0.075% total (maker + taker)
    slippage_pct: float = 0.05  # 0.05% slippage
    position_size_pct: float = 0.1  # 10% of capital per position
    max_positions: int = 5
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

@dataclass
class BacktestResults:
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    regime_performance: Dict[str, Dict[str, float]]
    equity_curve: np.ndarray
    trades: List[Dict[str, Any]]

class VectorizedBacktester:
    """High-performance vectorized backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = None
        
    @staticmethod
    @jit(nopython=True)
    def _calculate_positions_numba(signals: np.ndarray, 
                                   prices: np.ndarray,
                                   position_size: float,
                                   transaction_cost: float,
                                   slippage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-optimized position calculation"""
        n = len(signals)
        positions = np.zeros(n)
        equity = np.zeros(n)
        trades_pnl = np.zeros(n)
        
        cash = 1.0  # Start with 100% cash
        current_position = 0.0
        entry_price = 0.0
        
        for i in range(n):
            if signals[i] == 1 and current_position == 0:  # Buy signal
                # Calculate slippage and transaction costs
                execution_price = prices[i] * (1 + slippage)
                cost = transaction_cost
                
                # Calculate position size
                position_value = position_size
                shares = position_value / execution_price
                total_cost = position_value * (1 + cost)
                
                if cash >= total_cost:
                    current_position = shares
                    entry_price = execution_price
                    cash -= total_cost
                    
            elif signals[i] == -1 and current_position > 0:  # Sell signal
                # Calculate slippage and transaction costs
                execution_price = prices[i] * (1 - slippage)
                cost = transaction_cost
                
                # Close position
                gross_proceeds = current_position * execution_price
                net_proceeds = gross_proceeds * (1 - cost)
                
                # Calculate P&L
                trade_pnl = net_proceeds - (position_size)
                trades_pnl[i] = trade_pnl
                
                cash += net_proceeds
                current_position = 0.0
                entry_price = 0.0
            
            # Calculate current portfolio value
            if current_position > 0:
                position_value = current_position * prices[i]
                equity[i] = cash + position_value
            else:
                equity[i] = cash
            
            positions[i] = current_position
        
        return positions, equity, trades_pnl
    
    def run_backtest(self, 
                     price_data: pd.DataFrame,
                     signals: pd.DataFrame,
                     benchmark_returns: Optional[pd.Series] = None) -> BacktestResults:
        """Run comprehensive vectorized backtest"""
        
        logger.info("Starting vectorized backtest...")
        start_time = datetime.now()
        
        # Prepare data
        prices = price_data['close'].values
        signal_values = signals['signal_strength'].values if 'signal_strength' in signals else signals['signal'].values
        
        # Convert signals to numeric (1 for buy, -1 for sell, 0 for hold)
        numeric_signals = self._convert_signals_to_numeric(signal_values)
        
        # Calculate positions and equity using Numba
        positions, equity_curve, trades_pnl = self._calculate_positions_numba(
            numeric_signals,
            prices,
            self.config.position_size_pct,
            self.config.transaction_cost_pct / 100,
            self.config.slippage_pct / 100
        )
        
        # Calculate regime-based performance
        regime_performance = self._calculate_regime_performance(
            price_data, equity_curve, numeric_signals
        )
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(
            equity_curve, trades_pnl, benchmark_returns
        )
        
        # Add regime analysis
        results.regime_performance = regime_performance
        
        # Extract individual trades
        results.trades = self._extract_trades(
            price_data, numeric_signals, positions, trades_pnl
        )
        
        end_time = datetime.now()
        logger.info(f"Backtest completed in {(end_time - start_time).total_seconds():.2f} seconds")
        
        self.results = results
        return results
    
    def _convert_signals_to_numeric(self, signals: np.ndarray) -> np.ndarray:
        """Convert string signals to numeric"""
        numeric_signals = np.zeros(len(signals))
        
        for i, signal in enumerate(signals):
            if isinstance(signal, str):
                signal_lower = signal.lower()
                if any(word in signal_lower for word in ['buy', 'bullish', 'long']):
                    numeric_signals[i] = 1
                elif any(word in signal_lower for word in ['sell', 'bearish', 'short']):
                    numeric_signals[i] = -1
            elif isinstance(signal, (int, float)):
                if signal > 0.5:
                    numeric_signals[i] = 1
                elif signal < -0.5:
                    numeric_signals[i] = -1
        
        return numeric_signals
    
    def _calculate_regime_performance(self, 
                                      price_data: pd.DataFrame, 
                                      equity_curve: np.ndarray,
                                      signals: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate performance by market regime"""
        
        # Calculate volatility regime
        returns = price_data['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        vol_quartiles = volatility.quantile([0.25, 0.5, 0.75])
        
        # Calculate trend regime using ADX or simple MA crossover
        ma_short = price_data['close'].rolling(10).mean()
        ma_long = price_data['close'].rolling(50).mean()
        
        regimes = []
        for i in range(len(price_data)):
            vol = volatility.iloc[i] if i < len(volatility) else np.nan
            ma_s = ma_short.iloc[i] if i < len(ma_short) else np.nan
            ma_l = ma_long.iloc[i] if i < len(ma_long) else np.nan
            
            if pd.isna(vol) or pd.isna(ma_s) or pd.isna(ma_l):
                regimes.append(RegimeType.RANGING.value)
                continue
            
            if vol > vol_quartiles[0.75]:
                regimes.append(RegimeType.HIGH_VOLATILITY.value)
            elif ma_s > ma_l * 1.02:  # 2% above
                regimes.append(RegimeType.TRENDING_UP.value)
            elif ma_s < ma_l * 0.98:  # 2% below
                regimes.append(RegimeType.TRENDING_DOWN.value)
            else:
                regimes.append(RegimeType.RANGING.value)
        
        # Calculate performance by regime
        regime_performance = {}
        
        for regime_type in [r.value for r in RegimeType]:
            regime_mask = np.array([r == regime_type for r in regimes])
            
            if regime_mask.sum() == 0:
                continue
            
            regime_equity = equity_curve[regime_mask]
            regime_signals = signals[regime_mask]
            
            if len(regime_equity) > 1:
                regime_returns = np.diff(regime_equity) / regime_equity[:-1]
                
                regime_performance[regime_type] = {
                    'total_return': (regime_equity[-1] / regime_equity[0] - 1) * 100,
                    'volatility': np.std(regime_returns) * np.sqrt(252) * 100,
                    'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(252) if np.std(regime_returns) > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(regime_equity) * 100,
                    'num_periods': len(regime_equity),
                    'signal_frequency': np.sum(np.abs(regime_signals)) / len(regime_signals) * 100
                }
        
        return regime_performance
    
    def _calculate_performance_metrics(self, 
                                       equity_curve: np.ndarray,
                                       trades_pnl: np.ndarray,
                                       benchmark_returns: Optional[pd.Series] = None) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        
        # Basic returns calculation
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Annualized metrics
        trading_days = len(equity_curve)
        years = trading_days / 252
        annual_return = ((equity_curve[-1] / equity_curve[0]) ** (1/years) - 1) * 100
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.config.risk_free_rate * 100) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve) * 100
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade-based metrics
        winning_trades = trades_pnl[trades_pnl > 0]
        losing_trades = trades_pnl[trades_pnl < 0]
        
        total_trades = np.sum(np.abs(trades_pnl) > 0)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        
        profit_factor = abs(np.sum(winning_trades) / np.sum(losing_trades)) if np.sum(losing_trades) < 0 else np.inf
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=int(total_trades),
            regime_performance={},  # Will be filled separately
            equity_curve=equity_curve,
            trades=[]  # Will be filled separately
        )
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)
    
    def _extract_trades(self, 
                        price_data: pd.DataFrame,
                        signals: np.ndarray,
                        positions: np.ndarray,
                        trades_pnl: np.ndarray) -> List[Dict[str, Any]]:
        """Extract individual trade records"""
        trades = []
        in_position = False
        entry_idx = 0
        entry_price = 0.0
        
        for i, (signal, position, pnl) in enumerate(zip(signals, positions, trades_pnl)):
            if signal == 1 and not in_position and position > 0:  # Entry
                in_position = True
                entry_idx = i
                entry_price = price_data['close'].iloc[i]
                
            elif signal == -1 and in_position and pnl != 0:  # Exit
                exit_price = price_data['close'].iloc[i]
                duration = i - entry_idx
                
                trade = {
                    'entry_date': price_data.index[entry_idx],
                    'exit_date': price_data.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (self.config.position_size_pct * self.config.initial_capital)) * 100,
                    'duration_days': duration,
                    'direction': 'long'
                }
                
                trades.append(trade)
                in_position = False
        
        return trades
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive backtest report"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        report = f"""
MIRRORCORE-X VECTORIZED BACKTEST REPORT
{'='*50}

OVERALL PERFORMANCE
- Total Return: {self.results.total_return:.2f}%
- Annual Return: {self.results.annual_return:.2f}%
- Volatility: {self.results.volatility:.2f}%
- Sharpe Ratio: {self.results.sharpe_ratio:.2f}
- Maximum Drawdown: {self.results.max_drawdown:.2f}%
- Calmar Ratio: {self.results.calmar_ratio:.2f}

TRADING STATISTICS
- Total Trades: {self.results.total_trades}
- Win Rate: {self.results.win_rate:.2f}%
- Profit Factor: {self.results.profit_factor:.2f}
- Average Win: ${self.results.avg_win:.2f}
- Average Loss: ${self.results.avg_loss:.2f}

REGIME PERFORMANCE
"""
        
        for regime, metrics in self.results.regime_performance.items():
            report += f"\n{regime.upper()}:\n"
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report += f"  {metric}: {value:.2f}\n"
                else:
                    report += f"  {metric}: {value}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Backtest report saved to {save_path}")
        
        return report
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MirrorCore-X Backtest Results', fontsize=16)
        
        # Equity curve
        axes[0, 0].plot(self.results.equity_curve)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].grid(True)
        
        # Drawdown
        peak = np.maximum.accumulate(self.results.equity_curve)
        drawdown = (self.results.equity_curve - peak) / peak * 100
        axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_ylabel('Drawdown %')
        axes[0, 1].grid(True)
        
        # Trade P&L distribution
        if self.results.trades:
            trade_pnls = [trade['pnl'] for trade in self.results.trades]
            axes[1, 0].hist(trade_pnls, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Trade P&L Distribution')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # Regime performance
        if self.results.regime_performance:
            regimes = list(self.results.regime_performance.keys())
            returns = [self.results.regime_performance[r]['total_return'] for r in regimes]
            
            axes[1, 1].bar(regimes, returns)
            axes[1, 1].set_title('Performance by Regime')
            axes[1, 1].set_ylabel('Return (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Backtest plots saved to {save_path}")
        
        plt.show()

# Integration function for MirrorCore-X
async def run_mirrorcore_backtest(scanner_results: pd.DataFrame,
                                  price_data: pd.DataFrame,
                                  config: Optional[BacktestConfig] = None) -> BacktestResults:
    """Run backtest on MirrorCore-X scanner results"""
    
    if config is None:
        config = BacktestConfig()
    
    backtester = VectorizedBacktester(config)
    
    # Prepare signals from scanner results
    signals = scanner_results[['symbol', 'signal', 'composite_score']].copy()
    
    # Run backtest
    results = backtester.run_backtest(price_data, signals)
    
    # Generate and save report
    report = backtester.generate_report('backtest_report.txt')
    backtester.plot_results('backtest_results.png')
    
    logger.info("MirrorCore-X backtest completed successfully")
    
    return results

# Usage example
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Sample price data
    price_data = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
    }, index=dates)
    
    # Sample signals
    signals = pd.DataFrame({
        'signal': np.random.choice(['Buy', 'Sell', 'Hold'], len(dates), p=[0.1, 0.1, 0.8]),
        'signal_strength': np.random.uniform(0, 1, len(dates))
    }, index=dates)
    
    # Run backtest
    config = BacktestConfig(initial_capital=100000, transaction_cost_pct=0.075)
    backtester = VectorizedBacktester(config)
    
    results = backtester.run_backtest(price_data, signals)
    
    print(backtester.generate_report())
    backtester.plot_results()
