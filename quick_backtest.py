
import pandas as pd
import numpy as np
from typing import Dict

class QuickBacktest:
    """Fast backtest to validate strategy before going live"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        
    def run(self, historical_data: pd.DataFrame, strategy_signals: pd.DataFrame) -> Dict:
        """Run simple backtest with realistic fees"""
        
        capital = self.initial_capital
        positions = []
        trades = []
        
        for idx, row in strategy_signals.iterrows():
            signal = row.get('signal', 'neutral')
            price = row.get('price', 0)
            
            if signal in ['Strong Buy', 'Buy'] and not positions:
                # Enter long
                size = (capital * 0.95) / price  # 95% capital, 5% buffer
                fee = size * price * 0.003  # 0.3% fee
                positions.append({'type': 'long', 'entry': price, 'size': size})
                capital -= (size * price + fee)
                
            elif signal in ['Strong Sell', 'Sell'] and positions:
                # Exit long
                pos = positions.pop()
                pnl = (price - pos['entry']) * pos['size']
                fee = pos['size'] * price * 0.003
                capital += (pos['size'] * price - fee)
                
                trades.append({
                    'entry': pos['entry'],
                    'exit': price,
                    'pnl': pnl - fee,
                    'return_pct': ((price - pos['entry']) / pos['entry']) * 100
                })
        
        # Calculate metrics
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            wins = [t for t in trades if t['pnl'] > 0]
            win_rate = len(wins) / len(trades)
            
            returns = [t['return_pct'] for t in trades]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            return {
                'final_capital': capital,
                'total_pnl': total_pnl,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'max_drawdown': self._calc_max_drawdown(trades)
            }
        
        return {'error': 'No trades executed'}
    
    def _calc_max_drawdown(self, trades) -> float:
        """Calculate maximum drawdown"""
        cumulative = [self.initial_capital]
        for trade in trades:
            cumulative.append(cumulative[-1] + trade['pnl'])
        
        peak = cumulative[0]
        max_dd = 0
        
        for val in cumulative:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd * 100
