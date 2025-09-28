
# trade_analyzer_agent.py
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import asyncio
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TradeAnalyzerAgent:
    """Enhanced TradeAnalyzerAgent compatible with high-performance SyncBus"""
    
    def __init__(self):
        self.trades = []  # All trades
        self.pnl_history = []
        self.symbol_pnls = defaultdict(list)
        self.strategy_pnls = defaultdict(list)
        self.running_pnl = 0.0
        
        # Enhanced SyncBus compatibility
        self.name = "TradeAnalyzerAgent"
        self.data_interests = ['trades', 'market_data']
        self.is_paused = False
        self.command_queue = []
        self.last_update = time.time()
        
        # Performance tracking
        self.daily_pnl = defaultdict(float)
        self.monthly_stats = defaultdict(dict)
        self.risk_metrics = {
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced update method compatible with SyncBus architecture"""
        # Process commands first
        await self._process_commands()
        
        if self.is_paused:
            return {'status': 'paused'}
        
        try:
            # Process trade data from SyncBus
            trades_data = data.get('trades', [])
            
            # Handle both single trade and list of trades
            if isinstance(trades_data, list):
                for trade in trades_data:
                    if isinstance(trade, dict):
                        self.record_trade(trade)
            elif isinstance(trades_data, dict):
                self.record_trade(trades_data)
            
            # Update risk metrics
            self._update_risk_metrics()
            
            self.last_update = time.time()
            
            return {
                'status': 'active',
                'total_trades': len(self.trades),
                'running_pnl': self.running_pnl,
                'risk_metrics': self.risk_metrics.copy(),
                'last_update': self.last_update,
                'confidence': self._calculate_confidence()
            }
            
        except Exception as e:
            logger.error(f"TradeAnalyzerAgent update failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _process_commands(self):
        """Process commands from SyncBus"""
        while self.command_queue:
            command_msg = self.command_queue.pop(0)
            command = command_msg.get('command')
            
            if command == 'pause':
                self.is_paused = True
                logger.info("TradeAnalyzer paused")
            elif command == 'resume':
                self.is_paused = False
                logger.info("TradeAnalyzer resumed")
            elif command == 'emergency_stop':
                self.is_paused = True
                logger.warning("TradeAnalyzer emergency stopped")
            elif command == 'reset_data':
                self._reset_all_data()
                logger.info("TradeAnalyzer data reset")
            elif command == 'export_trades':
                self.export_to_csv("emergency_export.csv")
                logger.info("Emergency trade export completed")
    
    def _reset_all_data(self):
        """Reset all trading data"""
        self.trades.clear()
        self.pnl_history.clear()
        self.symbol_pnls.clear()
        self.strategy_pnls.clear()
        self.running_pnl = 0.0
        self.daily_pnl.clear()
        self.monthly_stats.clear()
        self._reset_risk_metrics()
    
    def _reset_risk_metrics(self):
        """Reset risk metrics to defaults"""
        self.risk_metrics = {
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    def record_trade(self, trade):
        """Enhanced trade recording with better validation"""
        try:
            # Validate trade data
            if not isinstance(trade, dict):
                logger.warning(f"Invalid trade format: {type(trade)}")
                return
            
            symbol = trade.get("symbol", "UNKNOWN")
            entry = trade.get("entry")
            exit_ = trade.get("exit")
            pnl = trade.get("pnl", 0.0)
            strategy = trade.get("strategy", "UNSPECIFIED")
            timestamp = trade.get("timestamp", time.time())
            
            # Validate numeric fields
            try:
                pnl = float(pnl)
                timestamp = float(timestamp)
                if entry is not None:
                    entry = float(entry)
                if exit_ is not None:
                    exit_ = float(exit_)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid numeric values in trade: {e}")
                return
            
            trade_record = {
                "symbol": symbol,
                "entry": entry,
                "exit": exit_,
                "pnl": round(pnl, 4),
                "strategy": strategy,
                "timestamp": timestamp,
            }
            
            self.trades.append(trade_record)
            self.pnl_history.append(pnl)
            self.symbol_pnls[symbol].append(pnl)
            self.strategy_pnls[strategy].append(pnl)
            self.running_pnl += pnl
            
            # Update daily tracking
            date_key = time.strftime('%Y-%m-%d', time.localtime(timestamp))
            self.daily_pnl[date_key] += pnl
            
            logger.debug(f"[📈 TradeAnalyzer] Logged trade {symbol} | PnL: {pnl}")
            
        except Exception as e:
            logger.error(f"[TradeAnalyzer Error] Failed to record trade: {e}")
    
    def _update_risk_metrics(self):
        """Update comprehensive risk metrics"""
        if not self.pnl_history:
            return
        
        # Basic metrics
        winning_trades = [p for p in self.pnl_history if p > 0]
        losing_trades = [p for p in self.pnl_history if p < 0]
        
        self.risk_metrics['win_rate'] = len(winning_trades) / len(self.pnl_history)
        self.risk_metrics['avg_win'] = np.mean(winning_trades) if winning_trades else 0.0
        self.risk_metrics['avg_loss'] = np.mean(losing_trades) if losing_trades else 0.0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1
        self.risk_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Maximum drawdown
        cumulative_pnl = np.cumsum(self.pnl_history)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        self.risk_metrics['max_drawdown'] = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Sharpe ratio (simplified)
        if len(self.pnl_history) > 1:
            mean_return = np.mean(self.pnl_history)
            std_return = np.std(self.pnl_history)
            self.risk_metrics['sharpe_ratio'] = mean_return / std_return if std_return > 0 else 0.0
        else:
            self.risk_metrics['sharpe_ratio'] = 0.0
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence based on recent performance"""
        if not self.pnl_history:
            return 0.5
        
        recent_trades = self.pnl_history[-20:]  # Last 20 trades
        if len(recent_trades) < 5:
            return 0.5
        
        win_rate = sum(1 for pnl in recent_trades if pnl > 0) / len(recent_trades)
        avg_pnl = np.mean(recent_trades)
        
        # Confidence based on win rate and average PnL
        confidence = (win_rate * 0.7) + (min(max(avg_pnl, -0.1), 0.1) + 0.1) / 0.2 * 0.3
        return np.clip(confidence, 0.0, 1.0)
    
    def summary(self, top_n=5):
        """Enhanced summary with more detailed metrics"""
        print("\n" + "="*60)
        print("📊 ENHANCED TRADE ANALYSIS SUMMARY")
        print("="*60)
        print(f"🔁 Total Trades: {len(self.trades)}")
        print(f"📊 Running Net PnL: ${self.running_pnl:.2f}")
        
        if self.pnl_history:
            print(f"✅ Win Rate: {self.risk_metrics['win_rate']*100:.1f}%")
            print(f"💰 Avg Win: ${self.risk_metrics['avg_win']:.2f}")
            print(f"❌ Avg Loss: ${self.risk_metrics['avg_loss']:.2f}")
            print(f"📈 Profit Factor: {self.risk_metrics['profit_factor']:.2f}")
            print(f"📉 Max Drawdown: ${self.risk_metrics['max_drawdown']:.2f}")
            print(f"⚡ Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.2f}")
        
        # Top performing symbols
        print(f"\n🔝 Top {top_n} Symbols by PnL:")
        sorted_syms = sorted(self.symbol_pnls.items(), key=lambda x: sum(x[1]), reverse=True)[:top_n]
        for sym, pnls in sorted_syms:
            total_pnl = sum(pnls)
            trade_count = len(pnls)
            avg_pnl = total_pnl / trade_count if trade_count > 0 else 0
            print(f" • {sym:<12} → Total: ${total_pnl:>8.2f} | Trades: {trade_count:>3} | Avg: ${avg_pnl:>6.2f}")
        
        # Strategy performance
        print(f"\n🧠 Strategy Performance:")
        for strat, pnls in self.strategy_pnls.items():
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / len(pnls) if pnls else 0
            win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0
            print(f" • {strat:<20} → Total: ${total_pnl:>8.2f} | Avg: ${avg_pnl:>6.2f} | Win%: {win_rate:>5.1f}% | Trades: {len(pnls):>3}")
        
        print("="*60 + "\n")
    
    def recent_trades(self, count=5):
        """Get recent trades with enhanced formatting"""
        recent = self.trades[-count:] if self.trades else []
        
        print(f"\n📋 Last {len(recent)} Trades:")
        print("-" * 80)
        
        for trade in recent:
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(trade['timestamp']))
            pnl_str = f"${trade['pnl']:>8.2f}"
            pnl_color = "🟢" if trade['pnl'] > 0 else "🔴" if trade['pnl'] < 0 else "⚪"
            
            print(f"{timestamp_str} | {trade['symbol']:<12} | {trade['strategy']:<15} | {pnl_color} {pnl_str}")
        
        print("-" * 80)
        return recent
    
    def get_total_pnl(self):
        """Get total PnL with validation"""
        return round(self.running_pnl, 2)
    
    def analyze_by_strategy(self):
        """Enhanced strategy analysis"""
        print("\n🎯 DETAILED STRATEGY PERFORMANCE")
        print("=" * 70)
        
        for strat, pnl_list in self.strategy_pnls.items():
            if pnl_list:
                # Calculate metrics
                total_pnl = sum(pnl_list)
                avg_pnl = total_pnl / len(pnl_list)
                winning_trades = [p for p in pnl_list if p > 0]
                losing_trades = [p for p in pnl_list if p < 0]
                
                win_rate = len(winning_trades) / len(pnl_list) * 100
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                
                # Risk metrics
                volatility = np.std(pnl_list) if len(pnl_list) > 1 else 0
                sharpe = avg_pnl / volatility if volatility > 0 else 0
                
                # Grade assignment
                grade = self._assign_strategy_grade(win_rate, avg_pnl, sharpe)
                
                print(f"\n📊 {strat}")
                print(f"   Trades: {len(pnl_list):>6} | Total PnL: ${total_pnl:>8.2f} | Grade: {grade}")
                print(f"   Win Rate: {win_rate:>5.1f}% | Avg Win: ${avg_win:>6.2f} | Avg Loss: ${avg_loss:>6.2f}")
                print(f"   Volatility: ${volatility:>5.2f} | Sharpe: {sharpe:>5.2f}")
            else:
                print(f"\n📊 {strat}: No trades recorded")
        
        print("=" * 70)
    
    def _assign_strategy_grade(self, win_rate: float, avg_pnl: float, sharpe: float) -> str:
        """Assign letter grade to strategy"""
        if avg_pnl > 5 and win_rate > 60 and sharpe > 1.0:
            return "A+"
        elif avg_pnl > 2 and win_rate > 55 and sharpe > 0.5:
            return "A"
        elif avg_pnl > 0 and win_rate > 50 and sharpe > 0:
            return "B"
        elif avg_pnl > -1 and win_rate > 45:
            return "C"
        elif avg_pnl > -3:
            return "D"
        else:
            return "F"
    
    def get_drawdown_stats(self):
        """Enhanced drawdown analysis"""
        if not self.pnl_history:
            return None
        
        cumulative_pnl = pd.Series(self.pnl_history).cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        
        max_drawdown = drawdown.min()
        
        # Drawdown duration analysis
        in_drawdown = drawdown < 0
        drawdown_periods = []
        start_dd = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_dd is None:
                start_dd = i
            elif not is_dd and start_dd is not None:
                drawdown_periods.append(i - start_dd)
                start_dd = None
        
        # If still in drawdown
        if start_dd is not None:
            drawdown_periods.append(len(in_drawdown) - start_dd)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "avg_drawdown_duration": avg_drawdown_duration,
            "current_drawdown": drawdown.iloc[-1] if len(drawdown) > 0 else 0,
            "drawdown_periods": len(drawdown_periods)
        }
    
    def performance_metrics(self):
        """Enhanced performance metrics display"""
        print("\n📊 COMPREHENSIVE PERFORMANCE METRICS")
        print("=" * 60)
        
        if not self.pnl_history:
            print("No trades to analyze.")
            return
        
        # Basic stats
        total_trades = len(self.pnl_history)
        winning_trades = len([p for p in self.pnl_history if p > 0])
        losing_trades = len([p for p in self.pnl_history if p < 0])
        breakeven_trades = total_trades - winning_trades - losing_trades
        
        print(f"📈 Trading Statistics:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winning: {winning_trades} ({winning_trades/total_trades*100:.1f}%)")
        print(f"   Losing: {losing_trades} ({losing_trades/total_trades*100:.1f}%)")
        print(f"   Breakeven: {breakeven_trades}")
        
        # PnL stats
        print(f"\n💰 Profit & Loss:")
        print(f"   Total PnL: ${self.running_pnl:.2f}")
        print(f"   Average Win: ${self.risk_metrics['avg_win']:.2f}")
        print(f"   Average Loss: ${self.risk_metrics['avg_loss']:.2f}")
        print(f"   Profit Factor: {self.risk_metrics['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.2f}")
        
        # Drawdown stats
        dd_stats = self.get_drawdown_stats()
        if dd_stats:
            print(f"\n📉 Risk Metrics:")
            print(f"   Max Drawdown: ${dd_stats['max_drawdown']:.2f}")
            print(f"   Max DD Duration: {dd_stats['max_drawdown_duration']} trades")
            print(f"   Avg DD Duration: {dd_stats['avg_drawdown_duration']:.1f} trades")
            print(f"   Current Drawdown: ${dd_stats['current_drawdown']:.2f}")
        
        print("=" * 60)
    
    def export_to_csv(self, filepath="trade_log.csv"):
        """Enhanced CSV export with additional fields"""
        if self.trades:
            df = pd.DataFrame(self.trades)
            
            # Add calculated fields
            df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
            df['time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Add win/loss flag
            df['result'] = df['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS' if x < 0 else 'BE')
            
            df.to_csv(filepath, index=False)
            logger.info(f"✅ Exported {len(df)} trades to {filepath}")
            print(f"✅ Exported trade log to {filepath}")
        else:
            logger.warning("⚠️ No trade data to export.")
            print("⚠️ No trade data to export.")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status for monitoring"""
        return {
            'name': self.name,
            'is_paused': self.is_paused,
            'total_trades': len(self.trades),
            'running_pnl': self.running_pnl,
            'win_rate': self.risk_metrics['win_rate'],
            'max_drawdown': self.risk_metrics['max_drawdown'],
            'sharpe_ratio': self.risk_metrics['sharpe_ratio'],
            'last_update': self.last_update,
            'confidence': self._calculate_confidence(),
            'data_interests': self.data_interests
        }


# Example usage
if __name__ == "__main__":
    # Test the enhanced trade analyzer
    async def test_analyzer():
        analyzer = TradeAnalyzerAgent()
        
        # Test with sample trades
        sample_trades = [
            {"symbol": "BTC/USDT", "entry": 50000, "exit": 51000, "pnl": 100, "strategy": "momentum"},
            {"symbol": "ETH/USDT", "entry": 3000, "exit": 2950, "pnl": -50, "strategy": "mean_reversion"},
            {"symbol": "BTC/USDT", "entry": 51000, "exit": 51500, "pnl": 75, "strategy": "momentum"},
        ]
        
        # Test update method
        for trade in sample_trades:
            data = {'trades': [trade]}
            result = await analyzer.update(data)
            print(f"Update result: {result}")
        
        # Test command processing
        analyzer.command_queue.append({'command': 'pause'})
        result = await analyzer.update({'trades': []})
        print(f"Paused result: {result}")
        
        # Resume and test again
        analyzer.command_queue.append({'command': 'resume'})
        result = await analyzer.update({'trades': []})
        print(f"Resumed result: {result}")
        
        # Show summary
        analyzer.summary()
        analyzer.analyze_by_strategy()
        analyzer.performance_metrics()
        
        # Export test
        analyzer.export_to_csv("test_trades.csv")
    
    import asyncio
    asyncio.run(test_analyzer())
