# trade_analyzer_agent.py
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class TradeAnalyzerAgent:
    def __init__(self):
        self.trades = []  # All trades
        self.pnl_history = []
        self.symbol_pnls = defaultdict(list)
        self.strategy_pnls = defaultdict(list)
        self.running_pnl = 0.0
    
    def record_trade(self, trade):
        try:
            symbol = trade.get("symbol", "UNKNOWN")
            entry = trade.get("entry")
            exit_ = trade.get("exit")
            pnl = round(trade.get("pnl", 0.0), 4)
            strategy = trade.get("strategy", "UNSPECIFIED")
            timestamp = trade.get("timestamp", time.time())
            
            trade_record = {
                "symbol": symbol,
                "entry": entry,
                "exit": exit_,
                "pnl": pnl,
                "strategy": strategy,
                "timestamp": timestamp,
            }
            
            self.trades.append(trade_record)
            self.pnl_history.append(pnl)
            self.symbol_pnls[symbol].append(pnl)
            self.strategy_pnls[strategy].append(pnl)
            self.running_pnl += pnl
            
            print(f"[📈 TradeAnalyzer] Logged trade {symbol} | PnL: {pnl}")
        except Exception as e:
            print(f"[TradeAnalyzer Error] Failed to record trade: {e}")
    
    def summary(self, top_n=5):
        print("\n========== TRADE ANALYSIS SUMMARY ==========")
        print(f"🔁 Total Trades: {len(self.trades)}")
        print(f"📊 Running Net PnL: {round(self.running_pnl, 4)}")
        
        if self.pnl_history:
            winning_trades = [p for p in self.pnl_history if p > 0]
            losing_trades = [p for p in self.pnl_history if p < 0]
            
            avg_win = round(sum(winning_trades) / max(1, len(winning_trades)), 4)
            avg_loss = round(sum(losing_trades) / max(1, len(losing_trades)), 4)
            
            win_rate = round(len(winning_trades) / len(self.pnl_history) * 100, 2)
            
            print(f"✅ Avg Win: {avg_win} | ❌ Avg Loss: {avg_loss}")
            print(f"📊 Win Rate: {win_rate}%")
        
        # Top performers
        print("\n🔝 Top Symbols:")
        sorted_syms = sorted(self.symbol_pnls.items(), key=lambda x: sum(x[1]), reverse=True)[:top_n]
        for sym, pnls in sorted_syms:
            print(f" • {sym} → Total: {round(sum(pnls), 4)} | Trades: {len(pnls)}")
        
        print("\n🧠 Strategy Stats:")
        for strat, pnls in self.strategy_pnls.items():
            avg = round(sum(pnls) / len(pnls), 4)
            total = round(sum(pnls), 4)
            print(f" • {strat} → Avg PnL: {avg} | Total: {total} | Trades: {len(pnls)}")
        
        print("============================================\n")
    
    def recent_trades(self, count=5):
        return self.trades[-count:]
    
    def get_total_pnl(self):
        return self.running_pnl
    
    def analyze_by_strategy(self):
        print("\n🎯 Strategy Performance")
        for strat, pnl_list in self.strategy_pnls.items():
            if pnl_list:
                avg = sum(pnl_list) / len(pnl_list)
                total = sum(pnl_list)
                winning_trades = len([p for p in pnl_list if p > 0])
                win_rate = winning_trades / len(pnl_list) * 100
                print(f"{strat}: Trades={len(pnl_list)}, AvgPnL={avg:.4f}, TotalPnL={total:.4f}, WinRate={win_rate:.2f}%")
            else:
                print(f"{strat}: No trades recorded")
    
    def plot_pnl_curve(self):
        if not self.pnl_history:
            print("No PnL data to plot.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Cumulative PnL
        cumulative_pnl = pd.Series(self.pnl_history).cumsum()
        plt.subplot(1, 2, 1)
        plt.plot(cumulative_pnl, label="Cumulative PnL", color="green", linewidth=2)
        plt.title("📈 Cumulative PnL Curve")
        plt.xlabel("Trade #")
        plt.ylabel("Net PnL")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # PnL histogram
        plt.subplot(1, 2, 2)
        plt.hist(self.pnl_history, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title("📊 PnL Distribution")
        plt.xlabel("PnL")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_to_csv(self, filepath="trade_log.csv"):
        if self.trades:
            df = pd.DataFrame(self.trades)
            df.to_csv(filepath, index=False)
            print(f"✅ Exported trade log to {filepath}")
        else:
            print("⚠️ No trade data to export.")
    
    def get_drawdown_stats(self):
        """Calculate maximum drawdown and other risk metrics"""
        if not self.pnl_history:
            return None
        
        cumulative_pnl = pd.Series(self.pnl_history).cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_drawdown_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
            else:
                current_drawdown_duration = 0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "current_drawdown": drawdown.iloc[-1] if len(drawdown) > 0 else 0
        }
    
    def performance_metrics(self):
        """Calculate key performance metrics"""
        if not self.pnl_history:
            print("No trades to analyze.")
            return
        
        print("\n📊 PERFORMANCE METRICS")
        print("=" * 40)
        
        # Basic stats
        total_trades = len(self.pnl_history)
        winning_trades = len([p for p in self.pnl_history if p > 0])
        losing_trades = len([p for p in self.pnl_history if p < 0])
        
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {winning_trades/total_trades*100:.2f}%")
        
        # PnL stats
        avg_win = sum(p for p in self.pnl_history if p > 0) / max(1, winning_trades)
        avg_loss = sum(p for p in self.pnl_history if p < 0) / max(1, losing_trades)
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        print(f"Average Win: {avg_win:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Drawdown stats
        dd_stats = self.get_drawdown_stats()
        if dd_stats:
            print(f"Max Drawdown: {dd_stats['max_drawdown']:.4f}")
            print(f"Max Drawdown Duration: {dd_stats['max_drawdown_duration']} trades")
        
        print("=" * 40)


# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = TradeAnalyzerAgent()
    
    # Example trades
    sample_trades = [
        {"symbol": "AAPL", "entry": 150.0, "exit": 155.0, "pnl": 5.0, "strategy": "momentum"},
        {"symbol": "GOOGL", "entry": 2500.0, "exit": 2480.0, "pnl": -20.0, "strategy": "mean_reversion"},
        {"symbol": "TSLA", "entry": 800.0, "exit": 820.0, "pnl": 20.0, "strategy": "momentum"},
        {"symbol": "AAPL", "entry": 155.0, "exit": 148.0, "pnl": -7.0, "strategy": "momentum"},
        {"symbol": "MSFT", "entry": 300.0, "exit": 310.0, "pnl": 10.0, "strategy": "breakout"},
    ]
    
    # Record trades
    for trade in sample_trades:
        analyzer.record_trade(trade)
    
    # Generate reports
    analyzer.summary()
    analyzer.analyze_by_strategy()
    analyzer.performance_metrics()
    
    # Export to CSV
    analyzer.export_to_csv("example_trades.csv")