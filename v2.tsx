import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, AlertTriangle, Target, Shield, 
  Brain, Zap, Play, Pause, Settings, BarChart3, PieChart, LineChart,
  Clock, DollarSign, Percent, ArrowUp, ArrowDown, Minus, CheckCircle,
  XCircle, Info, Layers, Gauge, Trophy, Star, Eye, EyeOff
} from 'lucide-react';
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

// Professional market data structure
interface MarketFrame {
  timestamp: number;
  symbol: string;
  price: {
    open: number;
    high: number;
    low: number;
    close: number;
  };
  volume: number;
  indicators: {
    rsi: number;
    macd: { line: number; signal: number; histogram: number };
    bb: { upper: number; middle: number; lower: number };
    ema20: number;
    ema50: number;
    vwap: number;
    atr: number;
  };
  orderFlow: {
    bidVolume: number;
    askVolume: number;
    netFlow: number;
    largeOrders: number;
    smallOrders: number;
  };
  marketMicrostructure: {
    spread: number;
    depth: number;
    imbalance: number;
    toxicity: number;
  };
}

interface Strategy {
  id: string;
  name: string;
  description: string;
  signals: Signal[];
  riskParams: RiskParameters;
  performance: PerformanceMetrics;
  isActive: boolean;
}

interface Signal {
  timestamp: number;
  type: 'BUY' | 'SELL' | 'HOLD';
  strength: number; // 0-1
  confidence: number; // 0-1
  price: number;
  reasoning: string[];
  riskReward: number;
  stopLoss: number;
  takeProfit: number;
}

interface RiskParameters {
  maxPositionSize: number;
  maxDrawdown: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  maxConcurrentTrades: number;
  kellyFraction: number;
}

interface PerformanceMetrics {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  totalTrades: number;
  alpha: number;
  beta: number;
  calmarRatio: number;
  sortinoRatio: number;
}

interface BacktestResult {
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalCapital: number;
  performance: PerformanceMetrics;
  trades: Trade[];
  equityCurve: { date: string; equity: number; drawdown: number }[];
  monthlyReturns: { month: string; return: number }[];
}

interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  entryTime: number;
  exitTime?: number;
  entryPrice: number;
  exitPrice?: number;
  quantity: number;
  pnl?: number;
  commission: number;
  status: 'OPEN' | 'CLOSED' | 'CANCELLED';
}

// Advanced Signal Generation Engine
class SignalEngine {
  private static calculateTechnicalScore(frame: MarketFrame): number {
    const { rsi, macd, bb, ema20, ema50 } = frame.indicators;
    const price = frame.price.close;
    
    let score = 0;
    
    // RSI momentum
    if (rsi < 30) score += 0.3; // Oversold
    else if (rsi > 70) score -= 0.3; // Overbought
    else score += (50 - Math.abs(rsi - 50)) / 100; // Neutral momentum
    
    // MACD trend
    if (macd.line > macd.signal && macd.histogram > 0) score += 0.25;
    else if (macd.line < macd.signal && macd.histogram < 0) score -= 0.25;
    
    // Bollinger Bands position
    const bbPosition = (price - bb.lower) / (bb.upper - bb.lower);
    if (bbPosition < 0.2) score += 0.2; // Near lower band
    else if (bbPosition > 0.8) score -= 0.2; // Near upper band
    
    // EMA trend
    if (ema20 > ema50 && price > ema20) score += 0.25; // Uptrend
    else if (ema20 < ema50 && price < ema20) score -= 0.25; // Downtrend
    
    return Math.max(-1, Math.min(1, score));
  }
  
  private static calculateOrderFlowScore(frame: MarketFrame): number {
    const { bidVolume, askVolume, netFlow, largeOrders } = frame.orderFlow;
    const totalVolume = bidVolume + askVolume;
    
    if (totalVolume === 0) return 0;
    
    // Volume imbalance
    const imbalance = (bidVolume - askVolume) / totalVolume;
    
    // Large order activity
    const largeOrderRatio = largeOrders / frame.volume;
    
    // Net flow trend
    const flowScore = Math.tanh(netFlow / frame.volume);
    
    return (imbalance * 0.4 + flowScore * 0.4 + largeOrderRatio * 0.2) * 2;
  }
  
  private static calculateMicrostructureScore(frame: MarketFrame): number {
    const { spread, depth, imbalance, toxicity } = frame.marketMicrostructure;
    
    // Spread quality (lower is better)
    const spreadScore = Math.max(0, 1 - spread / frame.price.close);
    
    // Market depth (higher is better)
    const depthScore = Math.min(1, depth / (frame.volume * 0.1));
    
    // Order book imbalance
    const imbalanceScore = Math.tanh(imbalance);
    
    // Toxicity (lower is better)
    const toxicityScore = Math.max(0, 1 - toxicity);
    
    return (spreadScore * 0.3 + depthScore * 0.2 + imbalanceScore * 0.3 + toxicityScore * 0.2);
  }
  
  static generateSignal(frames: MarketFrame[], index: number): Signal | null {
    if (index < 20) return null; // Need historical data
    
    const current = frames[index];
    const previous = frames.slice(Math.max(0, index - 20), index);
    
    const technicalScore = this.calculateTechnicalScore(current);
    const orderFlowScore = this.calculateOrderFlowScore(current);
    const microstructureScore = this.calculateMicrostructureScore(current);
    
    // Combine scores with weights based on market conditions
    const volatility = current.indicators.atr / current.price.close;
    const techWeight = volatility < 0.02 ? 0.5 : 0.3; // Less weight in high vol
    const flowWeight = 0.3;
    const microWeight = 0.2;
    
    const compositeScore = (
      technicalScore * techWeight +
      orderFlowScore * flowWeight +
      microstructureScore * microWeight
    );
    
    const strength = Math.abs(compositeScore);
    const confidence = this.calculateConfidence(previous, current);
    
    // Signal threshold
    if (strength < 0.3 || confidence < 0.6) return null;
    
    const type = compositeScore > 0 ? 'BUY' : 'SELL';
    
    // Calculate risk/reward
    const atr = current.indicators.atr;
    const stopLoss = type === 'BUY' 
      ? current.price.close - (atr * 2)
      : current.price.close + (atr * 2);
    
    const takeProfit = type === 'BUY'
      ? current.price.close + (atr * 3)
      : current.price.close - (atr * 3);
    
    const riskReward = Math.abs(takeProfit - current.price.close) / 
                      Math.abs(current.price.close - stopLoss);
    
    return {
      timestamp: current.timestamp,
      type,
      strength,
      confidence,
      price: current.price.close,
      reasoning: this.generateReasoning(technicalScore, orderFlowScore, microstructureScore),
      riskReward,
      stopLoss,
      takeProfit
    };
  }
  
  private static calculateConfidence(historical: MarketFrame[], current: MarketFrame): number {
    // Pattern consistency
    const recentSignals = historical.slice(-10);
    const trendConsistency = this.calculateTrendConsistency(recentSignals);
    
    // Volume confirmation
    const avgVolume = historical.reduce((sum, f) => sum + f.volume, 0) / historical.length;
    const volumeConfirmation = Math.min(1, current.volume / avgVolume);
    
    // Market microstructure health
    const microHealth = 1 - current.marketMicrostructure.toxicity;
    
    return (trendConsistency * 0.4 + volumeConfirmation * 0.3 + microHealth * 0.3);
  }
  
  private static calculateTrendConsistency(frames: MarketFrame[]): number {
    if (frames.length < 5) return 0.5;
    
    let consistency = 0;
    for (let i = 1; i < frames.length; i++) {
      const curr = frames[i];
      const prev = frames[i - 1];
      
      if ((curr.price.close > prev.price.close && curr.indicators.ema20 > prev.indicators.ema20) ||
          (curr.price.close < prev.price.close && curr.indicators.ema20 < prev.indicators.ema20)) {
        consistency++;
      }
    }
    
    return consistency / (frames.length - 1);
  }
  
  private static generateReasoning(tech: number, flow: number, micro: number): string[] {
    const reasons: string[] = [];
    
    if (Math.abs(tech) > 0.3) {
      reasons.push(`Technical analysis ${tech > 0 ? 'bullish' : 'bearish'} (${(tech * 100).toFixed(1)}%)`);
    }
    
    if (Math.abs(flow) > 0.3) {
      reasons.push(`Order flow ${flow > 0 ? 'accumulation' : 'distribution'} detected`);
    }
    
    if (micro > 0.6) {
      reasons.push('Healthy market microstructure');
    } else if (micro < 0.4) {
      reasons.push('Degraded market microstructure');
    }
    
    return reasons;
  }
}

// Professional Backtesting Engine
class BacktestEngine {
  static async runBacktest(
    frames: MarketFrame[],
    strategy: Omit<Strategy, 'performance' | 'isActive'>,
    initialCapital: number = 100000
  ): Promise<BacktestResult> {
    
    const trades: Trade[] = [];
    let currentCapital = initialCapital;
    let openTrades: Trade[] = [];
    const equityCurve: { date: string; equity: number; drawdown: number }[] = [];
    let maxEquity = initialCapital;
    
    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];
      const signal = SignalEngine.generateSignal(frames, i);
      
      // Close existing trades if needed
      openTrades = openTrades.filter(trade => {
        const shouldClose = this.shouldCloseTrade(trade, frame, strategy.riskParams);
        if (shouldClose) {
          const closedTrade = this.closeTrade(trade, frame);
          trades.push(closedTrade);
          currentCapital += closedTrade.pnl || 0;
          return false;
        }
        return true;
      });
      
      // Open new trades
      if (signal && openTrades.length < strategy.riskParams.maxConcurrentTrades) {
        const trade = this.openTrade(signal, frame, currentCapital, strategy.riskParams);
        if (trade) {
          openTrades.push(trade);
          currentCapital -= trade.quantity * trade.entryPrice;
        }
      }
      
      // Calculate current equity
      const openPnL = openTrades.reduce((sum, trade) => {
        const currentPrice = frame.price.close;
        const unrealizedPnL = trade.side === 'BUY' 
          ? (currentPrice - trade.entryPrice) * trade.quantity
          : (trade.entryPrice - currentPrice) * trade.quantity;
        return sum + unrealizedPnL;
      }, 0);
      
      const currentEquity = currentCapital + openPnL;
      maxEquity = Math.max(maxEquity, currentEquity);
      const drawdown = (maxEquity - currentEquity) / maxEquity;
      
      equityCurve.push({
        date: new Date(frame.timestamp).toISOString(),
        equity: currentEquity,
        drawdown: drawdown * 100
      });
    }
    
    // Close remaining open trades
    openTrades.forEach(trade => {
      const closedTrade = this.closeTrade(trade, frames[frames.length - 1]);
      trades.push(closedTrade);
    });
    
    const performance = this.calculatePerformanceMetrics(trades, equityCurve, initialCapital);
    const monthlyReturns = this.calculateMonthlyReturns(equityCurve);
    
    return {
      startDate: new Date(frames[0].timestamp).toISOString(),
      endDate: new Date(frames[frames.length - 1].timestamp).toISOString(),
      initialCapital,
      finalCapital: equityCurve[equityCurve.length - 1].equity,
      performance,
      trades,
      equityCurve,
      monthlyReturns
    };
  }
  
  private static shouldCloseTrade(trade: Trade, frame: MarketFrame, riskParams: RiskParameters): boolean {
    const currentPrice = frame.price.close;
    const pnlPercent = trade.side === 'BUY'
      ? (currentPrice - trade.entryPrice) / trade.entryPrice
      : (trade.entryPrice - currentPrice) / trade.entryPrice;
    
    // Stop loss
    if (pnlPercent <= -riskParams.stopLossPercent / 100) return true;
    
    // Take profit
    if (pnlPercent >= riskParams.takeProfitPercent / 100) return true;
    
    return false;
  }
  
  private static openTrade(signal: Signal, frame: MarketFrame, capital: number, riskParams: RiskParameters): Trade | null {
    const riskAmount = capital * riskParams.maxPositionSize;
    const quantity = Math.floor(riskAmount / frame.price.close);
    
    if (quantity === 0) return null;
    
    return {
      id: `trade_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      symbol: frame.symbol,
      side: signal.type as 'BUY' | 'SELL',
      entryTime: frame.timestamp,
      entryPrice: frame.price.close,
      quantity,
      commission: quantity * frame.price.close * 0.001, // 0.1% commission
      status: 'OPEN'
    };
  }
  
  private static closeTrade(trade: Trade, frame: MarketFrame): Trade {
    const exitPrice = frame.price.close;
    const pnl = trade.side === 'BUY'
      ? (exitPrice - trade.entryPrice) * trade.quantity
      : (trade.entryPrice - exitPrice) * trade.quantity;
    
    return {
      ...trade,
      exitTime: frame.timestamp,
      exitPrice,
      pnl: pnl - trade.commission,
      status: 'CLOSED'
    };
  }
  
  private static calculatePerformanceMetrics(trades: Trade[], equityCurve: any[], initialCapital: number): PerformanceMetrics {
    const closedTrades = trades.filter(t => t.status === 'CLOSED');
    const totalReturn = ((equityCurve[equityCurve.length - 1].equity - initialCapital) / initialCapital) * 100;
    
    const returns = equityCurve.map((point, i) => {
      if (i === 0) return 0;
      return (point.equity - equityCurve[i - 1].equity) / equityCurve[i - 1].equity;
    }).slice(1);
    
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const returnStd = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length);
    const sharpeRatio = returnStd === 0 ? 0 : (avgReturn / returnStd) * Math.sqrt(252);
    
    const maxDrawdown = Math.max(...equityCurve.map(p => p.drawdown));
    
    const winningTrades = closedTrades.filter(t => (t.pnl || 0) > 0);
    const losingTrades = closedTrades.filter(t => (t.pnl || 0) <= 0);
    
    const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
    const avgWin = winningTrades.length > 0 ? winningTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / winningTrades.length : 0;
    const avgLoss = losingTrades.length > 0 ? Math.abs(losingTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / losingTrades.length) : 0;
    const profitFactor = avgLoss === 0 ? 0 : avgWin / avgLoss;
    
    const calmarRatio = maxDrawdown === 0 ? 0 : totalReturn / maxDrawdown;
    
    const downside = returns.filter(r => r < 0);
    const downsideStd = downside.length > 0 ? Math.sqrt(downside.reduce((sum, r) => sum + r * r, 0) / downside.length) : 0;
    const sortinoRatio = downsideStd === 0 ? 0 : (avgReturn / downsideStd) * Math.sqrt(252);
    
    return {
      totalReturn,
      sharpeRatio,
      maxDrawdown,
      winRate,
      profitFactor,
      avgWin,
      avgLoss,
      totalTrades: closedTrades.length,
      alpha: totalReturn - 5, // Assuming 5% benchmark
      beta: 1.0, // Simplified
      calmarRatio,
      sortinoRatio
    };
  }
  
  private static calculateMonthlyReturns(equityCurve: any[]): { month: string; return: number }[] {
    const monthlyData: { [key: string]: { start: number; end: number } } = {};
    
    equityCurve.forEach(point => {
      const date = new Date(point.date);
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      
      if (!monthlyData[monthKey]) {
        monthlyData[monthKey] = { start: point.equity, end: point.equity };
      } else {
        monthlyData[monthKey].end = point.equity;
      }
    });
    
    return Object.entries(monthlyData).map(([month, data]) => ({
      month,
      return: ((data.end - data.start) / data.start) * 100
    }));
  }
}

// Generate realistic market data for demo
const generateMarketData = (days: number = 30): MarketFrame[] => {
  const frames: MarketFrame[] = [];
  let price = 100;
  const startTime = Date.now() - (days * 24 * 60 * 60 * 1000);
  
  for (let i = 0; i < days * 24; i++) { // Hourly data
    const timestamp = startTime + (i * 60 * 60 * 1000);
    
    // Realistic price movement
    const trend = Math.sin(i / 100) * 0.1;
    const volatility = 0.02 + Math.random() * 0.01;
    const change = (Math.random() - 0.5) * volatility + trend;
    
    const newPrice = price * (1 + change);
    const volume = 1000 + Math.random() * 2000;
    
    // Calculate indicators
    const rsi = 30 + Math.random() * 40; // Simplified
    const atr = price * volatility;
    
    frames.push({
      timestamp,
      symbol: 'BTCUSD',
      price: {
        open: price,
        high: Math.max(price, newPrice),
        low: Math.min(price, newPrice),
        close: newPrice
      },
      volume,
      indicators: {
        rsi,
        macd: { line: 0, signal: 0, histogram: 0 },
        bb: { upper: newPrice * 1.02, middle: newPrice, lower: newPrice * 0.98 },
        ema20: newPrice * 0.99,
        ema50: newPrice * 0.98,
        vwap: newPrice,
        atr
      },
      orderFlow: {
        bidVolume: volume * (0.4 + Math.random() * 0.2),
        askVolume: volume * (0.4 + Math.random() * 0.2),
        netFlow: (Math.random() - 0.5) * volume * 0.1,
        largeOrders: Math.floor(volume * 0.1),
        smallOrders: Math.floor(volume * 0.9)
      },
      marketMicrostructure: {
        spread: newPrice * 0.001,
        depth: volume * 10,
        imbalance: (Math.random() - 0.5) * 0.2,
        toxicity: Math.random() * 0.3
      }
    });
    
    price = newPrice;
  }
  
  return frames;
};

// Main Elite Trading Interface Component
const EliteTradingInterface: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketFrame[]>([]);
  const [currentSignal, setCurrentSignal] = useState<Signal | null>(null);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('momentum');
  const [showBacktest, setShowBacktest] = useState(false);
  const [loading, setLoading] = useState(false);
  
  const intervalRef = useRef<number | null>(null);
  
  // Strategy definitions
  const strategies: { [key: string]: Omit<Strategy, 'performance' | 'isActive'> } = {
    momentum: {
      id: 'momentum',
      name: 'Multi-Factor Momentum',
      description: 'Combines technical analysis, order flow, and market microstructure',
      signals: [],
      riskParams: {
        maxPositionSize: 0.02,
        maxDrawdown: 0.15,
        stopLossPercent: 2,
        takeProfitPercent: 6,
        maxConcurrentTrades: 3,
        kellyFraction: 0.25
      }
    },
    meanReversion: {
      id: 'meanReversion',
      name: 'Smart Mean Reversion',
      description: 'Identifies oversold/overbought conditions with volume confirmation',
      signals: [],
      riskParams: {
        maxPositionSize: 0.015,
        maxDrawdown: 0.10,
        stopLossPercent: 1.5,
        takeProfitPercent: 3,
        maxConcurrentTrades: 5,
        kellyFraction: 0.20
      }
    }
  };
  
  const updateMarketData = useCallback(() => {
    setMarketData(prev => {
      const newData = [...prev];
      if (newData.length > 0) {
        // Add new frame
        const lastFrame = newData[newData.length - 1];
        const newFrame = { ...lastFrame };
        newFrame.timestamp = Date.now();
        
        // Update price
        const change = (Math.random() - 0.5) * 0.01;
        newFrame.price.close = lastFrame.price.close * (1 + change);
        newFrame.price.open = lastFrame.price.close;
        newFrame.price.high = Math.max(newFrame.price.open, newFrame.price.close);
        newFrame.price.low = Math.min(newFrame.price.open, newFrame.price.close);
        
        newData.push(newFrame);
        
        // Keep last 1000 frames
        if (newData.length > 1000) {
          newData.shift();
        }
      }
      return newData;
    });
  }, []);
  
  useEffect(() => {
    // Initialize with historical data
    const initialData = generateMarketData(30);
    setMarketData(initialData);
  }, []);
  
  useEffect(() => {
    if (marketData.length > 20) {
      const signal = SignalEngine.generateSignal(marketData, marketData.length - 1);
      setCurrentSignal(signal);
    }
  }, [marketData]);
  
  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(updateMarketData, 1000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, updateMarketData]);
  
  const runBacktest = async () => {
    setLoading(true);
    try {
      const strategy = strategies[selectedStrategy];
      const result = await BacktestEngine.runBacktest(marketData, strategy);
      setBacktestResult(result);
      setShowBacktest(true);
    } catch (error) {
      console.error('Backtest failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const latestFrame = marketData[marketData.length - 1];
  const chartData = marketData.slice(-50).map(frame => ({
    time: new Date(frame.timestamp).toLocaleTimeString(),
    price: frame.price.close,
    volume: frame.volume,
    rsi: frame.indicators.rsi
  }));
  
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };
  
  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900/10 to-slate-800 text-white p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-blue-600 rounded-xl">
            <Brain className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Elite Trading Interface
            </h1>
            <p className="text-slate-400">Professional Grade Market Analysis & Execution</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
          >
            {Object.entries(strategies).map(([key, strategy]) => (
              <option key={key} value={key}>{strategy.name}</option>
            ))}
          </select>
          
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              isRunning 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isRunning ? 'Pause' : 'Start'}</span>
          </button>
          
          <button
            onClick={runBacktest}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            <BarChart3 className="w-4 h-4" />
            <span>{loading ? 'Running...' : 'Backtest'}</span>
          </button>
        </div>
      </div>
      
      {/* Main Dashboard */}
      <div className="grid grid-cols-12 gap-6">
        {/* Price Chart */}
        <div className="col-span-8">
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold flex items-center space-x-2">
                <LineChart className="w-5 h-5 text-blue-400" />
                <span>Market Analysis</span>
                {latestFrame && (
                  <span className="text-sm text-slate-400 ml-4">
                    {latestFrame.symbol} - {formatCurrency(latestFrame.price.close)}
                  </span>
                )}
              </h2>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                <span className="text-sm text-slate-400">
                  {isRunning ? 'Live' : 'Paused'}
                </span>
              </div>
            </div>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsLineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#3B82F6" 
                    strokeWidth={2}
                    dot={false}
                  />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Signal Panel */}
          {currentSignal && (
            <div className="mt-6 bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <Target className="w-5 h-5 text-yellow-400" />
                  <span>Active Signal</span>
                </h3>
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                  currentSignal.type === 'BUY' 
                    ? 'bg-green-600/20 text-green-400 border border-green-600/30'
                    : 'bg-red-600/20 text-red-400 border border-red-600/30'
                }`}>
                  {currentSignal.type}
                </div>
              </div>
              
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div className="text-center">
                  <div className="text-sm text-slate-400 mb-1">Strength</div>
                  <div className="text-lg font-semibold">
                    {(currentSignal.strength * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-slate-400 mb-1">Confidence</div>
                  <div className="text-lg font-semibold">
                    {(currentSignal.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-slate-400 mb-1">Risk/Reward</div>
                  <div className="text-lg font-semibold">
                    1:{currentSignal.riskReward.toFixed(1)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-slate-400 mb-1">Entry Price</div>
                  <div className="text-lg font-semibold">
                    {formatCurrency(currentSignal.price)}
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="text-sm font-medium text-slate-300 mb-2">Signal Reasoning:</div>
                {currentSignal.reasoning.map((reason, index) => (
                  <div key={index} className="flex items-center space-x-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-slate-300">{reason}</span>
                  </div>
                ))}
              </div>
              
              <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-700">
                <div>
                  <div className="text-sm text-slate-400 mb-1">Stop Loss</div>
                  <div className="font-semibold text-red-400">
                    {formatCurrency(currentSignal.stopLoss)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-slate-400 mb-1">Take Profit</div>
                  <div className="font-semibold text-green-400">
                    {formatCurrency(currentSignal.takeProfit)}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Sidebar */}
        <div className="col-span-4 space-y-6">
          {/* Market Overview */}
          {latestFrame && (
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Activity className="w-5 h-5 text-green-400" />
                <span>Market Overview</span>
              </h3>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Price</span>
                  <span className="font-semibold">{formatCurrency(latestFrame.price.close)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Volume</span>
                  <span className="font-semibold">{latestFrame.volume.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">RSI</span>
                  <span className={`font-semibold ${
                    latestFrame.indicators.rsi > 70 ? 'text-red-400' :
                    latestFrame.indicators.rsi < 30 ? 'text-green-400' : 'text-yellow-400'
                  }`}>
                    {latestFrame.indicators.rsi.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">ATR</span>
                  <span className="font-semibold">{formatCurrency(latestFrame.indicators.atr)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Spread</span>
                  <span className="font-semibold">{formatCurrency(latestFrame.marketMicrostructure.spread)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Net Flow</span>
                  <span className={`font-semibold ${
                    latestFrame.orderFlow.netFlow > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {latestFrame.orderFlow.netFlow > 0 ? '+' : ''}{latestFrame.orderFlow.netFlow.toFixed(0)}
                  </span>
                </div>
              </div>
            </div>
          )}
          
          {/* Strategy Performance */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Trophy className="w-5 h-5 text-yellow-400" />
              <span>Strategy Performance</span>
            </h3>
            
            {backtestResult ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Total Return</span>
                  <span className={`font-semibold ${
                    backtestResult.performance.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatPercent(backtestResult.performance.totalReturn)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Sharpe Ratio</span>
                  <span className="font-semibold">{backtestResult.performance.sharpeRatio.toFixed(2)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Max Drawdown</span>
                  <span className="font-semibold text-red-400">
                    -{backtestResult.performance.maxDrawdown.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Win Rate</span>
                  <span className="font-semibold">{backtestResult.performance.winRate.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Profit Factor</span>
                  <span className="font-semibold">{backtestResult.performance.profitFactor.toFixed(2)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Total Trades</span>
                  <span className="font-semibold">{backtestResult.performance.totalTrades}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Calmar Ratio</span>
                  <span className="font-semibold">{backtestResult.performance.calmarRatio.toFixed(2)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Sortino Ratio</span>
                  <span className="font-semibold">{backtestResult.performance.sortinoRatio.toFixed(2)}</span>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <PieChart className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                <p className="text-slate-400">Run backtest to see performance metrics</p>
              </div>
            )}
          </div>
          
          {/* Risk Management */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Shield className="w-5 h-5 text-blue-400" />
              <span>Risk Management</span>
            </h3>
            
            <div className="space-y-4">
              {Object.entries(strategies[selectedStrategy].riskParams).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center">
                  <span className="text-slate-400 text-sm capitalize">
                    {key.replace(/([A-Z])/g, ' $1').toLowerCase()}
                  </span>
                  <span className="font-semibold text-sm">
                    {typeof value === 'number' && key.includes('Percent') ? `${value}%` :
                     typeof value === 'number' && key.includes('Fraction') ? value.toFixed(2) :
                     typeof value === 'number' && key.includes('Size') ? `${(value * 100).toFixed(1)}%` :
                     value.toString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
          
          {/* Order Flow */}
          {latestFrame && (
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Layers className="w-5 h-5 text-purple-400" />
                <span>Order Flow</span>
              </h3>
              
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400">Bid Volume</span>
                    <span className="text-green-400">{latestFrame.orderFlow.bidVolume.toFixed(0)}</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div 
                      className="bg-green-400 h-2 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${(latestFrame.orderFlow.bidVolume / latestFrame.volume) * 100}%` 
                      }}
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400">Ask Volume</span>
                    <span className="text-red-400">{latestFrame.orderFlow.askVolume.toFixed(0)}</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div 
                      className="bg-red-400 h-2 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${(latestFrame.orderFlow.askVolume / latestFrame.volume) * 100}%` 
                      }}
                    />
                  </div>
                </div>
                
                <div className="pt-2 border-t border-slate-700">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Large Orders</span>
                    <span className="font-semibold">{latestFrame.orderFlow.largeOrders}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Market Toxicity</span>
                    <span className={`font-semibold ${
                      latestFrame.marketMicrostructure.toxicity > 0.5 ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {(latestFrame.marketMicrostructure.toxicity * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Backtest Results Modal */}
      {showBacktest && backtestResult && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold flex items-center space-x-2">
                <BarChart3 className="w-6 h-6 text-blue-400" />
                <span>Backtest Results</span>
              </h2>
              <button
                onClick={() => setShowBacktest(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <XCircle className="w-6 h-6" />
              </button>
            </div>
            
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Performance Summary</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Period</span>
                    <span>
                      {new Date(backtestResult.startDate).toLocaleDateString()} - {new Date(backtestResult.endDate).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Initial Capital</span>
                    <span>{formatCurrency(backtestResult.initialCapital)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Final Capital</span>
                    <span className={backtestResult.finalCapital >= backtestResult.initialCapital ? 'text-green-400' : 'text-red-400'}>
                      {formatCurrency(backtestResult.finalCapital)}
                    </span>
                  </div>
                  <div className="flex justify-between font-semibold">
                    <span className="text-slate-400">Total P&L</span>
                    <span className={backtestResult.finalCapital >= backtestResult.initialCapital ? 'text-green-400' : 'text-red-400'}>
                      {formatCurrency(backtestResult.finalCapital - backtestResult.initialCapital)}
                    </span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-3">Key Metrics</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Sharpe Ratio</span>
                    <span className={backtestResult.performance.sharpeRatio > 1 ? 'text-green-400' : 'text-yellow-400'}>
                      {backtestResult.performance.sharpeRatio.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Calmar Ratio</span>
                    <span>{backtestResult.performance.calmarRatio.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Sortino Ratio</span>
                    <span>{backtestResult.performance.sortinoRatio.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Alpha</span>
                    <span className={backtestResult.performance.alpha > 0 ? 'text-green-400' : 'text-red-400'}>
                      {formatPercent(backtestResult.performance.alpha)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Equity Curve Chart */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3">Equity Curve</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={backtestResult.equityCurve.map(point => ({
                    date: new Date(point.date).toLocaleDateString(),
                    equity: point.equity,
                    drawdown: -point.drawdown
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="date" stroke="#9CA3AF" fontSize={12} />
                    <YAxis stroke="#9CA3AF" fontSize={12} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="equity" 
                      stroke="#3B82F6" 
                      fill="#3B82F6" 
                      fillOpacity={0.1}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="drawdown" 
                      stroke="#EF4444" 
                      fill="#EF4444" 
                      fillOpacity={0.1}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Recent Trades */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Recent Trades</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-2">Time</th>
                      <th className="text-left py-2">Side</th>
                      <th className="text-left py-2">Entry</th>
                      <th className="text-left py-2">Exit</th>
                      <th className="text-left py-2">Quantity</th>
                      <th className="text-left py-2">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtestResult.trades.slice(-10).map((trade) => (
                      <tr key={trade.id} className="border-b border-slate-800">
                        <td className="py-2">{new Date(trade.entryTime).toLocaleString()}</td>
                        <td className="py-2">
                          <span className={`px-2 py-1 rounded text-xs ${
                            trade.side === 'BUY' ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
                          }`}>
                            {trade.side}
                          </span>
                        </td>
                        <td className="py-2">{formatCurrency(trade.entryPrice)}</td>
                        <td className="py-2">{trade.exitPrice ? formatCurrency(trade.exitPrice) : '-'}</td>
                        <td className="py-2">{trade.quantity}</td>
                        <td className="py-2">
                          {trade.pnl !== undefined ? (
                            <span className={trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                              {formatCurrency(trade.pnl)}
                            </span>
                          ) : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EliteTradingInterface;