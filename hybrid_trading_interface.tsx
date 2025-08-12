import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  TrendingUp, TrendingDown, Activity, AlertTriangle, Target, Shield, Brain, Zap, Play, Pause, Settings, BarChart3, PieChart, LineChart, Clock, DollarSign, Percent, ArrowUp, ArrowDown, Minus, CheckCircle, XCircle, Info, Layers, Gauge, Trophy, Star, Eye, EyeOff, Crown
} from 'lucide-react';
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { fetchHybridMarketFrames } from './services/marketDataProvider';

// --- Hybrid MarketFrame: Combines advanced analytics fields from temporal_precognition_engine and v2 ---
interface HybridMarketFrame {
  timestamp: number;
  symbol: string;
  price: {
    open: number;
    high: number;
    low: number;
    close: number;
  };
  volume: number;
  volatility: number;
  momentum_score: number;
  rsi: number;
  trend_score: number;
  confidence_score: number;
  composite_score: number;
  volume_score: number;
  anchor_index: number;
  price_range: number;
  predicted_return: number;
  predicted_consistent: number;
  indicators: {
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
  temporal_ghost: {
    next_moves: Array<{
      timestamp: number;
      predicted_price: number;
      confidence: number;
      probability: number;
      pattern_match: number;
    }>;
    certainty_river: number;
    time_distortion: number;
    pattern_resonance: number;
    quantum_coherence: number;
    fractal_depth: number;
  };
  intention_field: {
    accumulation_pressure: number;
    breakout_membrane: number;
    momentum_vector: {
      direction: 'up' | 'down' | 'sideways';
      strength: number;
      timing: number;
      acceleration: number;
    };
    whale_presence: number;
    institutional_flow: number;
    retail_sentiment: number;
    liquidity_depth: number;
  };
  power_level: {
    edge_percentage: number;
    opportunity_intensity: number;
    risk_shadow: number;
    profit_magnetism: number;
    certainty_coefficient: number;
    godmode_factor: number;
    quantum_advantage: number;
    reality_bend_strength: number;
  };
  market_layers: {
    fear_greed_index: number;
    volatility_regime: 'low' | 'medium' | 'high' | 'extreme';
    trend_strength: number;
    support_resistance: {
      nearest_support: number;
      nearest_resistance: number;
      strength: number;
    };
    market_phase: 'accumulation' | 'markup' | 'distribution' | 'markdown';
  };
  consciousness_matrix: {
    awareness_level: number;
    collective_intelligence: number;
    prediction_accuracy: number;
    reality_stability: number;
    temporal_variance: number;
  };
}

// --- Hybrid Signal and Backtest types (from v2) ---
interface HybridSignal {
  timestamp: number;
  type: 'BUY' | 'SELL' | 'HOLD';
  strength: number;
  confidence: number;
  price: number;
  reasoning: string[];
  riskReward: number;
  stopLoss: number;
  takeProfit: number;
}


// --- Hybrid UI: Professional charting + advanced analytics overlays ---
const HybridTradingInterface: React.FC = () => {
  const [marketData, setMarketData] = useState<HybridMarketFrame[]>([]);
  const [currentFrame, setCurrentFrame] = useState<HybridMarketFrame | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedView, setSelectedView] = useState<'overview' | 'quantum' | 'orderflow'>('overview');
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    fetchHybridMarketFrames('1d')
      .then(data => {
        setMarketData(data);
        setCurrentFrame(data[data.length - 1]);
      })
      .catch(err => {
        console.error('Failed to load market data', err);
      });
  }, []);

  useEffect(() => {
    if (isRunning && marketData.length > 0) {
      intervalRef.current = window.setInterval(() => {
        setMarketData(prev => {
          const newData = [...prev];
          const lastFrame = newData[newData.length - 1];
          
          // Create new frame with realistic price movement
          const change = (Math.random() - 0.5) * 0.015;
          const newPrice = lastFrame.price.close * (1 + change);
          
          const newFrame: HybridMarketFrame = {
            ...lastFrame,
            timestamp: Date.now(),
            price: {
              open: lastFrame.price.close,
              high: Math.max(lastFrame.price.close, newPrice),
              low: Math.min(lastFrame.price.close, newPrice),
              close: newPrice
            },
            volume: 1000 + Math.random() * 3000,
            rsi: Math.max(0, Math.min(100, lastFrame.rsi + (Math.random() - 0.5) * 5)),
            momentum_score: Math.abs(change) * 1000 + Math.random() * 50,
            power_level: {
              ...lastFrame.power_level,
              godmode_factor: Math.max(0, Math.min(1, lastFrame.power_level.godmode_factor + (Math.random() - 0.5) * 0.1))
            },
            intention_field: {
              ...lastFrame.intention_field,
              whale_presence: Math.max(0, Math.min(1, lastFrame.intention_field.whale_presence + (Math.random() - 0.5) * 0.05))
            }
          };
          
          newData.push(newFrame);
          if (newData.length > 1000) newData.shift();
          return newData;
        });
      }, 2000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, marketData.length]);

  useEffect(() => {
    if (marketData.length > 0) {
      setCurrentFrame(marketData[marketData.length - 1]);
    }
  }, [marketData]);

  const chartData = useMemo(() => 
    marketData.slice(-100).map((frame, index) => ({
      time: new Date(frame.timestamp).toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit' 
      }),
      price: frame.price.close,
      volume: frame.volume,
      rsi: frame.rsi,
      trend_score: frame.trend_score,
      confidence_score: frame.confidence_score * 100,
      composite_score: frame.composite_score * 100,
      godmode: frame.power_level.godmode_factor * 100,
      whale: frame.intention_field.whale_presence * 100,
      quantum: frame.temporal_ghost.quantum_coherence * 100,
      high: frame.price.high,
      low: frame.price.low
    })), [marketData]
  );

  const priceChange = useMemo(() => {
    if (marketData.length < 2) return { value: 0, percentage: 0 };
    const current = marketData[marketData.length - 1]?.price.close || 0;
    const previous = marketData[marketData.length - 2]?.price.close || 0;
    const value = current - previous;
    const percentage = previous ? (value / previous) * 100 : 0;
    return { value, percentage };
  }, [marketData]);

  const renderSidebar = () => {
    if (!currentFrame) return null;

    switch (selectedView) {
      case 'overview':
        return (
          <div className="space-y-6">
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Activity className="w-5 h-5 text-green-400" />
                <span>Market Overview</span>
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Price</span>
                  <div className="text-right">
                    <div className="font-semibold text-lg">${currentFrame.price.close.toFixed(2)}</div>
                    <div className={`text-sm flex items-center ${
                      priceChange.value > 0 ? 'text-green-400' : priceChange.value < 0 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {priceChange.value > 0 ? <ArrowUp className="w-3 h-3 mr-1" /> : 
                       priceChange.value < 0 ? <ArrowDown className="w-3 h-3 mr-1" /> : 
                       <Minus className="w-3 h-3 mr-1" />}
                      {priceChange.percentage.toFixed(2)}%
                    </div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Volume</span>
                  <span className="font-semibold">{currentFrame.volume.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">RSI</span>
                  <span className={`font-semibold ${
                    currentFrame.rsi > 70 ? 'text-red-400' :
                    currentFrame.rsi < 30 ? 'text-green-400' : 'text-yellow-400'
                  }`}>
                    {currentFrame.rsi.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Volatility</span>
                  <span className="font-semibold">{(currentFrame.volatility * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Market Phase</span>
                  <span className={`font-semibold capitalize ${
                    currentFrame.market_layers.market_phase === 'markup' ? 'text-green-400' :
                    currentFrame.market_layers.market_phase === 'markdown' ? 'text-red-400' : 'text-yellow-400'
                  }`}>
                    {currentFrame.market_layers.market_phase}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Target className="w-5 h-5 text-blue-400" />
                <span>Support & Resistance</span>
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Resistance</span>
                  <span className="font-semibold text-red-400">
                    ${currentFrame.market_layers.support_resistance.nearest_resistance.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Current</span>
                  <span className="font-semibold">${currentFrame.price.close.toFixed(2)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Support</span>
                  <span className="font-semibold text-green-400">
                    ${currentFrame.market_layers.support_resistance.nearest_support.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
      
      case 'quantum':
        return (
          <div className="space-y-6">
            <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-500/30 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Crown className="w-5 h-5 text-purple-400" />
                <span>Quantum Analytics</span>
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Godmode Factor</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-slate-700 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${currentFrame.power_level.godmode_factor * 100}%` }}
                      />
                    </div>
                    <span className="font-semibold text-purple-400">
                      {(currentFrame.power_level.godmode_factor * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Quantum Coherence</span>
                  <span className="font-semibold text-blue-400">
                    {(currentFrame.temporal_ghost.quantum_coherence * 100).toFixed(0)}%
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Reality Bend</span>
                  <span className="font-semibold text-amber-400">
                    {(currentFrame.power_level.reality_bend_strength * 100).toFixed(0)}%
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Certainty River</span>
                  <span className="font-semibold text-cyan-400">
                    {(currentFrame.temporal_ghost.certainty_river * 100).toFixed(0)}%
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Prediction Accuracy</span>
                  <span className="font-semibold text-emerald-400">
                    {(currentFrame.consciousness_matrix.prediction_accuracy * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
            
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                <span>Power Levels</span>
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Edge %</span>
                  <span className="font-semibold text-green-400">
                    {currentFrame.power_level.edge_percentage.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Profit Magnetism</span>
                  <span className="font-semibold text-yellow-400">
                    {(currentFrame.power_level.profit_magnetism * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Risk Shadow</span>
                  <span className="font-semibold text-red-400">
                    {(currentFrame.power_level.risk_shadow * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'orderflow':
        return (
          <div className="space-y-6">
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Activity className="w-5 h-5 text-orange-400" />
                <span>Order Flow</span>
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Bid Volume</span>
                  <span className="font-semibold text-green-400">
                    {currentFrame.orderFlow.bidVolume.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Ask Volume</span>
                  <span className="font-semibold text-red-400">
                    {currentFrame.orderFlow.askVolume.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Net Flow</span>
                  <span className={`font-semibold ${
                    currentFrame.orderFlow.netFlow > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {currentFrame.orderFlow.netFlow > 0 ? '+' : ''}{currentFrame.orderFlow.netFlow.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Whale Presence</span>
                  <span className="font-semibold text-purple-400">
                    {(currentFrame.intention_field.whale_presence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
            
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Layers className="w-5 h-5 text-indigo-400" />
                <span>Market Microstructure</span>
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Spread</span>
                  <span className="font-semibold">{currentFrame.marketMicrostructure.spread.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Depth</span>
                  <span className="font-semibold">{currentFrame.marketMicrostructure.depth.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Imbalance</span>
                  <span className={`font-semibold ${
                    currentFrame.marketMicrostructure.imbalance > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {(currentFrame.marketMicrostructure.imbalance * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Toxicity</span>
                  <span className="font-semibold text-amber-400">
                    {(currentFrame.marketMicrostructure.toxicity * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
        
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900/10 to-black text-white p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl">
            <Brain className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Quantum Elite Trading Interface
            </h1>
            <p className="text-slate-400">Advanced Hybrid Market Analytics & Execution Platform</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              isRunning 
                ? 'bg-red-600 hover:bg-red-700 shadow-lg shadow-red-600/25' 
                : 'bg-green-600 hover:bg-green-700 shadow-lg shadow-green-600/25'
            }`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isRunning ? 'Pause Feed' : 'Start Live'}</span>
          </button>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
            <span className="text-sm text-slate-400">
              {isRunning ? 'LIVE FEED' : 'PAUSED'}
            </span>
          </div>
          <button
            onClick={() => {
              fetchHybridMarketFrames().then(data => {
                setMarketData(data);
                setCurrentFrame(data[data.length - 1]);
              });
            }}
            className="ml-4 px-3 py-2 bg-blue-700 rounded text-white"
          >
            Refresh Data
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-2 mb-6">
        {[
          { id: 'overview', label: 'Market Overview', icon: BarChart3 },
          { id: 'quantum', label: 'Quantum Analytics', icon: Crown },
          { id: 'orderflow', label: 'Order Flow', icon: Activity }
        ].map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setSelectedView(tab.id as typeof selectedView)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                selectedView === tab.id
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/25'
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 hover:text-white'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Main Dashboard */}
      <div className="grid grid-cols-12 gap-6">
        {/* Price Chart with overlays */}
        <div className="col-span-8">
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4">
                <h2 className="text-xl font-semibold flex items-center space-x-2">
                  <LineChart className="w-5 h-5 text-blue-400" />
                  <span>Advanced Market Analysis</span>
                </h2>
                {currentFrame && (
                  <div className="flex items-center space-x-4 text-sm">
                    <span className="text-slate-400">{currentFrame.symbol}</span>
                    <span className="font-bold text-xl">${currentFrame.price.close.toFixed(2)}</span>
                    <span className={`flex items-center font-medium ${
                      priceChange.value > 0 ? 'text-green-400' : priceChange.value < 0 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {priceChange.value > 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : 
                       priceChange.value < 0 ? <TrendingDown className="w-4 h-4 mr-1" /> : 
                       <Minus className="w-4 h-4 mr-1" />}
                      {priceChange.percentage.toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="text-xs text-slate-400">
                  Last Updated: {currentFrame ? new Date(currentFrame.timestamp).toLocaleTimeString() : '--:--:--'}
                </div>
              </div>
            </div>
            
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="godmodeGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.2}/>
                      <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis 
                    dataKey="time" 
                    stroke="#9CA3AF" 
                    fontSize={12}
                    tick={{ fill: '#9CA3AF' }}
                  />
                  <YAxis 
                    yAxisId="price"
                    orientation="right"
                    stroke="#9CA3AF" 
                    fontSize={12}
                    tick={{ fill: '#9CA3AF' }}
                    domain={['dataMin - 100', 'dataMax + 100']}
                  />
                  <YAxis 
                    yAxisId="indicators"
                    orientation="left"
                    stroke="#9CA3AF" 
                    fontSize={12}
                    tick={{ fill: '#9CA3AF' }}
                    domain={[0, 100]}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      backdropFilter: 'blur(8px)'
                    }}
                    labelStyle={{ color: '#F3F4F6' }}
                  />
                  
                  {/* Price Area */}
                  <Area 
                    yAxisId="price"
                    type="monotone" 
                    dataKey="price" 
                    stroke="#3B82F6" 
                    strokeWidth={2}
                    fill="url(#priceGradient)"
                    dot={false}
                  />
                  
                  {/* Advanced Analytics Overlays */}
                  <Line 
                    yAxisId="indicators"
                    type="monotone" 
                    dataKey="godmode" 
                    stroke="#8B5CF6" 
                    strokeWidth={2} 
                    dot={false}
                    strokeDasharray="5 5"
                  />
                  <Line 
                    yAxisId="indicators"
                    type="monotone" 
                    dataKey="whale" 
                    stroke="#F59E0B" 
                    strokeWidth={1.5} 
                    dot={false}
                  />
                  <Line 
                    yAxisId="indicators"
                    type="monotone" 
                    dataKey="quantum" 
                    stroke="#10B981" 
                    strokeWidth={1.5} 
                    dot={false}
                    strokeDasharray="3 3"
                  />
                  <Line 
                    yAxisId="indicators"
                    type="monotone" 
                    dataKey="rsi" 
                    stroke="#F472B6" 
                    strokeWidth={1} 
                    dot={false}
                    opacity={0.7}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            {/* Legend */}
            <div className="flex flex-wrap items-center justify-center space-x-6 mt-4 text-xs">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-slate-400">Price</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-1 bg-purple-500 rounded"></div>
                <span className="text-slate-400">Godmode Factor</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-1 bg-amber-500 rounded"></div>
                <span className="text-slate-400">Whale Presence</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-1 bg-emerald-500 rounded border-dashed border border-emerald-500"></div>
                <span className="text-slate-400">Quantum Coherence</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-1 bg-pink-500 rounded opacity-70"></div>
                <span className="text-slate-400">RSI</span>
              </div>
            </div>
          </div>
          
          {/* Quick Stats Bar */}
          <div className="mt-6 grid grid-cols-4 gap-4">
            {currentFrame && [
              {
                label: 'Market Phase',
                value: currentFrame.market_layers.market_phase,
                icon: Target,
                color: currentFrame.market_layers.market_phase === 'markup' ? 'text-green-400' : 
                       currentFrame.market_layers.market_phase === 'markdown' ? 'text-red-400' : 'text-yellow-400'
              },
              {
                label: 'Volatility Regime',
                value: currentFrame.market_layers.volatility_regime,
                icon: Activity,
                color: currentFrame.market_layers.volatility_regime === 'extreme' ? 'text-red-400' : 
                       currentFrame.market_layers.volatility_regime === 'high' ? 'text-orange-400' : 'text-green-400'
              },
              {
                label: 'Momentum Vector',
                value: currentFrame.intention_field.momentum_vector.direction,
                icon: currentFrame.intention_field.momentum_vector.direction === 'up' ? TrendingUp : 
                      currentFrame.intention_field.momentum_vector.direction === 'down' ? TrendingDown : Minus,
                color: currentFrame.intention_field.momentum_vector.direction === 'up' ? 'text-green-400' : 
                       currentFrame.intention_field.momentum_vector.direction === 'down' ? 'text-red-400' : 'text-gray-400'
              },
              {
                label: 'Certainty Level',
                value: `${(currentFrame.temporal_ghost.certainty_river * 100).toFixed(0)}%`,
                icon: Shield,
                color: currentFrame.temporal_ghost.certainty_river > 0.7 ? 'text-green-400' : 
                       currentFrame.temporal_ghost.certainty_river > 0.4 ? 'text-yellow-400' : 'text-red-400'
              }
            ].map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-slate-400 text-sm">{stat.label}</p>
                      <p className={`font-semibold capitalize ${stat.color}`}>{stat.value}</p>
                    </div>
                    <Icon className={`w-5 h-5 ${stat.color}`} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Sidebar: Advanced analytics */}
        <div className="col-span-4">
          {renderSidebar()}
        </div>
      </div>

      {/* Advanced Analytics Bottom Panel */}
      {selectedView === 'quantum' && currentFrame && (
        <div className="mt-8 bg-gradient-to-br from-purple-900/10 to-blue-900/10 border border-purple-500/20 rounded-xl p-6">
          <h3 className="text-xl font-semibold mb-6 flex items-center space-x-2">
            <Brain className="w-6 h-6 text-purple-400" />
            <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Temporal Ghost Predictions
            </span>
          </h3>
          
          <div className="grid grid-cols-5 gap-4">
            {currentFrame.temporal_ghost.next_moves.map((move, index) => (
              <div key={index} className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-4 backdrop-blur-sm">
                <div className="text-center">
                  <div className="text-slate-400 text-sm mb-2">
                    T+{index + 1}min
                  </div>
                  <div className="font-bold text-lg mb-2">
                    ${move.predicted_price.toFixed(2)}
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Confidence:</span>
                      <span className={`font-medium ${
                        move.confidence > 0.8 ? 'text-green-400' :
                        move.confidence > 0.6 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {(move.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Probability:</span>
                      <span className="font-medium text-blue-400">
                        {(move.probability * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Pattern Match:</span>
                      <span className="font-medium text-purple-400">
                        {(move.pattern_match * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default HybridTradingInterface;