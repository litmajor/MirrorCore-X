
import React, { useState } from 'react';
import { useAPI } from '../hooks/useAPI';
import { TrendingUp, TrendingDown, Activity, BarChart3, Layers, Target, AlertCircle } from 'lucide-react';
import { MetricCardSkeleton, CardSkeleton } from '../components/Skeleton';

const TechnicalAnalysis: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const { data: scannerData, loading: scanLoading, error: scanError } = useAPI('/api/scanner/realtime');
  const { data: technicalData, loading: techLoading, error: techError } = useAPI(`/api/technical/analysis/${selectedSymbol.replace('/', '_')}`);

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Technical Analysis</h1>
        <p className="page-subtitle">Comprehensive market indicators and signals</p>
      </div>

      {/* Symbol Selector */}
      {scanLoading ? (
        <CardSkeleton />
      ) : scanError || !scannerData?.data || scannerData.data.length === 0 ? (
        <div className="brand-card text-center py-12">
          <AlertCircle className="w-16 h-16 text-txt-secondary mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">Scanner Data Unavailable</h3>
          <p className="text-txt-secondary">Start the backend scanner to view technical analysis.</p>
        </div>
      ) : (
        <>
          <div className="bg-bg-secondary rounded-lg p-4">
            <select 
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="bg-bg-tertiary text-white rounded px-4 py-2 border border-brand-cyan/30"
            >
              {scannerData.data.map((item: any) => (
                <option key={item.symbol} value={item.symbol}>{item.symbol}</option>
              ))}
            </select>
          </div>

          {techLoading ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
              </div>
              <CardSkeleton />
              <CardSkeleton />
            </div>
          ) : technicalData && !technicalData.error ? (
        <>
          {/* Price & Signal Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="metric-card">
              <h3 className="text-txt-secondary text-sm mb-2">Price</h3>
              <p className="text-2xl font-bold text-white">${technicalData.price?.toFixed(2)}</p>
            </div>
            <div className="metric-card">
              <h3 className="text-txt-secondary text-sm mb-2">Signal</h3>
              <p className={`text-xl font-bold ${
                technicalData.signal?.includes('Buy') ? 'text-success' : 
                technicalData.signal?.includes('Sell') ? 'text-error' : 'text-warning'
              }`}>{technicalData.signal}</p>
            </div>
            <div className="metric-card">
              <h3 className="text-txt-secondary text-sm mb-2">Composite Score</h3>
              <p className="text-2xl font-bold text-brand-cyan">{technicalData.composite_score?.toFixed(1)}</p>
            </div>
            <div className="metric-card">
              <h3 className="text-txt-secondary text-sm mb-2">Confidence</h3>
              <p className="text-2xl font-bold text-brand-purple">{(technicalData.advanced?.confidence_score * 100)?.toFixed(0)}%</p>
            </div>
          </div>

          {/* Technical Indicators */}
          <div className="chart-container">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-brand-cyan" />
              Technical Indicators
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-txt-secondary text-sm">RSI</p>
                <p className={`text-xl font-bold ${
                  technicalData.indicators?.rsi > 70 ? 'text-error' :
                  technicalData.indicators?.rsi < 30 ? 'text-success' : 'text-white'
                }`}>{technicalData.indicators?.rsi?.toFixed(1)}</p>
              </div>
              <div>
                <p className="text-txt-secondary text-sm">MACD</p>
                <p className="text-xl font-bold text-white">{technicalData.indicators?.macd?.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-txt-secondary text-sm">ATR</p>
                <p className="text-xl font-bold text-white">{technicalData.indicators?.atr?.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-txt-secondary text-sm">VWAP</p>
                <p className="text-xl font-bold text-white">${technicalData.indicators?.vwap?.toFixed(2)}</p>
              </div>
            </div>
          </div>

          {/* Bollinger Bands & EMAs */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="chart-container">
              <h3 className="text-lg font-semibold text-white mb-4">Bollinger Bands</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-txt-secondary">Upper</span>
                  <span className="text-error font-bold">${technicalData.indicators?.bb_upper?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">Middle</span>
                  <span className="text-white font-bold">${technicalData.indicators?.bb_middle?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">Lower</span>
                  <span className="text-success font-bold">${technicalData.indicators?.bb_lower?.toFixed(2)}</span>
                </div>
              </div>
            </div>

            <div className="chart-container">
              <h3 className="text-lg font-semibold text-white mb-4">Moving Averages</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-txt-secondary">EMA 5</span>
                  <span className="text-white font-bold">${technicalData.indicators?.ema_5?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">EMA 13</span>
                  <span className="text-white font-bold">${technicalData.indicators?.ema_13?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">EMA 20</span>
                  <span className="text-white font-bold">${technicalData.indicators?.ema_20?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">EMA 50</span>
                  <span className="text-white font-bold">${technicalData.indicators?.ema_50?.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Momentum & Volume */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="chart-container">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-success" />
                Momentum Analysis
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-txt-secondary">Short-term</span>
                  <span className="text-white font-bold">{(technicalData.momentum?.momentum_short * 100)?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">7-day</span>
                  <span className="text-white font-bold">{(technicalData.momentum?.momentum_7d * 100)?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">30-day</span>
                  <span className="text-white font-bold">{(technicalData.momentum?.momentum_30d * 100)?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">Trend Score</span>
                  <span className="text-brand-cyan font-bold">{technicalData.momentum?.trend_score?.toFixed(1)}</span>
                </div>
              </div>
            </div>

            <div className="chart-container">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-brand-purple" />
                Volume Analysis
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-txt-secondary">Current Volume</span>
                  <span className="text-white font-bold">{technicalData.volume?.current?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">Volume Ratio</span>
                  <span className="text-white font-bold">{technicalData.volume?.ratio?.toFixed(2)}x</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">Composite Score</span>
                  <span className="text-brand-cyan font-bold">{technicalData.volume?.composite_score?.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-txt-secondary">POC Distance</span>
                  <span className="text-white font-bold">{(technicalData.volume?.poc_distance * 100)?.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Advanced Patterns */}
          <div className="chart-container">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Layers className="w-5 h-5 mr-2 text-warning" />
              Advanced Pattern Recognition
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-txt-secondary text-sm mb-1">Ichimoku</p>
                <div className={`inline-block px-3 py-1 rounded ${technicalData.patterns?.ichimoku_bullish ? 'bg-success/20 text-success' : 'bg-error/20 text-error'}`}>
                  {technicalData.patterns?.ichimoku_bullish ? 'Bullish' : 'Bearish'}
                </div>
              </div>
              <div className="text-center">
                <p className="text-txt-secondary text-sm mb-1">VWAP</p>
                <div className={`inline-block px-3 py-1 rounded ${technicalData.patterns?.vwap_bullish ? 'bg-success/20 text-success' : 'bg-error/20 text-error'}`}>
                  {technicalData.patterns?.vwap_bullish ? 'Above' : 'Below'}
                </div>
              </div>
              <div className="text-center">
                <p className="text-txt-secondary text-sm mb-1">EMA Cross</p>
                <div className={`inline-block px-3 py-1 rounded ${technicalData.patterns?.ema_crossover ? 'bg-success/20 text-success' : 'bg-error/20 text-error'}`}>
                  {technicalData.patterns?.ema_crossover ? 'Golden' : 'Death'}
                </div>
              </div>
              <div className="text-center">
                <p className="text-txt-secondary text-sm mb-1">Fib Confluence</p>
                <p className="text-xl font-bold text-brand-cyan">{technicalData.patterns?.fib_confluence?.toFixed(0)}</p>
              </div>
            </div>
          </div>

          {/* Advanced Analytics */}
          <div className="chart-container">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2 text-brand-cyan" />
              Advanced Analytics
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-txt-secondary text-sm mb-1">Cluster Validated</p>
                <p className={`text-lg font-bold ${technicalData.advanced?.cluster_validated ? 'text-success' : 'text-txt-secondary'}`}>
                  {technicalData.advanced?.cluster_validated ? 'Yes' : 'No'}
                </p>
              </div>
              <div>
                <p className="text-txt-secondary text-sm mb-1">Reversion Probability</p>
                <p className="text-lg font-bold text-white">{(technicalData.advanced?.reversion_probability * 100)?.toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-txt-secondary text-sm mb-1">Market Regime</p>
                <p className="text-lg font-bold text-brand-purple">{technicalData.advanced?.regime}</p>
              </div>
            </div>
          </div>
        </>
          ) : null}
        </>
      )}
    </div>
  );
};

export default TechnicalAnalysis;
