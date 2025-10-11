
import React, { useState, useEffect } from 'react';
import { Brain, Sparkles, TrendingUp, Shield, AlertTriangle } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';

const Oracle: React.FC = () => {
  const { data: oracleData } = useAPI('/api/oracle/directives');
  const { data: imagination } = useAPI('/api/imagination/analysis');
  const { data: bayesian } = useAPI('/api/bayesian/beliefs');

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Oracle & Imagination Engine</h1>
        <p className="page-subtitle">Advanced AI-driven trading insights and counterfactual analysis</p>
      </div>

      {/* Oracle Directives */}
      <div className="brand-card">
        <div className="flex items-center space-x-3 mb-6">
          <Brain className="w-6 h-6 text-brand-purple" />
          <h2 className="text-xl font-semibold text-white">Oracle Trading Directives</h2>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {oracleData?.directives?.map((directive: any, idx: number) => (
            <div key={idx} className="p-4 bg-bg-surface rounded-lg border border-accent-dark">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="text-lg font-semibold text-white">{directive.symbol}</h3>
                  <p className="text-sm text-txt-secondary">Strategy: {directive.strategy}</p>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  directive.action === 'buy' ? 'bg-success/20 text-success' : 
                  directive.action === 'sell' ? 'bg-error/20 text-error' : 
                  'bg-neutral/20 text-neutral'
                }`}>
                  {directive.action.toUpperCase()}
                </span>
              </div>

              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-txt-secondary">Confidence</p>
                  <p className="text-white font-semibold">{(directive.confidence * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-txt-secondary">Amount</p>
                  <p className="text-white font-semibold">{directive.amount?.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-txt-secondary">Price</p>
                  <p className="text-white font-semibold">${directive.price?.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-txt-secondary">Method</p>
                  <p className="text-white font-semibold">{directive.method}</p>
                </div>
              </div>

              {directive.enhanced_momentum_score && (
                <div className="mt-3 pt-3 border-t border-accent-dark">
                  <p className="text-xs text-txt-secondary">
                    Momentum: {directive.enhanced_momentum_score?.toFixed(3)} | 
                    Reversion: {(directive.reversion_probability * 100).toFixed(1)}% | 
                    Regime: {directive.volatility_regime}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Bayesian Beliefs */}
      <div className="brand-card">
        <div className="flex items-center space-x-3 mb-6">
          <Shield className="w-6 h-6 text-brand-cyan" />
          <h2 className="text-xl font-semibold text-white">Bayesian Strategy Beliefs</h2>
        </div>

        <div className="space-y-4">
          {bayesian?.top_strategies?.map((strategy: any, idx: number) => (
            <div key={idx} className="p-4 bg-bg-surface rounded-lg border border-accent-dark">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold text-white">{strategy.name}</h3>
                <span className="text-brand-cyan font-semibold">
                  {(strategy.probability * 100).toFixed(1)}%
                </span>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-txt-secondary">Confidence Interval</span>
                  <span className="text-white">
                    [{(strategy.confidence_interval[0] * 100).toFixed(1)}% - {(strategy.confidence_interval[1] * 100).toFixed(1)}%]
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-txt-secondary">Uncertainty</span>
                  <span className="text-white">{(strategy.uncertainty * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-txt-secondary">Sample Size</span>
                  <span className="text-white">{strategy.sample_size}</span>
                </div>
                {strategy.explanation && (
                  <p className="text-xs text-txt-secondary mt-2 pt-2 border-t border-accent-dark">
                    {strategy.explanation}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>

        {bayesian?.market_context && (
          <div className="mt-6 p-4 bg-bg-surface rounded-lg border border-accent-dark">
            <h3 className="text-sm font-semibold text-white mb-3">Market Context</h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-txt-secondary">Regime</p>
                <p className="text-white font-semibold">{bayesian.market_context.regime}</p>
              </div>
              <div>
                <p className="text-txt-secondary">Trend Strength</p>
                <p className="text-white font-semibold">{(bayesian.market_context.trend_strength * 100).toFixed(1)}%</p>
              </div>
              <div>
                <p className="text-txt-secondary">Volatility</p>
                <p className="text-white font-semibold">{(bayesian.market_context.volatility_percentile * 100).toFixed(1)}%</p>
              </div>
              <div>
                <p className="text-txt-secondary">Volume</p>
                <p className="text-white font-semibold">{(bayesian.market_context.volume_percentile * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Imagination Analysis */}
      <div className="brand-card">
        <div className="flex items-center space-x-3 mb-6">
          <Sparkles className="w-6 h-6 text-brand-purple" />
          <h2 className="text-xl font-semibold text-white">Imagination Engine Analysis</h2>
        </div>

        {imagination?.summary && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="metric-card">
                <p className="text-txt-secondary text-sm mb-1">Scenarios Tested</p>
                <p className="text-2xl font-bold text-white">{imagination.summary.scenarios_tested}</p>
              </div>
              <div className="metric-card">
                <p className="text-txt-secondary text-sm mb-1">Avg Robustness</p>
                <p className="text-2xl font-bold text-brand-cyan">{(imagination.summary.average_robustness * 100).toFixed(1)}%</p>
              </div>
              <div className="metric-card">
                <p className="text-txt-secondary text-sm mb-1">Best Strategy</p>
                <p className="text-lg font-bold text-success">{imagination.summary.best_strategy?.name}</p>
              </div>
              <div className="metric-card">
                <p className="text-txt-secondary text-sm mb-1">Vulnerabilities</p>
                <p className="text-2xl font-bold text-error">{imagination.summary.total_vulnerabilities}</p>
              </div>
            </div>

            {imagination.summary.recommendations && (
              <div className="p-4 bg-bg-surface rounded-lg border border-accent-dark">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center">
                  <AlertTriangle className="w-4 h-4 mr-2 text-warning" />
                  Recommendations
                </h3>
                <ul className="space-y-2">
                  {imagination.summary.recommendations.map((rec: string, idx: number) => (
                    <li key={idx} className="text-sm text-txt-secondary">• {rec}</li>
                  ))}
                </ul>
              </div>
            )}

            {imagination.vulnerabilities && (
              <div className="p-4 bg-bg-surface rounded-lg border border-accent-dark">
                <h3 className="text-sm font-semibold text-white mb-3">Strategy Vulnerabilities</h3>
                {Object.entries(imagination.vulnerabilities).map(([strategy, vulns]: [string, any]) => (
                  <div key={strategy} className="mb-3 last:mb-0">
                    <p className="font-semibold text-white mb-1">{strategy}</p>
                    <ul className="space-y-1">
                      {vulns.map((vuln: string, idx: number) => (
                        <li key={idx} className="text-xs text-error">• {vuln}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Oracle;
