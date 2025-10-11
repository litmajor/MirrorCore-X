
import React from 'react';
import { useAPI } from '../hooks/useAPI';
import { Brain, Activity, TrendingUp, Target, Zap, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const RLAgent: React.FC = () => {
  const { data: rlStatus } = useAPI('/api/rl/status');
  const { data: performance } = useAPI('/api/performance/summary');

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">RL Trading Agent</h1>
        <p className="page-subtitle">Reinforcement Learning model status and performance</p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Agent Status</h3>
            <Brain className="w-5 h-5 text-brand-purple" />
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${rlStatus?.is_trained ? 'bg-success' : 'bg-error'}`} />
            <p className="text-xl font-bold text-white">
              {rlStatus?.is_trained ? 'Active' : 'Not Trained'}
            </p>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Algorithm</h3>
            <Zap className="w-5 h-5 text-brand-cyan" />
          </div>
          <p className="text-xl font-bold text-white">{rlStatus?.algorithm || 'PPO'}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Confidence</h3>
            <Target className="w-5 h-5 text-success" />
          </div>
          <p className="text-xl font-bold text-white">
            {((rlStatus?.confidence || 0) * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Model Info */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4">Model Information</h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-txt-secondary">Model Path:</span>
            <span className="text-white font-mono text-sm">{rlStatus?.model_path || 'N/A'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-txt-secondary">Total Predictions:</span>
            <span className="text-white">{rlStatus?.total_predictions || 0}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-txt-secondary">Training Episodes:</span>
            <span className="text-white">100,000</span>
          </div>
        </div>
      </div>

      {/* Training Metrics */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4">RL Performance Metrics</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-txt-secondary text-sm">Sharpe Ratio</p>
            <p className="text-2xl font-bold text-success">{performance?.sharpe_ratio?.toFixed(2) || '0.00'}</p>
          </div>
          <div>
            <p className="text-txt-secondary text-sm">Win Rate</p>
            <p className="text-2xl font-bold text-brand-cyan">{performance?.win_rate?.toFixed(1) || '0'}%</p>
          </div>
          <div>
            <p className="text-txt-secondary text-sm">Profit Factor</p>
            <p className="text-2xl font-bold text-brand-purple">{performance?.profit_factor?.toFixed(2) || '0.00'}</p>
          </div>
          <div>
            <p className="text-txt-secondary text-sm">Max Drawdown</p>
            <p className="text-2xl font-bold text-error">{performance?.max_drawdown?.toFixed(1) || '0'}%</p>
          </div>
        </div>
      </div>

      {/* Recent Actions */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Actions</h3>
        {rlStatus?.recent_actions && rlStatus.recent_actions.length > 0 ? (
          <div className="space-y-2">
            {rlStatus.recent_actions.map((action: any, idx: number) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-bg-tertiary rounded">
                <span className="text-txt-secondary">{action.timestamp}</span>
                <span className={`font-bold ${action.position > 0 ? 'text-success' : 'text-error'}`}>
                  {action.position > 0 ? 'LONG' : 'SHORT'} {Math.abs(action.position).toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex items-center justify-center p-8 text-txt-secondary">
            <AlertCircle className="w-5 h-5 mr-2" />
            No recent actions available
          </div>
        )}
      </div>
    </div>
  );
};

export default RLAgent;
