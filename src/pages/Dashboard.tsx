
import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity, DollarSign, Percent, AlertTriangle } from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAPI } from '../hooks/useAPI';

const Dashboard: React.FC = () => {
  const { data: wsData, isConnected } = useWebSocket('ws://0.0.0.0:5000');
  const { data: marketData } = useAPI('/api/market/overview');
  const { data: performance } = useAPI('/api/performance/summary');
  
  const [metrics, setMetrics] = useState({
    totalPnL: 0,
    winRate: 0,
    activeSignals: 0,
    systemHealth: 100
  });

  useEffect(() => {
    if (wsData?.system_performance) {
      setMetrics({
        totalPnL: wsData.system_performance.pnl || 0,
        winRate: wsData.system_performance.win_rate || 0,
        activeSignals: wsData.scanner_data?.length || 0,
        systemHealth: wsData.system_health?.health_score * 100 || 100
      });
    }
  }, [wsData]);

  const MetricCard = ({ title, value, change, icon: Icon, trend }: any) => (
    <div className="metric-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-txt-secondary text-sm font-medium">{title}</h3>
        <Icon className="w-5 h-5 text-brand-cyan" />
      </div>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-3xl font-bold text-white">{value}</p>
          {change !== undefined && (
            <div className={`flex items-center mt-2 text-sm ${trend === 'up' ? 'text-success' : 'text-error'}`}>
              {trend === 'up' ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              <span>{Math.abs(change).toFixed(2)}%</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between page-header">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">Real-time trading system overview</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={isConnected ? 'status-active' : 'status-inactive'} />
          <span className="text-sm text-txt-secondary">{isConnected ? 'Live' : 'Disconnected'}</span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total P&L"
          value={`$${metrics.totalPnL.toFixed(2)}`}
          change={5.2}
          trend="up"
          icon={DollarSign}
        />
        <MetricCard
          title="Win Rate"
          value={`${metrics.winRate.toFixed(1)}%`}
          change={2.1}
          trend="up"
          icon={Percent}
        />
        <MetricCard
          title="Active Signals"
          value={metrics.activeSignals}
          icon={Activity}
        />
        <MetricCard
          title="System Health"
          value={`${metrics.systemHealth.toFixed(0)}%`}
          change={-0.5}
          trend="down"
          icon={AlertTriangle}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Price Chart */}
        <div className="chart-container">
          <h3 className="text-lg font-semibold text-white mb-4">Market Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={wsData?.market_data?.slice(-50) || []}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="timestamp" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="price" 
                stroke="#3b82f6" 
                fillOpacity={1} 
                fill="url(#colorPrice)" 
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Strategy Performance */}
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Strategy Performance</h3>
          <div className="space-y-4">
            {Object.entries(wsData?.strategy_grades || {}).map(([name, grade]: [string, any]) => (
              <div key={name} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${grade === 'A' ? 'bg-green-400' : grade === 'B' ? 'bg-yellow-400' : 'bg-red-400'}`} />
                  <span className="text-slate-300">{name}</span>
                </div>
                <span className={`font-bold ${grade === 'A' ? 'text-green-400' : grade === 'B' ? 'text-yellow-400' : 'text-red-400'}`}>
                  {grade}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
