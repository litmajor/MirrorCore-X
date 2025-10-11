
import React, { useState, useEffect } from 'react';
import { useAPI } from '../hooks/useAPI';
import { Activity, AlertCircle, CheckCircle, XCircle, Cpu, Database, Terminal } from 'lucide-react';

const AgentMonitor: React.FC = () => {
  const { data: agentData, refetch } = useAPI('/api/agents/states');
  const { data: logsData } = useAPI('/api/agents/logs');

  useEffect(() => {
    const interval = setInterval(() => {
      refetch();
    }, 3000); // Refresh every 3 seconds

    return () => clearInterval(interval);
  }, [refetch]);

  const getHealthColor = (score: number) => {
    if (score > 0.8) return 'text-success';
    if (score > 0.5) return 'text-warning';
    return 'text-error';
  };

  const getHealthIcon = (score: number) => {
    if (score > 0.8) return <CheckCircle className="w-4 h-4 text-success" />;
    if (score > 0.5) return <AlertCircle className="w-4 h-4 text-warning" />;
    return <XCircle className="w-4 h-4 text-error" />;
  };

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Agent Monitor</h1>
        <p className="page-subtitle">Real-time agent states and system logs</p>
      </div>

      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Total Agents</h3>
            <Cpu className="w-5 h-5 text-brand-cyan" />
          </div>
          <p className="text-2xl font-bold text-white">
            {Object.keys(agentData?.agents || {}).length}
          </p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Healthy Agents</h3>
            <CheckCircle className="w-5 h-5 text-success" />
          </div>
          <p className="text-2xl font-bold text-success">
            {Object.values(agentData?.agents || {}).filter((a: any) => a.health.health_score > 0.8).length}
          </p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">System Ticks</h3>
            <Activity className="w-5 h-5 text-brand-purple" />
          </div>
          <p className="text-2xl font-bold text-white">
            {agentData?.tick_count || 0}
          </p>
        </div>
      </div>

      {/* Agent States Grid */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4">Agent States</h3>
        <div className="space-y-3">
          {Object.entries(agentData?.agents || {}).map(([agentId, agentInfo]: [string, any]) => (
            <div key={agentId} className="bg-bg-tertiary rounded-lg p-4 border border-brand-cyan/10">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  {getHealthIcon(agentInfo.health.health_score)}
                  <h4 className="font-semibold text-white">{agentId}</h4>
                </div>
                <div className="flex items-center space-x-4">
                  <span className={`text-sm ${getHealthColor(agentInfo.health.health_score)}`}>
                    Health: {(agentInfo.health.health_score * 100).toFixed(1)}%
                  </span>
                  {agentInfo.circuit_breaker.is_open && (
                    <span className="px-2 py-1 bg-error/20 text-error text-xs rounded">
                      Circuit Open
                    </span>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 text-sm mt-3">
                <div>
                  <span className="text-txt-secondary">Success:</span>
                  <span className="ml-2 text-success">{agentInfo.health.success_count}</span>
                </div>
                <div>
                  <span className="text-txt-secondary">Failures:</span>
                  <span className="ml-2 text-error">{agentInfo.health.failure_count}</span>
                </div>
                <div>
                  <span className="text-txt-secondary">Status:</span>
                  <span className="ml-2 text-white">{agentInfo.state.status || 'unknown'}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Logs */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
          <Terminal className="w-5 h-5" />
          <span>Recent System Logs</span>
        </h3>
        <div className="bg-bg-tertiary rounded-lg p-4 max-h-96 overflow-y-auto">
          <div className="space-y-2 font-mono text-xs">
            {logsData?.logs && logsData.logs.length > 0 ? (
              logsData.logs.slice().reverse().map((log: any, idx: number) => (
                <div key={idx} className="flex items-start space-x-3 pb-2 border-b border-brand-cyan/10">
                  <span className="text-txt-secondary whitespace-nowrap">
                    {new Date(log.timestamp * 1000).toLocaleTimeString()}
                  </span>
                  <span className={`font-semibold ${
                    log.event_type === 'error' ? 'text-error' :
                    log.event_type === 'order_filled' ? 'text-success' :
                    'text-brand-cyan'
                  }`}>
                    {log.event_type}
                  </span>
                  <span className="text-txt-secondary flex-1">
                    {log.symbol || 'system'}: {JSON.stringify(log.data).substring(0, 80)}...
                  </span>
                </div>
              ))
            ) : (
              <div className="text-txt-secondary text-center py-4">
                No logs available
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentMonitor;
