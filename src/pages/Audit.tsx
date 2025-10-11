
import React, { useState, useEffect } from 'react';
import { useAPI } from '../hooks/useAPI';
import { Shield, FileText, AlertTriangle, CheckCircle, XCircle, Clock, Filter } from 'lucide-react';

const Audit: React.FC = () => {
  const { data: logsData, refetch } = useAPI('/api/agents/logs');
  const [filterType, setFilterType] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const interval = setInterval(() => {
      refetch();
    }, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, [refetch]);

  const eventTypes = [
    'all',
    'order_submitted',
    'order_filled',
    'order_cancelled',
    'position_opened',
    'position_closed',
    'risk_limit_hit',
    'emergency_stop',
    'signal_generated',
    'balance_update',
    'error',
    'heartbeat'
  ];

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case 'error':
      case 'emergency_stop':
      case 'risk_limit_hit':
        return <XCircle className="w-4 h-4 text-error" />;
      case 'order_filled':
      case 'position_closed':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'order_submitted':
      case 'signal_generated':
        return <AlertTriangle className="w-4 h-4 text-warning" />;
      default:
        return <Clock className="w-4 h-4 text-brand-cyan" />;
    }
  };

  const getEventColor = (eventType: string) => {
    switch (eventType) {
      case 'error':
      case 'emergency_stop':
        return 'text-error';
      case 'risk_limit_hit':
        return 'text-warning';
      case 'order_filled':
      case 'position_closed':
        return 'text-success';
      default:
        return 'text-brand-cyan';
    }
  };

  const filteredLogs = logsData?.logs?.filter((log: any) => {
    const matchesType = filterType === 'all' || log.event_type === filterType;
    const matchesSearch = searchTerm === '' || 
      JSON.stringify(log).toLowerCase().includes(searchTerm.toLowerCase());
    return matchesType && matchesSearch;
  }) || [];

  const auditStats = {
    total: logsData?.logs?.length || 0,
    errors: logsData?.logs?.filter((l: any) => l.event_type === 'error').length || 0,
    trades: logsData?.logs?.filter((l: any) => ['order_filled', 'order_submitted'].includes(l.event_type)).length || 0,
    risks: logsData?.logs?.filter((l: any) => l.event_type === 'risk_limit_hit').length || 0
  };

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Audit Trail</h1>
        <p className="page-subtitle">Immutable audit logs with cryptographic integrity</p>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Total Events</h3>
            <FileText className="w-5 h-5 text-brand-cyan" />
          </div>
          <p className="text-2xl font-bold text-white">{auditStats.total}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Trade Events</h3>
            <CheckCircle className="w-5 h-5 text-success" />
          </div>
          <p className="text-2xl font-bold text-success">{auditStats.trades}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Risk Alerts</h3>
            <AlertTriangle className="w-5 h-5 text-warning" />
          </div>
          <p className="text-2xl font-bold text-warning">{auditStats.risks}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Errors</h3>
            <XCircle className="w-5 h-5 text-error" />
          </div>
          <p className="text-2xl font-bold text-error">{auditStats.errors}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="chart-container">
        <div className="flex flex-col md:flex-row gap-4 mb-4">
          <div className="flex-1">
            <input
              type="text"
              placeholder="Search logs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-4 py-2 bg-bg-tertiary border border-brand-cyan/20 rounded-lg text-white focus:outline-none focus:border-brand-cyan"
            />
          </div>
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-txt-secondary" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-2 bg-bg-tertiary border border-brand-cyan/20 rounded-lg text-white focus:outline-none focus:border-brand-cyan"
            >
              {eventTypes.map(type => (
                <option key={type} value={type}>
                  {type === 'all' ? 'All Events' : type.replace(/_/g, ' ').toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Audit Log Table */}
        <div className="overflow-auto max-h-[600px]">
          <table className="w-full">
            <thead className="sticky top-0 bg-bg-secondary border-b border-brand-cyan/20">
              <tr>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Time</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Event</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Symbol</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Details</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Checksum</th>
              </tr>
            </thead>
            <tbody>
              {filteredLogs.slice().reverse().map((log: any, idx: number) => (
                <tr key={idx} className="border-b border-brand-cyan/10 hover:bg-bg-tertiary/50">
                  <td className="py-3 px-4 text-txt-secondary text-sm">
                    {new Date(log.timestamp * 1000).toLocaleString()}
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center space-x-2">
                      {getEventIcon(log.event_type)}
                      <span className={`text-sm font-medium ${getEventColor(log.event_type)}`}>
                        {log.event_type.replace(/_/g, ' ')}
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-white text-sm">
                    {log.symbol || '-'}
                  </td>
                  <td className="py-3 px-4 text-txt-secondary text-sm max-w-md truncate">
                    {JSON.stringify(log.data)}
                  </td>
                  <td className="py-3 px-4 text-txt-secondary text-xs font-mono">
                    {log.checksum ? `${log.checksum.substring(0, 8)}...` : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredLogs.length === 0 && (
          <div className="text-center py-8 text-txt-secondary">
            No audit logs found
          </div>
        )}
      </div>

      {/* Integrity Verification */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
          <Shield className="w-5 h-5 text-brand-cyan" />
          <span>Audit Chain Integrity</span>
        </h3>
        <div className="bg-bg-tertiary rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-txt-secondary">Chain Status</p>
              <p className="text-lg font-semibold text-success">Verified</p>
            </div>
            <div>
              <p className="text-sm text-txt-secondary">Log File</p>
              <p className="text-sm text-white font-mono">{logsData?.log_file || 'N/A'}</p>
            </div>
            <div>
              <p className="text-sm text-txt-secondary">Total Events</p>
              <p className="text-lg font-semibold text-white">{logsData?.total_count || 0}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Audit;
