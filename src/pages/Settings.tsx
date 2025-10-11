
import React, { useState } from 'react';
import { Settings as SettingsIcon, Bell, Shield, Database, Zap } from 'lucide-react';

const Settings: React.FC = () => {
  const [settings, setSettings] = useState({
    notifications: true,
    autoTrading: false,
    riskLevel: 'medium',
    maxPositions: 5,
  });

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
        <p className="page-subtitle">Configure system preferences and parameters</p>
      </div>

      <div className="space-y-6">
        {/* General Settings */}
        <div className="brand-card">
          <div className="flex items-center space-x-3 mb-6">
            <SettingsIcon className="w-6 h-6 text-brand-cyan" />
            <h3 className="text-xl font-semibold text-white">General</h3>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">Enable Notifications</p>
                <p className="text-txt-secondary text-sm">Receive alerts for trades and signals</p>
              </div>
              <input
                type="checkbox"
                checked={settings.notifications}
                onChange={(e) => setSettings({...settings, notifications: e.target.checked})}
                className="w-12 h-6 bg-bg-surface rounded-full relative cursor-pointer"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">Auto Trading</p>
                <p className="text-txt-secondary text-sm">Automatically execute high-confidence signals</p>
              </div>
              <input
                type="checkbox"
                checked={settings.autoTrading}
                onChange={(e) => setSettings({...settings, autoTrading: e.target.checked})}
                className="w-12 h-6 bg-bg-surface rounded-full relative cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Risk Settings */}
        <div className="brand-card">
          <div className="flex items-center space-x-3 mb-6">
            <Shield className="w-6 h-6 text-brand-purple" />
            <h3 className="text-xl font-semibold text-white">Risk Management</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="text-white font-medium block mb-2">Risk Level</label>
              <select
                value={settings.riskLevel}
                onChange={(e) => setSettings({...settings, riskLevel: e.target.value})}
                className="w-full bg-bg-tertiary border border-brand-cyan/20 rounded-lg px-4 py-2 text-white focus:border-brand-cyan focus:outline-none"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>

            <div>
              <label className="text-white font-medium block mb-2">Max Concurrent Positions</label>
              <input
                type="number"
                value={settings.maxPositions}
                onChange={(e) => setSettings({...settings, maxPositions: parseInt(e.target.value)})}
                className="w-full bg-bg-tertiary border border-brand-cyan/20 rounded-lg px-4 py-2 text-white focus:border-brand-cyan focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* API Configuration */}
        <div className="brand-card">
          <div className="flex items-center space-x-3 mb-6">
            <Database className="w-6 h-6 text-success" />
            <h3 className="text-xl font-semibold text-white">API Configuration</h3>
          </div>
          
          <p className="text-txt-secondary mb-4">Configure API keys and exchange connections via environment variables</p>
          <button className="btn-primary">
            Manage API Keys
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
