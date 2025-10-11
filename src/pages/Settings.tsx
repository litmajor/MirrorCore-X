
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
      <h1 className="text-3xl font-bold text-white">Settings</h1>

      <div className="space-y-6">
        {/* General Settings */}
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700 rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-6">
            <SettingsIcon className="w-6 h-6 text-blue-400" />
            <h3 className="text-xl font-semibold text-white">General</h3>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">Enable Notifications</p>
                <p className="text-slate-400 text-sm">Receive alerts for trades and signals</p>
              </div>
              <input
                type="checkbox"
                checked={settings.notifications}
                onChange={(e) => setSettings({...settings, notifications: e.target.checked})}
                className="w-12 h-6 bg-slate-700 rounded-full relative cursor-pointer"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">Auto Trading</p>
                <p className="text-slate-400 text-sm">Automatically execute high-confidence signals</p>
              </div>
              <input
                type="checkbox"
                checked={settings.autoTrading}
                onChange={(e) => setSettings({...settings, autoTrading: e.target.checked})}
                className="w-12 h-6 bg-slate-700 rounded-full relative cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Risk Settings */}
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700 rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-6">
            <Shield className="w-6 h-6 text-purple-400" />
            <h3 className="text-xl font-semibold text-white">Risk Management</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="text-white font-medium block mb-2">Risk Level</label>
              <select
                value={settings.riskLevel}
                onChange={(e) => setSettings({...settings, riskLevel: e.target.value})}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white"
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
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white"
              />
            </div>
          </div>
        </div>

        {/* API Configuration */}
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700 rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-6">
            <Database className="w-6 h-6 text-green-400" />
            <h3 className="text-xl font-semibold text-white">API Configuration</h3>
          </div>
          
          <p className="text-slate-400 mb-4">Configure API keys and exchange connections via environment variables</p>
          <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium transition-colors">
            Manage API Keys
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
