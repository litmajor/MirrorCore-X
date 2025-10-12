
import React, { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, Users, Lock } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';

const VaultManagement: React.FC = () => {
  const { data: vaultData, loading, error } = useAPI('/api/vault/metrics');

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-white">Vault Management</h2>

      {/* Vault Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm">Total AUM</h3>
            <DollarSign className="w-5 h-5 text-success" />
          </div>
          <p className="text-3xl font-bold text-white">${vaultData?.aum?.toLocaleString() || '0'}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm">Share Price</h3>
            <TrendingUp className="w-5 h-5 text-brand-cyan" />
          </div>
          <p className="text-3xl font-bold text-white">${vaultData?.sharePrice?.toFixed(4) || '0.0000'}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm">Total Depositors</h3>
            <Users className="w-5 h-5 text-brand-purple" />
          </div>
          <p className="text-3xl font-bold text-white">{vaultData?.depositors || '0'}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm">TVL Locked</h3>
            <Lock className="w-5 h-5 text-warning" />
          </div>
          <p className="text-3xl font-bold text-white">${vaultData?.tvlLocked?.toLocaleString() || '0'}</p>
        </div>
      </div>

      {/* Deposit/Withdraw Interface */}
      <div className="brand-card">
        <h3 className="text-xl font-semibold text-white mb-6">Manage Position</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="text-white font-medium block mb-2">Deposit Amount</label>
            <input
              type="number"
              placeholder="0.00"
              className="w-full bg-bg-tertiary border border-brand-cyan/20 rounded-lg px-4 py-3 text-white"
            />
            <button className="w-full mt-4 bg-success hover:bg-success/80 text-white font-bold py-3 rounded-lg transition-colors">
              Deposit
            </button>
          </div>
          <div>
            <label className="text-white font-medium block mb-2">Withdraw Shares</label>
            <input
              type="number"
              placeholder="0.00"
              className="w-full bg-bg-tertiary border border-brand-cyan/20 rounded-lg px-4 py-3 text-white"
            />
            <button className="w-full mt-4 bg-error hover:bg-error/80 text-white font-bold py-3 rounded-lg transition-colors">
              Withdraw
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VaultManagement;
