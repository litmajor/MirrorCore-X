
import React from 'react';
import { Brain, Zap, Shield, TrendingUp, Sparkles, Activity, Target, Layers, MessageCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Landing: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: "Multi-Agent Intelligence",
      description: "19+ specialized trading strategies working in harmony, powered by Bayesian optimization and ensemble learning"
    },
    {
      icon: MessageCircle,
      title: "Mirrax AI Assistant",
      description: "Conversational AI that explains performance, analyzes markets, and provides actionable insights in real-time"
    },
    {
      icon: Sparkles,
      title: "Oracle & Imagination Engine",
      description: "Predictive AI that simulates 1000+ future market scenarios to stress-test every decision"
    },
    {
      icon: Shield,
      title: "Advanced Risk Management",
      description: "Real-time circuit breakers, flash crash protection, and dynamic position sizing"
    },
    {
      icon: TrendingUp,
      title: "Mathematical Optimization",
      description: "Quadratic programming with Ledoit-Wolf shrinkage for optimal portfolio allocation"
    },
    {
      icon: Zap,
      title: "Reinforcement Learning",
      description: "Self-improving RL agents trained on 100K+ episodes for adaptive execution"
    },
    {
      icon: Layers,
      title: "Crypto-Native Design",
      description: "24/7 trading, extreme volatility handling, on-chain metrics, and funding rate arbitrage"
    }
  ];

  const stats = [
    { value: "2.0-2.8", label: "Sharpe Ratio", suffix: "" },
    { value: "45-65", label: "Annual Return", suffix: "%" },
    { value: "15-25", label: "Max Drawdown", suffix: "%" },
    { value: "19+", label: "Active Strategies", suffix: "" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-bg-primary via-bg-secondary to-bg-primary">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Animated Background Grid */}
        <div className="absolute inset-0 opacity-20">
          <div className="absolute inset-0" style={{
            backgroundImage: `linear-gradient(#00D9FF 1px, transparent 1px), linear-gradient(90deg, #00D9FF 1px, transparent 1px)`,
            backgroundSize: '50px 50px'
          }} />
        </div>

        <div className="container mx-auto px-6 py-20 relative z-10">
          {/* Logo */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center space-x-4 mb-6">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-brand-cyan to-brand-purple flex items-center justify-center">
                <Brain className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-6xl font-bold bg-gradient-to-r from-brand-cyan to-brand-purple bg-clip-text text-transparent">
                MIRRORCORE-X
              </h1>
            </div>
            <p className="text-2xl text-txt-secondary font-light tracking-wide">
              COGNITIVE TRADING SYSTEM
            </p>
          </div>

          {/* Main Tagline */}
          <div className="text-center max-w-4xl mx-auto mb-12">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6 leading-tight">
              Next-Generation AI Trading
              <br />
              <span className="bg-gradient-to-r from-brand-cyan to-brand-purple bg-clip-text text-transparent">
                Built for Extreme Markets
              </span>
            </h2>
            <p className="text-xl text-txt-secondary leading-relaxed">
              Harness the power of 19 specialized trading agents, reinforcement learning, 
              and mathematical optimization to navigate crypto's 24/7 volatility with institutional-grade precision.
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-20">
            <button
              onClick={() => navigate('/dashboard')}
              className="px-8 py-4 bg-gradient-to-r from-brand-cyan to-brand-purple text-white text-lg font-semibold rounded-lg hover:shadow-2xl hover:shadow-brand-cyan/50 transform hover:scale-105 transition-all"
            >
              Launch Dashboard
            </button>
            <button
              onClick={() => navigate('/ai-insights')}
              className="px-8 py-4 bg-bg-surface border-2 border-brand-cyan text-brand-cyan text-lg font-semibold rounded-lg hover:bg-brand-cyan/10 transition-all"
            >
              <MessageCircle className="w-5 h-5 inline mr-2" />
              Try AI Assistant
            </button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-5xl mx-auto mb-20">
            {stats.map((stat, idx) => (
              <div key={idx} className="text-center p-6 bg-bg-surface/50 backdrop-blur-sm rounded-lg border border-accent-dark">
                <div className="text-4xl font-bold bg-gradient-to-r from-brand-cyan to-brand-purple bg-clip-text text-transparent mb-2">
                  {stat.value}{stat.suffix}
                </div>
                <div className="text-txt-secondary text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20 bg-bg-secondary/50">
        <div className="container mx-auto px-6">
          <h3 className="text-3xl font-bold text-center text-white mb-12">
            What Makes MirrorCore-X Different
          </h3>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, idx) => (
              <div key={idx} className="p-6 bg-bg-surface rounded-xl border border-accent-dark hover:border-brand-cyan transition-all group">
                <div className="w-12 h-12 bg-gradient-to-br from-brand-cyan to-brand-purple rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h4 className="text-xl font-semibold text-white mb-3">{feature.title}</h4>
                <p className="text-txt-secondary leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Architecture Overview */}
      <div className="py-20">
        <div className="container mx-auto px-6">
          <h3 className="text-3xl font-bold text-center text-white mb-12">
            Institutional-Grade Architecture
          </h3>

          <div className="max-w-5xl mx-auto">
            <div className="grid md:grid-cols-2 gap-8">
              <div className="p-8 bg-bg-surface rounded-xl border border-accent-dark">
                <Activity className="w-10 h-10 text-brand-cyan mb-4" />
                <h4 className="text-2xl font-semibold text-white mb-4">Real-Time Processing</h4>
                <ul className="space-y-3 text-txt-secondary">
                  <li>• WebSocket market data streaming</li>
                  <li>• 10ms agent communication latency</li>
                  <li>• 500+ symbols scanned per minute</li>
                  <li>• Sub-100ms signal generation</li>
                </ul>
              </div>

              <div className="p-8 bg-bg-surface rounded-xl border border-accent-dark">
                <Target className="w-10 h-10 text-brand-purple mb-4" />
                <h4 className="text-2xl font-semibold text-white mb-4">Proven Performance</h4>
                <ul className="space-y-3 text-txt-secondary">
                  <li>• 2x better risk-adjusted returns vs buy-and-hold</li>
                  <li>• 85% flash crash detection accuracy</li>
                  <li>• 99.9% system uptime</li>
                  <li>• 40-60% reduced unnecessary rebalancing</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-20 bg-gradient-to-r from-brand-cyan/10 to-brand-purple/10">
        <div className="container mx-auto px-6 text-center">
          <h3 className="text-4xl font-bold text-white mb-6">
            Ready to Experience the Future of Trading?
          </h3>
          <p className="text-xl text-txt-secondary mb-8 max-w-2xl mx-auto">
            Join the next generation of algorithmic traders leveraging AI, machine learning, 
            and advanced mathematics to dominate crypto markets.
          </p>
          <button
            onClick={() => navigate('/dashboard')}
            className="px-10 py-5 bg-gradient-to-r from-brand-cyan to-brand-purple text-white text-xl font-semibold rounded-lg hover:shadow-2xl hover:shadow-brand-cyan/50 transform hover:scale-105 transition-all"
          >
            Start Trading Now
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="py-8 border-t border-accent-dark">
        <div className="container mx-auto px-6 text-center text-txt-secondary">
          <p>© 2025 MirrorCore-X. Advanced Multi-Agent Cognitive Trading System.</p>
        </div>
      </div>
    </div>
  );
};

export default Landing;
