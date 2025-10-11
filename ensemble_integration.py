
"""
Ensemble Integration Module
Manages adaptive strategy weighting and regime-aware ensemble optimization
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedEnsembleManager:
    """
    Manages the full ensemble of strategies with adaptive weighting and regime detection.
    Integrates with MirrorCore-X SyncBus architecture.
    """
    
    def __init__(self, strategy_trainer, sync_bus, risk_profile: str = 'moderate'):
        self.strategy_trainer = strategy_trainer
        self.sync_bus = sync_bus
        self.risk_profile = risk_profile
        self.adaptive_enabled = True
        
        # Import adaptive optimizer
        try:
            from advanced_strategies import AdaptiveEnsembleOptimizer
            strategies = list(strategy_trainer.strategies.values())
            self.ensemble_optimizer = AdaptiveEnsembleOptimizer(strategies, risk_profile)
            self.optimizer_available = True
        except ImportError:
            logger.warning("Adaptive ensemble optimizer not available")
            self.ensemble_optimizer = None
            self.optimizer_available = False
        
        self.current_weights = {}
        self.current_regime = 'normal'
        self.performance_tracking = {}
        
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update ensemble weights based on current market regime and performance"""
        try:
            scanner_data = data.get('scanner_data', [])
            if not scanner_data:
                return {}
            
            df = pd.DataFrame(scanner_data)
            
            # Detect current regime
            if self.ensemble_optimizer:
                self.current_regime = self.ensemble_optimizer.detect_regime(df)
            
            # Calculate strategy performance
            recent_performance = self._calculate_recent_performance()
            
            # Use mathematical optimization for weights
            if hasattr(self.strategy_trainer, 'optimize_ensemble_weights'):
                math_weights = self.strategy_trainer.optimize_ensemble_weights(market_data=df)
                
                # Blend mathematical weights with adaptive weights
                if self.ensemble_optimizer and self.adaptive_enabled:
                    adaptive_weights = self.ensemble_optimizer.calculate_weights(
                        regime=self.current_regime,
                        recent_performance=recent_performance
                    )
                    
                    # 70% mathematical optimization, 30% adaptive heuristic
                    self.current_weights = {
                        name: 0.7 * math_weights.get(name, 0) + 0.3 * adaptive_weights.get(name, 0)
                        for name in set(list(math_weights.keys()) + list(adaptive_weights.keys()))
                    }
                else:
                    self.current_weights = math_weights
            else:
                # Fallback to adaptive weights only
                if self.ensemble_optimizer and self.adaptive_enabled:
                    self.current_weights = self.ensemble_optimizer.calculate_weights(
                        regime=self.current_regime,
                        recent_performance=recent_performance
                    )
            
            # Update global state
            await self.sync_bus.update_state('ensemble_weights', self.current_weights)
            await self.sync_bus.update_state('market_regime', self.current_regime)
            
            # Get optimization report if available
            opt_report = {}
            if hasattr(self.strategy_trainer, 'get_optimization_report'):
                opt_report = self.strategy_trainer.get_optimization_report()
            
            logger.info(f"Ensemble updated: regime={self.current_regime}, weights={len(self.current_weights)}")
            
            return {
                'regime': self.current_regime,
                'weights': self.current_weights,
                'performance': recent_performance,
                'optimization_report': opt_report
            }
            
        except Exception as e:
            logger.error(f"Ensemble update failed: {e}")
            return {}
    
    def _calculate_recent_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate recent performance metrics for each strategy"""
        performance = {}
        
        for strategy_name, perf_data in self.strategy_trainer.performance_tracker.items():
            if not perf_data:
                continue
            
            recent = perf_data[-20:]  # Last 20 trades
            
            if recent:
                wins = len([p for p in recent if p > 0])
                avg_return = np.mean(recent)
                volatility = np.std(recent) if len(recent) > 1 else 0.5
                sharpe = (avg_return / volatility) if volatility > 0 else 0
                
                performance[strategy_name] = {
                    'win_rate': wins / len(recent),
                    'sharpe': sharpe,
                    'volatility': volatility,
                    'avg_return': avg_return
                }
        
        return performance
    
    async def generate_ensemble_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate consensus signal from all strategies"""
        if not self.ensemble_optimizer:
            return {'direction': 'HOLD', 'strength': 0, 'confidence': 0}
        
        # Get signals from all strategies
        signals = {}
        for strategy_name, agent in self.strategy_trainer.strategies.items():
            try:
                signal_value = agent.evaluate(df) if hasattr(agent, 'evaluate') else 0
                signals[strategy_name] = float(signal_value)
            except Exception as e:
                logger.error(f"Strategy {strategy_name} evaluation failed: {e}")
                signals[strategy_name] = 0.0
        
        # Aggregate with current weights
        consensus = self.ensemble_optimizer.aggregate_signals(signals, self.current_weights)
        
        return consensus
    
    def get_status(self) -> Dict[str, Any]:
        """Get ensemble status"""
        return {
            'optimizer_available': self.optimizer_available,
            'adaptive_enabled': self.adaptive_enabled,
            'current_regime': self.current_regime,
            'risk_profile': self.risk_profile,
            'active_strategies': len(self.current_weights),
            'weights': self.current_weights
        }


async def create_enhanced_ensemble(strategy_trainer, sync_bus, risk_profile: str = 'moderate'):
    """Factory function to create enhanced ensemble manager"""
    ensemble = EnhancedEnsembleManager(strategy_trainer, sync_bus, risk_profile)
    
    # Attach to SyncBus
    sync_bus.attach('ensemble_manager', ensemble)
    
    logger.info(f"Enhanced ensemble manager created with {risk_profile} risk profile")
    return ensemble
