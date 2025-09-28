"""
bayesian_integration.py

Clean integration layer for adding Bayesian beliefs to MirrorCore-X
without modifying the main system file.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from bayesian_oracle import (
    BayesianOracleEnhancement, 
    EnhancedTradingOracleEngine,
    BayesianStrategyTrainerEnhancement
)

logger = logging.getLogger(__name__)

class MirrorCoreBayesianIntegration:
    """Clean integration wrapper for Bayesian enhancements."""
    
    def __init__(self, sync_bus, enable_bayesian: bool = True):
        self.sync_bus = sync_bus
        self.enabled = enable_bayesian
        self.bayesian_enhancement = None  # Instance of BayesianOracleEnhancement
        self.enhanced_oracle = None
        self.enhanced_trainer = None
        self.monitor = None
        self.original_components = {}
        
    async def apply_enhancements(self) -> bool:
        """Apply Bayesian enhancements to existing MirrorCore system."""
        if not self.enabled:
            logger.info("Bayesian enhancements disabled")
            return False
            
        try:
            # Get original components
            original_oracle = self.sync_bus.agents.get('oracle')
            strategy_trainer = self.sync_bus.agents.get('strategy_trainer') 
            
            if not original_oracle or not strategy_trainer:
                logger.error("Required MirrorCore components not found")
                return False
                
            # Store originals for potential rollback
            self.original_components = {
                'oracle': original_oracle,
                'strategy_trainer': strategy_trainer
            }
            
            # Create the core Bayesian enhancement object
            self.bayesian_enhancement = BayesianOracleEnhancement()

            # Create enhanced components, passing the Bayesian enhancement
            self.enhanced_oracle = EnhancedTradingOracleEngine(self.bayesian_enhancement)
            self.enhanced_trainer = BayesianStrategyTrainerEnhancement(self.bayesian_enhancement)

            # Replace in sync_bus
            self.sync_bus.detach("oracle")
            self.sync_bus.detach("strategy_trainer")
            self.sync_bus.attach("enhanced_oracle", self.enhanced_oracle)
            self.sync_bus.attach("enhanced_strategy_trainer", self.enhanced_trainer)

            # Add monitor
            self.monitor = BayesianMonitor(self.bayesian_enhancement)
            self.sync_bus.attach("bayesian_monitor", self.monitor)

            logger.info("Bayesian enhancements successfully applied")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply Bayesian enhancements: {e}")
            await self.rollback()
            return False
    
    async def rollback(self):
        """Rollback to original components if enhancement fails."""
        try:
            if self.original_components:
                # Remove enhanced components
                for name in ["enhanced_oracle", "enhanced_strategy_trainer", "bayesian_monitor"]:
                    self.sync_bus.detach(name)
                
                # Restore originals
                self.sync_bus.attach("oracle", self.original_components['oracle'])
                self.sync_bus.attach("strategy_trainer", self.original_components['strategy_trainer'])
                
                logger.info("Rolled back to original MirrorCore components")
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    def get_bayesian_insights(self) -> Optional[Dict[str, Any]]:
        """Get current Bayesian insights if available."""
        if not self.enabled or not self.bayesian_enhancement:
            return None
        try:
            # Use a minimal empty DataFrame if no data is available
            import pandas as pd
            empty_df = pd.DataFrame()
            return self.bayesian_enhancement.get_strategy_recommendation(empty_df)
        except Exception as e:
            logger.error(f"Failed to get Bayesian insights: {e}")
            return None
    
    async def export_beliefs(self, filename: str = "bayesian_beliefs.json") -> bool:
        """Export current beliefs state."""
        if not self.enabled or not self.bayesian_enhancement:
            return False
        try:
            import json
            # You may want to implement export_beliefs_state in BayesianOracleEnhancement
            if hasattr(self.bayesian_enhancement, 'export_beliefs_summary'):
                beliefs_state = self.bayesian_enhancement.export_beliefs_summary().to_dict(orient='records')
            else:
                beliefs_state = {}
            with open(filename, 'w') as f:
                json.dump(beliefs_state, f, indent=2)
            logger.info(f"Bayesian beliefs exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export beliefs: {e}")
            return False

class BayesianMonitor:
    """Monitor agent for Bayesian system health."""
    
    def __init__(self, bayesian_enhancement):
        self.bayesian_enhancement = bayesian_enhancement
        self.last_decay = 0.0
        self.decay_interval = 3600.0  # 1 hour
        self.health_checks = 0
        
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and maintain Bayesian system."""
        import time
        
        current_time = time.time()
        self.health_checks += 1
        
        # Apply periodic decay (if implemented)
        if current_time - self.last_decay > self.decay_interval:
            try:
                if hasattr(self.bayesian_enhancement, 'decay_old_beliefs'):
                    self.bayesian_enhancement.decay_old_beliefs()
                self.last_decay = current_time
                logger.debug("Applied Bayesian belief decay")
            except Exception as e:
                logger.error(f"Belief decay failed: {e}")

        # Export state periodically
        beliefs_state = None
        if self.health_checks % 100 == 0:  # Every 100 ticks
            try:
                if hasattr(self.bayesian_enhancement, 'export_beliefs_summary'):
                    beliefs_state = self.bayesian_enhancement.export_beliefs_summary().to_dict(orient='records')
            except Exception as e:
                logger.error(f"Beliefs export failed: {e}")

        return {
            "bayesian_health": {
                "last_decay": self.last_decay,
                "health_checks": self.health_checks,
                "beliefs_state": beliefs_state
            }
        }

# Integration function for easy use
async def add_bayesian_to_mirrorcore(sync_bus, enable: bool = True) -> MirrorCoreBayesianIntegration:
    """
    Easy integration function.
    
    Usage:
        # After creating your MirrorCore system
        sync_bus, trade_analyzer, scanner, exchange = await create_mirrorcore_system(dry_run=True)
        
        # Add Bayesian enhancements
        bayesian_integration = await add_bayesian_to_mirrorcore(sync_bus, enable=True)
        
        # Run system as normal - now with Bayesian intelligence!
        for i in range(100):
            await sync_bus.tick()
            
            # Optional: Get insights
            if i % 20 == 0:
                insights = bayesian_integration.get_bayesian_insights()
                if insights:
                    print(f"Best strategy: {insights.get('best_strategy')}")
    """
    
    integration = MirrorCoreBayesianIntegration(sync_bus, enable)
    
    if enable:
        success = await integration.apply_enhancements()
        if success:
            logger.info("MirrorCore enhanced with Bayesian intelligence")
        else:
            logger.warning("Bayesian enhancement failed, running with original system")
    
    return integration

# Configuration class for easy customization
class BayesianConfig:
    """Configuration for Bayesian system behavior."""
    
    def __init__(self):
        # Belief update parameters
        self.decay_factor = 0.98
        self.confidence_weighting = True
        self.max_history_length = 200
        
        # Regime detection sensitivity
        self.regime_sensitivity = 0.3
        self.volume_importance = 0.4
        self.volatility_threshold = 0.8
        
        # Strategy selection
        self.uncertainty_penalty = 0.3
        self.regime_weight = 0.8
        self.min_evidence_threshold = 3
        
        # System maintenance
        self.decay_interval = 3600.0  # 1 hour
        self.export_interval = 6000.0  # ~100 minutes
        
    def apply_to_integration(self, integration: MirrorCoreBayesianIntegration):
        """Apply config to integration instance."""
        if integration.bayesian_enhancement:
            # Apply configuration to Bayesian components
            bayesian_oracle = integration.bayesian_enhancement
            # Update regime detector
            if hasattr(bayesian_oracle.base_oracle, 'market_regime_detector'):
                detector = bayesian_oracle.base_oracle.market_regime_detector
                # Only set regime_sensitivity if the attribute exists
                if hasattr(detector, 'regime_sensitivity'):
                    detector.regime_sensitivity = self.regime_sensitivity
                else:
                    logger.warning("MarketRegimeDetector has no attribute 'regime_sensitivity'; skipping assignment.")
            # Update belief trackers
            for belief in bayesian_oracle.base_oracle.strategy_beliefs.values():
                if hasattr(belief.overall_belief, 'decay_factor'):
                    belief.overall_belief.decay_factor = self.decay_factor
                else:
                    logger.warning(f"BeliefTracker for {getattr(belief, 'name', 'unknown')} has no attribute 'decay_factor'; skipping assignment.")
                for regime_belief in belief.regime_beliefs.values():
                    if hasattr(regime_belief, 'decay_factor'):
                        regime_belief.decay_factor = self.decay_factor
                    else:
                        logger.warning(f"Regime BeliefTracker has no attribute 'decay_factor'; skipping assignment.")
        
        if integration.monitor:
            integration.monitor.decay_interval = self.decay_interval
            
        logger.info("Applied Bayesian configuration")

# Example usage patterns
if __name__ == "__main__":
    print("""
    Bayesian Integration Examples:
    
    # Basic integration:
    bayesian_integration = await add_bayesian_to_mirrorcore(sync_bus)
    
    # With custom config:
    config = BayesianConfig()
    config.decay_factor = 0.95  # Slower decay
    config.uncertainty_penalty = 0.5  # Higher penalty for uncertainty
    
    bayesian_integration = await add_bayesian_to_mirrorcore(sync_bus)
    config.apply_to_integration(bayesian_integration)
    
    # Disable if needed:
    bayesian_integration = await add_bayesian_to_mirrorcore(sync_bus, enable=False)
    """)
