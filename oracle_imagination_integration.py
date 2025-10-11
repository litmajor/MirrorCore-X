
"""
Oracle & Imagination Integration Module
Integrates Trading Oracle Engine and Imagination Engine into MirrorCore-X
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from imagination_engine import ImaginationEngine
from imagination_integration import ImaginationIntegration
from bayesian_integration import add_bayesian_to_mirrorcore
import pandas as pd

logger = logging.getLogger(__name__)

class OracleImaginationIntegration:
    """
    Unified integration layer for Oracle and Imagination engines
    """
    
    def __init__(self, sync_bus, scanner, strategy_trainer, execution_daemon, trade_analyzer, arch_ctrl):
        self.sync_bus = sync_bus
        self.scanner = scanner
        self.strategy_trainer = strategy_trainer
        self.execution_daemon = execution_daemon
        self.trade_analyzer = trade_analyzer
        self.arch_ctrl = arch_ctrl
        
        # Components to be initialized
        self.oracle_engine = None
        self.imagination_engine = None
        self.bayesian_integration = None
        
        # Integration state
        self.is_initialized = False
        self.last_oracle_directives = []
        self.last_imagination_results = {}
        
        logger.info("Oracle & Imagination Integration initialized")
    
    async def initialize(self, enable_bayesian: bool = True, enable_imagination: bool = True):
        """Initialize all enhancement systems"""
        try:
            # 1. Create Trading Oracle Engine
            self.oracle_engine = await self._create_oracle_engine()
            
            # 2. Initialize Bayesian enhancement if enabled
            if enable_bayesian:
                self.bayesian_integration = await add_bayesian_to_mirrorcore(
                    self.sync_bus, 
                    enable=True
                )
                logger.info("Bayesian Oracle integrated")
            
            # 3. Initialize Imagination Engine if enabled
            if enable_imagination:
                market_data = await self.sync_bus.get_state('market_data') or []
                if not market_data:
                    # Generate initial market data if empty
                    market_data = await self._generate_initial_market_data()
                
                self.imagination_engine = ImaginationEngine(
                    self.oracle_engine,
                    self.strategy_trainer,
                    self.execution_daemon
                )
                
                await self.imagination_engine.initialize(market_data)
                logger.info("Imagination Engine integrated")
            
            # 4. Attach oracle to SyncBus
            self.sync_bus.attach('oracle_engine', self.oracle_engine)
            
            self.is_initialized = True
            logger.info("✨ All enhancement systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhancement systems: {e}")
            return False
    
    async def _create_oracle_engine(self):
        """Create and configure Trading Oracle Engine"""
        from mirrax import TradingOracleEngine, RLIntegrationLayer
        
        # Check if RL is available
        rl_integration = None
        try:
            market_data = await self.sync_bus.get_state('market_data') or []
            if market_data:
                rl_integration = RLIntegrationLayer(self.sync_bus, self.scanner)
                logger.info("RL Integration Layer created")
        except Exception as e:
            logger.warning(f"RL Integration not available: {e}")
        
        oracle = TradingOracleEngine(
            strategy_trainer=self.strategy_trainer,
            trade_analyzer=self.trade_analyzer,
            rl_integration=rl_integration
        )
        
        logger.info("Trading Oracle Engine created")
        return oracle
    
    async def _generate_initial_market_data(self) -> List[Dict]:
        """Generate initial market data for bootstrapping"""
        from mirrax import MarketDataGenerator
        
        generator = MarketDataGenerator(sentiment_bias=0.0, volatility=0.02)
        market_data = []
        
        for _ in range(100):
            data = await generator.generate()
            market_data.append(data.model_dump() if hasattr(data, 'model_dump') else data)
        
        logger.info("Generated initial market data for bootstrapping")
        return market_data
    
    async def run_enhanced_cycle(self) -> Dict[str, Any]:
        """Run a complete enhanced trading cycle"""
        if not self.is_initialized:
            logger.warning("Enhancement systems not initialized")
            return {'status': 'not_initialized'}
        
        results = {
            'oracle_directives': [],
            'imagination_insights': {},
            'bayesian_recommendations': {},
            'timestamp': asyncio.get_event_loop().time()
        }
        
        try:
            # 1. Get current market context
            market_context = await self._get_market_context()
            scanner_data = await self.sync_bus.get_state('scanner_data') or []
            
            # 2. Get RL predictions if available
            rl_predictions = await self.sync_bus.get_state('rl_predictions') or []
            
            # 3. Generate Oracle directives
            if self.oracle_engine and scanner_data:
                directives = await self.oracle_engine.evaluate_trading_timelines(
                    market_context=market_context,
                    scanner_data=scanner_data,
                    rl_predictions=rl_predictions
                )
                self.last_oracle_directives = directives
                results['oracle_directives'] = directives
                
                # Update state
                await self.sync_bus.update_state('oracle_directives', directives)
            
            # 4. Get Bayesian recommendations if available
            if self.bayesian_integration:
                try:
                    scanner_df = pd.DataFrame(scanner_data) if scanner_data else pd.DataFrame()
                    bayesian_rec = self.bayesian_integration.get_bayesian_insights()
                    if bayesian_rec:
                        results['bayesian_recommendations'] = bayesian_rec
                except Exception as e:
                    logger.warning(f"Bayesian insights not available: {e}")
            
            # 5. Run Imagination analysis (periodic, not every cycle)
            if self.imagination_engine:
                imagination_status = self.imagination_engine.get_status()
                if imagination_status.get('initialized'):
                    # Check if analysis is due
                    current_time = asyncio.get_event_loop().time()
                    last_analysis = imagination_status.get('last_analysis', 0)
                    
                    if current_time - last_analysis > 3600:  # Every hour
                        imagination_results = await self.imagination_engine.run_counterfactual_analysis()
                        self.last_imagination_results = imagination_results
                        results['imagination_insights'] = imagination_results
                    else:
                        results['imagination_insights'] = {
                            'status': 'cached',
                            'last_analysis_age_minutes': (current_time - last_analysis) / 60
                        }
            
            logger.info(f"Enhanced cycle complete: {len(results['oracle_directives'])} directives generated")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced cycle failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _get_market_context(self) -> Dict[str, Any]:
        """Get current market context for Oracle"""
        scanner_data = await self.sync_bus.get_state('scanner_data') or []
        trades = await self.sync_bus.get_state('trades') or []
        
        # Calculate portfolio value
        portfolio_value = self.trade_analyzer.get_total_pnl() + 10000.0
        
        # Calculate volatility
        if scanner_data:
            scanner_df = pd.DataFrame(scanner_data)
            volatility = float(
                scanner_df['price'].pct_change().std()
            ) if len(scanner_df) >= 10 else 0.01
            sentiment = scanner_df.get('momentum_7d', pd.Series([0])).mean()
        else:
            volatility = 0.01
            sentiment = 0.0
        
        return {
            'portfolio_value': portfolio_value,
            'volatility': volatility,
            'sentiment': sentiment,
            'total_trades': len(trades)
        }
    
    async def force_imagination_analysis(self) -> Dict[str, Any]:
        """Force immediate imagination analysis"""
        if not self.imagination_engine:
            return {'status': 'not_initialized'}
        
        logger.info("🔄 Forcing imagination analysis...")
        results = await self.imagination_engine.force_reanalysis()
        self.last_imagination_results = results
        return results
    
    def get_oracle_directives(self) -> List[Dict]:
        """Get last oracle directives"""
        return self.last_oracle_directives
    
    def get_imagination_insights(self) -> Dict[str, Any]:
        """Get last imagination results"""
        return self.last_imagination_results
    
    async def export_analysis(self, filepath: str = "enhanced_analysis.json"):
        """Export comprehensive analysis"""
        import json
        
        export_data = {
            'oracle_directives': self.last_oracle_directives,
            'imagination_insights': self.last_imagination_results,
            'bayesian_beliefs': None,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        # Add Bayesian beliefs if available
        if self.bayesian_integration:
            try:
                beliefs_df = self.bayesian_integration.bayesian_enhancement.export_beliefs_summary()
                export_data['bayesian_beliefs'] = beliefs_df.to_dict(orient='records')
            except Exception as e:
                logger.warning(f"Could not export Bayesian beliefs: {e}")
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Enhanced analysis exported to {filepath}")
        return filepath
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            'is_initialized': self.is_initialized,
            'oracle_active': self.oracle_engine is not None,
            'bayesian_active': self.bayesian_integration is not None,
            'imagination_active': self.imagination_engine is not None,
            'last_directives_count': len(self.last_oracle_directives),
            'imagination_status': self.imagination_engine.get_status() if self.imagination_engine else {}
        }


async def integrate_oracle_and_imagination(sync_bus, scanner, strategy_trainer, 
                                          execution_daemon, trade_analyzer, arch_ctrl,
                                          enable_bayesian: bool = True,
                                          enable_imagination: bool = True) -> OracleImaginationIntegration:
    """
    Main integration function
    
    Usage:
        integration = await integrate_oracle_and_imagination(
            sync_bus, scanner, strategy_trainer, 
            execution_daemon, trade_analyzer, arch_ctrl
        )
        
        # Run enhanced cycle
        results = await integration.run_enhanced_cycle()
        
        # Get directives
        directives = integration.get_oracle_directives()
    """
    
    integration = OracleImaginationIntegration(
        sync_bus, scanner, strategy_trainer,
        execution_daemon, trade_analyzer, arch_ctrl
    )
    
    success = await integration.initialize(
        enable_bayesian=enable_bayesian,
        enable_imagination=enable_imagination
    )
    
    if success:
        logger.info("✨ Oracle & Imagination fully integrated")
    else:
        logger.warning("⚠️ Integration partially failed, check logs")
    
    return integration
