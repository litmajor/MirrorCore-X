
"""
Enhanced MirrorOptimizer for comprehensive parameter optimization across all MirrorCore-X components
"""

from bayes_opt import BayesianOptimization
from typing import Dict, Callable, Optional, Any, List, Tuple
import numpy as np
import logging
from abc import ABC, abstractmethod
import json
import asyncio
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizableAgent(ABC):
    """Abstract base class for agents that can be optimized."""
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameters as a dictionary."""
        pass
    
    @abstractmethod
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set hyperparameters from a dictionary."""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters before setting them."""
        pass
    
    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the agent's performance. Higher values are better."""
        pass

class ComprehensiveMirrorOptimizer:
    """Enhanced Bayesian optimization engine covering ALL MirrorCore-X parameters."""
    
    def __init__(self, system_components: Dict[str, Any]):
        """
        Initialize with complete system components.
        
        Args:
            system_components: Dictionary containing all MirrorCore-X components
        """
        self.components = system_components
        self.optimization_history = {}
        self.global_optimization_results = {}
        
        # Define comprehensive parameter bounds for ALL components
        self.parameter_bounds = {
            # Scanner/Market Analysis Parameters
            'scanner': {
                'momentum_period': (5, 50),
                'rsi_window': (5, 50), 
                'volume_threshold': (0.5, 10.0),
                'macd_fast': (8, 16),
                'macd_slow': (20, 35),
                'bb_period': (15, 25),
                'bb_std': (1.5, 2.5),
                'ichimoku_conversion': (7, 12),
                'ichimoku_base': (22, 30),
                'adx_period': (10, 18),
                'momentum_lookback': (3, 15),
                'volatility_window': (10, 30),
                'volume_sma_period': (15, 25)
            },
            
            # Strategy Trainer Parameters
            'strategy_trainer': {
                'learning_rate': (0.001, 0.1),
                'confidence_threshold': (0.1, 0.9),
                'min_weight': (0.05, 0.2),
                'max_weight': (0.8, 1.0),
                'lookback_window': (5, 50),
                'pnl_scale_factor': (0.05, 0.5),
                'performance_decay': (0.9, 0.99),
                'weight_adjustment_rate': (0.01, 0.2)
            },
            
            # ARCH_CTRL Emotional Parameters
            'arch_ctrl': {
                'fear_sensitivity': (0.1, 1.0),
                'confidence_boost_rate': (0.01, 0.1),
                'stress_decay_rate': (0.005, 0.05),
                'volatility_fear_threshold': (0.02, 0.08),
                'override_confidence_min': (0.8, 0.95),
                'emotional_momentum_threshold': (0.1, 0.5),
                'stress_trigger_limit': (3, 10),
                'trust_decay_rate': (0.01, 0.1)
            },
            
            # Execution Daemon Parameters
            'execution_daemon': {
                'position_size_factor': (0.5, 2.0),
                'risk_per_trade': (0.01, 0.05),
                'max_positions': (1, 10),
                'slippage_tolerance': (0.001, 0.01),
                'timeout_seconds': (5, 30),
                'retry_attempts': (1, 5),
                'profit_target_multiplier': (1.5, 3.0),
                'stop_loss_multiplier': (0.5, 1.5)
            },
            
            # Risk Sentinel Parameters
            'risk_sentinel': {
                'max_drawdown': (-0.5, -0.1),
                'max_position_limit': (0.05, 0.3),
                'max_loss_per_trade': (-0.1, -0.01),
                'correlation_limit': (0.3, 0.8),
                'var_confidence': (0.9, 0.99),
                'lookback_periods': (20, 100),
                'volatility_multiplier': (1.0, 3.0)
            },
            
            # Trading Strategies Parameters
            'ut_bot': {
                'atr_period': (10, 20),
                'sensitivity': (1.0, 3.0),
                'key_value': (1.0, 5.0),
                'atr_multiplier': (2.0, 4.0)
            },
            
            'gradient_trend': {
                'ema_fast': (5, 15),
                'ema_slow': (20, 50),
                'gradient_threshold': (0.001, 0.01),
                'trend_strength_min': (0.3, 0.8),
                'smoothing_period': (3, 10)
            },
            
            'volume_sr': {
                'volume_threshold': (1.2, 3.0),
                'support_resistance_period': (20, 50),
                'breakout_confirmation': (2, 8),
                'volume_ma_period': (10, 30)
            },
            
            # Enhanced Scanner Parameters
            'momentum_scanner': {
                'scan_interval': (30, 300),
                'top_n_results': (10, 100),
                'min_volume_usd': (100000, 5000000),
                'momentum_bias': (0.3, 0.8),
                'cluster_lookback': (5, 20),
                'volume_threshold_multiplier': (1.5, 3.0)
            },
            
            # Continuous Scanner Parameters
            'continuous_scanner': {
                'tick_interval': (1, 10),
                'signal_generation_interval': (60, 300),
                'market_state_update_interval': (120, 600),
                'data_persistence_limit': (500, 2000),
                'regime_sensitivity': (0.2, 0.5),
                'reversion_threshold': (0.5, 0.8)
            },
            
            # Meta Agent Parameters
            'meta_agent': {
                'optimization_interval': (1800, 7200),
                'performance_threshold': (-0.2, 0.1),
                'adaptation_rate': (0.1, 0.5),
                'system_health_threshold': (0.6, 0.9),
                'emergency_threshold': (-0.3, -0.1)
            },
            
            # SyncBus Performance Parameters
            'syncbus': {
                'tick_timeout': (5, 30),
                'queue_max_size': (50, 200),
                'circuit_breaker_threshold': (3, 10),
                'agent_restart_delay': (10, 120),
                'health_check_interval': (30, 300)
            },
            
            # Bayesian Oracle Parameters (if using bayesian_integration)
            'bayesian_oracle': {
                'decay_factor': (0.9, 0.99),
                'confidence_weighting': (0.1, 1.0),
                'max_history_length': (100, 500),
                'regime_sensitivity': (0.2, 0.6),
                'uncertainty_penalty': (0.1, 0.5),
                'min_evidence_threshold': (2, 10)
            },
            
            # RL Trading System Parameters (if using RL)
            'rl_system': {
                'learning_rate': (0.0001, 0.01),
                'gamma': (0.9, 0.999),
                'epsilon': (0.01, 0.3),
                'batch_size': (16, 128),
                'memory_size': (1000, 10000),
                'update_frequency': (1, 10),
                'target_update_frequency': (10, 100)
            }
        }
        
        logger.info(f"Comprehensive MirrorOptimizer initialized with {len(self.parameter_bounds)} component types")
    
    def get_all_optimizable_components(self) -> Dict[str, OptimizableAgent]:
        """Get all components that implement OptimizableAgent interface."""
        optimizable = {}
        
        for name, component in self.components.items():
            if hasattr(component, 'get_hyperparameters') and \
               hasattr(component, 'set_hyperparameters') and \
               hasattr(component, 'validate_params') and \
               hasattr(component, 'evaluate'):
                optimizable[name] = component
                logger.info(f"Found optimizable component: {name}")
        
        return optimizable
    
    def optimize_component(self, 
                          component_name: str,
                          component: OptimizableAgent,
                          custom_bounds: Optional[Dict[str, tuple]] = None,
                          iterations: int = 20,
                          init_points: int = 8) -> Dict[str, Any]:
        """
        Optimize a single component with comprehensive parameter coverage.
        """
        # Use custom bounds or default bounds for this component type
        bounds = custom_bounds or self.parameter_bounds.get(component_name, {})
        
        if not bounds:
            logger.warning(f"No bounds defined for component {component_name}")
            return {}
        
        # Filter bounds to only include parameters the component actually has
        current_params = component.get_hyperparameters()
        filtered_bounds = {k: v for k, v in bounds.items() if k in current_params}
        
        if not filtered_bounds:
            logger.warning(f"No matching parameters found for component {component_name}")
            return {}
        
        logger.info(f"Optimizing {component_name} with parameters: {list(filtered_bounds.keys())}")
        
        # Store original parameters
        original_params = component.get_hyperparameters()
        
        def objective_function(**params):
            try:
                # Validate and set parameters
                if not component.validate_params(params):
                    return -np.inf
                
                component.set_hyperparameters(params)
                score = component.evaluate()
                
                logger.debug(f"{component_name} - Params: {params} -> Score: {score:.4f}")
                return score
                
            except Exception as e:
                logger.error(f"Error evaluating {component_name}: {e}")
                return -np.inf
        
        # Set up Bayesian optimization
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=filtered_bounds,
            random_state=42,
            verbose=1
        )
        
        try:
            # Run optimization
            optimizer.maximize(init_points=init_points, n_iter=iterations)
            
            # Get results
            if optimizer.max:
                best_params = optimizer.max.get("params", {})
                best_score = optimizer.max.get("target", -np.inf)
                
                # Apply best parameters
                if best_params:
                    component.set_hyperparameters(best_params)
                
                # Store optimization history
                self.optimization_history[component_name] = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'original_params': original_params,
                    'bounds': filtered_bounds,
                    'iterations': iterations,
                    'all_results': [
                        {'params': dict(res['params']), 'target': res['target']}
                        for res in optimizer.res
                    ]
                }
                
                logger.info(f"Optimization complete for {component_name}: score={best_score:.4f}")
                return best_params
            
        except Exception as e:
            logger.error(f"Optimization failed for {component_name}: {e}")
            # Rollback to original parameters
            component.set_hyperparameters(original_params)
            
        return {}
    
    def optimize_all_components(self, 
                               iterations_per_component: int = 15,
                               parallel: bool = False) -> Dict[str, Any]:
        """
        Optimize ALL optimizable components in the system.
        """
        optimizable_components = self.get_all_optimizable_components()
        results = {}
        
        logger.info(f"Starting comprehensive optimization of {len(optimizable_components)} components")
        
        if parallel:
            # Parallel optimization (use with caution - may cause conflicts)
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        self.optimize_component, 
                        name, 
                        component, 
                        None, 
                        iterations_per_component
                    ): name
                    for name, component in optimizable_components.items()
                }
                
                for future in concurrent.futures.as_completed(futures):
                    component_name = futures[future]
                    try:
                        result = future.result()
                        results[component_name] = result
                    except Exception as e:
                        logger.error(f"Parallel optimization failed for {component_name}: {e}")
                        results[component_name] = {}
        else:
            # Sequential optimization (safer)
            for name, component in optimizable_components.items():
                try:
                    result = self.optimize_component(name, component, None, iterations_per_component)
                    results[name] = result
                except Exception as e:
                    logger.error(f"Sequential optimization failed for {name}: {e}")
                    results[name] = {}
        
        # Store global results
        self.global_optimization_results = {
            'timestamp': time.time(),
            'total_components': len(optimizable_components),
            'successful_optimizations': len([r for r in results.values() if r]),
            'results': results,
            'performance_improvement': self._calculate_improvement()
        }
        
        logger.info(f"Comprehensive optimization complete: {len([r for r in results.values() if r])}/{len(optimizable_components)} successful")
        return results
    
    def optimize_by_category(self, category: str, iterations: int = 20) -> Dict[str, Any]:
        """
        Optimize all components in a specific category.
        """
        category_mapping = {
            'trading': ['ut_bot', 'gradient_trend', 'volume_sr', 'strategy_trainer'],
            'analysis': ['scanner', 'momentum_scanner', 'continuous_scanner'],
            'risk': ['risk_sentinel', 'arch_ctrl'],
            'execution': ['execution_daemon', 'meta_agent'],
            'system': ['syncbus'],
            'ml': ['bayesian_oracle', 'rl_system']
        }
        
        if category not in category_mapping:
            logger.error(f"Unknown category: {category}")
            return {}
        
        component_names = category_mapping[category]
        optimizable_components = self.get_all_optimizable_components()
        
        results = {}
        for name in component_names:
            if name in optimizable_components:
                result = self.optimize_component(name, optimizable_components[name], None, iterations)
                results[name] = result
        
        logger.info(f"Category '{category}' optimization complete: {len(results)} components")
        return results
    
    def _calculate_improvement(self) -> Dict[str, float]:
        """Calculate performance improvement from optimization."""
        improvements = {}
        
        for component_name, history in self.optimization_history.items():
            if 'best_score' in history and 'all_results' in history:
                initial_scores = [r['target'] for r in history['all_results'][:3]]  # First 3 evaluations
                best_score = history['best_score']
                
                if initial_scores:
                    avg_initial = np.mean(initial_scores)
                    improvement = (best_score - avg_initial) / abs(avg_initial) if avg_initial != 0 else 0
                    improvements[component_name] = improvement
        
        return improvements
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            'optimization_history': self.optimization_history,
            'global_results': self.global_optimization_results,
            'parameter_coverage': {
                category: list(bounds.keys()) 
                for category, bounds in self.parameter_bounds.items()
            },
            'total_parameters': sum(len(bounds) for bounds in self.parameter_bounds.values()),
            'optimizable_components': list(self.get_all_optimizable_components().keys())
        }
    
    def save_comprehensive_results(self, filename: str = "comprehensive_optimization_results.json"):
        """Save all optimization results."""
        report = self.get_optimization_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive optimization results saved to {filename}")
    
    def load_optimization_state(self, filename: str):
        """Load previous optimization state."""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.optimization_history = state.get('optimization_history', {})
            self.global_optimization_results = state.get('global_results', {})
            
            logger.info(f"Optimization state loaded from {filename}")
        except Exception as e:
            logger.error(f"Failed to load optimization state: {e}")


# Integration helper for MirrorCore-X system
def create_comprehensive_optimizer(sync_bus, components_dict) -> ComprehensiveMirrorOptimizer:
    """
    Create a comprehensive optimizer with all MirrorCore-X components.
    
    Args:
        sync_bus: The main SyncBus instance
        components_dict: Dictionary of all system components
    
    Returns:
        Configured ComprehensiveMirrorOptimizer instance
    """
    # Ensure all components are accessible
    all_components = {
        'sync_bus': sync_bus,
        **components_dict
    }
    
    optimizer = ComprehensiveMirrorOptimizer(all_components)
    
    # Add any custom parameter bounds for specific instances
    # This can be extended based on actual component implementations
    
    return optimizer


# Example usage and testing
if __name__ == "__main__":
    # Test the comprehensive optimizer
    print("=== Comprehensive MirrorOptimizer Test ===")
    
    # Mock components for testing
    class MockOptimizableComponent(OptimizableAgent):
        def __init__(self, name):
            self.name = name
            self.param1 = 0.5
            self.param2 = 10
            
        def get_hyperparameters(self):
            return {'param1': self.param1, 'param2': self.param2}
        
        def set_hyperparameters(self, params):
            self.param1 = params.get('param1', self.param1)
            self.param2 = params.get('param2', self.param2)
        
        def validate_params(self, params):
            return all(isinstance(v, (int, float)) for v in params.values())
        
        def evaluate(self):
            # Mock evaluation - optimal at param1=0.7, param2=15
            return 1.0 - abs(self.param1 - 0.7) - abs(self.param2 - 15) * 0.01
    
    # Create mock system
    mock_components = {
        'scanner': MockOptimizableComponent('scanner'),
        'strategy_trainer': MockOptimizableComponent('strategy_trainer'),
        'arch_ctrl': MockOptimizableComponent('arch_ctrl')
    }
    
    # Initialize comprehensive optimizer
    optimizer = ComprehensiveMirrorOptimizer(mock_components)
    
    # Test optimization
    results = optimizer.optimize_all_components(iterations_per_component=10)
    
    print(f"Optimization Results: {results}")
    
    # Generate report
    report = optimizer.get_optimization_report()
    print(f"Total parameters covered: {report['total_parameters']}")
    print(f"Component categories: {len(report['parameter_coverage'])}")
    
    # Save results
    optimizer.save_comprehensive_results("test_optimization_results.json")
    
    print("=== Test Complete ===")
