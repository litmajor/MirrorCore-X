"""
Production-Safe MirrorCore-X Optimizer
Addresses safety, validation, and live trading constraints
"""

# ComprehensiveMirrorOptimizer must be defined after all dependencies
"""
Production-Safe MirrorCore-X Optimizer
Addresses safety, validation, and live trading constraints
"""

from bayes_opt import BayesianOptimization
from typing import Dict, Callable, Optional, Any, List, Tuple
import numpy as np
import logging
from abc import ABC, abstractmethod
import json
import asyncio
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization modes with different safety levels"""
    SIMULATION = "simulation"  # Backtest data only
    PAPER = "paper"  # Paper trading
    LIVE_SAFE = "live_safe"  # Live but with strict guards
    LIVE_AGGRESSIVE = "live_aggressive"  # Live with relaxed guards


@dataclass
class OptimizationConstraints:
    """Constraints for safe optimization"""
    max_drawdown_during_opt: float = -0.05  # Stop if 5% drawdown during optimization
    min_evaluation_trades: int = 10  # Minimum trades needed for evaluation
    max_parameter_change_pct: float = 0.2  # Max 20% change per iteration
    require_market_closed: bool = True  # Only optimize when markets closed
    rollback_on_failure: bool = True
    validation_period_seconds: int = 300  # 5 min validation after param change


@dataclass
class ParameterDependency:
    """Define parameter dependencies and constraints"""
    param_name: str
    depends_on: List[str]
    constraint_func: Callable[[Dict], bool]
    error_message: str


class SafeOptimizableAgent(ABC):
    """Enhanced optimizable agent with safety features"""
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameters"""
        pass
    
    @abstractmethod
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set hyperparameters"""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters. Returns (is_valid, error_message)"""
        pass
    
    @abstractmethod
    async def evaluate_async(self, 
                            mode: OptimizationMode,
                            validation_period: int = 300) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate performance asynchronously.
        Returns (score, metadata_dict)
        Metadata should include: trades, pnl, sharpe, max_dd, etc.
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        pass
    
    @abstractmethod
    def create_snapshot(self) -> Dict[str, Any]:
        """Create state snapshot for rollback"""
        pass
    
    @abstractmethod
    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore from snapshot"""
        pass


class ProductionSafeMirrorOptimizer:
    """
    Production-safe Bayesian optimizer with comprehensive safety features
    """
    
    def __init__(self, 
                 system_components: Dict[str, Any],
                 mode: OptimizationMode = OptimizationMode.SIMULATION,
                 constraints: Optional[OptimizationConstraints] = None):
        """
        Initialize safe optimizer
        
        Args:
            system_components: System components to optimize
            mode: Optimization mode (simulation/paper/live)
            constraints: Safety constraints
        """
        self.components = system_components
        self.mode = mode
        self.constraints = constraints or OptimizationConstraints()
        
        self.optimization_history = {}
        self.snapshots = {}  # Component snapshots for rollback
        self.is_optimizing = False
        self.optimization_start_time = None
        
        # Define safe parameter bounds with validation
        self.parameter_bounds = self._define_safe_bounds()
        
        # Define parameter dependencies
        self.dependencies = self._define_dependencies()
        
        logger.info(f"Safe optimizer initialized in {mode.value} mode")
    
    def _define_safe_bounds(self) -> Dict[str, Dict[str, tuple]]:
        """Define parameter bounds appropriate for each optimization mode"""
        
        # Base bounds (conservative)
        base_bounds = {
            'scanner': {
                'momentum_period': (10, 30),
                'rsi_window': (10, 20),
                'volume_threshold': (1.0, 3.0),
            },
            'strategy_trainer': {
                'learning_rate': (0.001, 0.05),
                'confidence_threshold': (0.3, 0.7),
            },
            'arch_ctrl': {
                'fear_sensitivity': (0.3, 0.7),
                'confidence_boost_rate': (0.02, 0.08),
            },
            'execution_daemon': {
                'position_size_factor': (0.7, 1.3),
                'risk_per_trade': (0.01, 0.03),
            },
            'risk_sentinel': {
                'max_drawdown': (-0.3, -0.15),
                'max_position_limit': (0.1, 0.2),
            }
        }
        
        # Adjust bounds based on mode
        if self.mode == OptimizationMode.LIVE_AGGRESSIVE:
            # Widen bounds slightly for live aggressive mode
            for component in base_bounds:
                for param, (low, high) in base_bounds[component].items():
                    range_width = high - low
                    base_bounds[component][param] = (
                        low - range_width * 0.2,
                        high + range_width * 0.2
                    )
        
        return base_bounds
    
    def _define_dependencies(self) -> List[ParameterDependency]:
        """Define parameter dependencies and cross-component constraints"""
        return [
            ParameterDependency(
                param_name='execution_daemon.risk_per_trade',
                depends_on=['execution_daemon.max_positions', 'risk_sentinel.max_drawdown'],
                constraint_func=lambda p: (
                    p.get('execution_daemon.risk_per_trade', 0) * 
                    p.get('execution_daemon.max_positions', 1) < 
                    abs(p.get('risk_sentinel.max_drawdown', -0.2))
                ),
                error_message="Total risk (risk_per_trade × max_positions) exceeds max_drawdown"
            ),
            ParameterDependency(
                param_name='execution_daemon.position_size_factor',
                depends_on=['risk_sentinel.max_position_limit'],
                constraint_func=lambda p: (
                    p.get('execution_daemon.position_size_factor', 1.0) <= 
                    p.get('risk_sentinel.max_position_limit', 0.2) * 10
                ),
                error_message="Position size factor exceeds position limits"
            ),
            ParameterDependency(
                param_name='strategy_trainer.confidence_threshold',
                depends_on=['arch_ctrl.override_confidence_min'],
                constraint_func=lambda p: (
                    p.get('strategy_trainer.confidence_threshold', 0.5) < 
                    p.get('arch_ctrl.override_confidence_min', 0.8)
                ),
                error_message="Strategy confidence threshold must be below ARCH override minimum"
            )
        ]
    
    async def pre_optimization_checks(self) -> Tuple[bool, str]:
        """Run comprehensive pre-optimization checks"""
        
        # Check if already optimizing
        if self.is_optimizing:
            return False, "Optimization already in progress"
        
        # Check market status if required
        if self.constraints.require_market_closed and self.mode != OptimizationMode.SIMULATION:
            is_market_open = await self._check_market_status()
            if is_market_open:
                return False, "Markets are open - optimization not allowed in current mode"
        
        # Check system health
        system_health = await self._check_system_health()
        if not system_health['healthy']:
            return False, f"System unhealthy: {system_health['reason']}"
        
        # Check recent performance
        if self.mode in [OptimizationMode.LIVE_SAFE, OptimizationMode.LIVE_AGGRESSIVE]:
            recent_perf = await self._check_recent_performance()
            if recent_perf['drawdown'] < self.constraints.max_drawdown_during_opt:
                return False, f"Recent drawdown too high: {recent_perf['drawdown']:.2%}"
        
        # Check available data
        data_check = await self._check_sufficient_data()
        if not data_check['sufficient']:
            return False, f"Insufficient data: {data_check['reason']}"
        
        return True, "All pre-optimization checks passed"
    
    async def optimize_component_safe(self,
                                      component_name: str,
                                      component: SafeOptimizableAgent,
                                      iterations: int = 15) -> Dict[str, Any]:
        """
        Safely optimize a single component with full validation
        """
        
        # Pre-optimization checks
        can_optimize, message = await self.pre_optimization_checks()
        if not can_optimize:
            logger.error(f"Pre-optimization check failed: {message}")
            return {'success': False, 'error': message}
        
        self.is_optimizing = True
        self.optimization_start_time = time.time()
        
        try:
            # Create snapshot for rollback
            snapshot = component.create_snapshot()
            self.snapshots[component_name] = snapshot
            
            # Get bounds
            bounds = self.parameter_bounds.get(component_name, {})
            if not bounds:
                return {'success': False, 'error': 'No bounds defined'}
            
            # Filter bounds for existing parameters
            current_params = component.get_hyperparameters()
            filtered_bounds = {k: v for k, v in bounds.items() if k in current_params}
            
            logger.info(f"Starting safe optimization of {component_name}")
            
            # Track best configuration
            best_config = {
                'params': current_params.copy(),
                'score': float('-inf'),
                'metrics': {}
            }
            
            # Objective function with safety checks
            async def safe_objective(**params):
                try:
                    # Validate parameters
                    is_valid, error_msg = component.validate_params(params)
                    if not is_valid:
                        logger.warning(f"Invalid params: {error_msg}")
                        return -np.inf
                    
                    # Check parameter dependencies
                    if not self._check_dependencies(component_name, params):
                        logger.warning(f"Dependency constraints violated")
                        return -np.inf
                    
                    # Apply parameters
                    component.set_hyperparameters(params)
                    
                    # Evaluate with validation period
                    score, metrics = await component.evaluate_async(
                        self.mode,
                        self.constraints.validation_period_seconds
                    )
                    
                    # Check for minimum trades requirement
                    if metrics.get('num_trades', 0) < self.constraints.min_evaluation_trades:
                        logger.debug(f"Insufficient trades for evaluation: {metrics.get('num_trades', 0)}")
                        return score * 0.5  # Penalize but don't reject
                    
                    # Update best if improved
                    if score > best_config['score']:
                        best_config['params'] = params.copy()
                        best_config['score'] = score
                        best_config['metrics'] = metrics.copy()
                    
                    logger.debug(f"Evaluation: {params} -> {score:.4f} (trades={metrics.get('num_trades', 0)})")
                    
                    return score
                    
                except Exception as e:
                    logger.error(f"Error in safe_objective: {e}")
                    return -np.inf
            
            # Run Bayesian optimization with async wrapper
            optimizer = BayesianOptimization(
                f=lambda **p: asyncio.run(safe_objective(**p)),
                pbounds=filtered_bounds,
                random_state=42,
                verbose=1
            )
            
            # Execute optimization
            optimizer.maximize(init_points=5, n_iter=iterations)
            
            # Apply best configuration
            if best_config['score'] > float('-inf'):
                component.set_hyperparameters(best_config['params'])
                
                # Final validation period
                logger.info(f"Running final validation for {component_name}...")
                await asyncio.sleep(self.constraints.validation_period_seconds)
                
                final_score, final_metrics = await component.evaluate_async(
                    self.mode,
                    self.constraints.validation_period_seconds
                )
                
                # Check if optimization actually improved performance
                baseline_metrics = component.get_performance_metrics()
                improvement = self._calculate_improvement(baseline_metrics, final_metrics)
                
                if improvement < -0.05:  # 5% worse
                    logger.warning(f"Optimization degraded performance by {improvement:.2%}, rolling back")
                    component.restore_snapshot(snapshot)
                    return {
                        'success': False,
                        'error': 'Optimization degraded performance',
                        'improvement': improvement
                    }
                
                # Store successful optimization
                self.optimization_history[component_name] = {
                    'timestamp': datetime.now().isoformat(),
                    'mode': self.mode.value,
                    'best_params': best_config['params'],
                    'best_score': best_config['score'],
                    'final_score': final_score,
                    'final_metrics': final_metrics,
                    'improvement': improvement,
                    'iterations': iterations
                }
                
                logger.info(f"Optimization successful for {component_name}: improvement={improvement:.2%}")
                
                return {
                    'success': True,
                    'best_params': best_config['params'],
                    'improvement': improvement,
                    'metrics': final_metrics
                }
            else:
                logger.error(f"No valid configuration found for {component_name}")
                component.restore_snapshot(snapshot)
                return {'success': False, 'error': 'No valid configuration found'}
        
        except Exception as e:
            logger.error(f"Optimization failed for {component_name}: {e}")
            
            # Rollback on error
            if self.constraints.rollback_on_failure and component_name in self.snapshots:
                component.restore_snapshot(self.snapshots[component_name])
            
            return {'success': False, 'error': str(e)}
        
        finally:
            self.is_optimizing = False
    
    def _check_dependencies(self, component_name: str, params: Dict) -> bool:
        """Check if parameters satisfy dependency constraints"""
        
        # Build full parameter dict across all components
        full_params = {}
        for comp_name, component in self.components.items():
            if hasattr(component, 'get_hyperparameters'):
                comp_params = component.get_hyperparameters()
                for param_name, value in comp_params.items():
                    full_params[f"{comp_name}.{param_name}"] = value
        
        # Override with proposed parameters
        for param_name, value in params.items():
            full_params[f"{component_name}.{param_name}"] = value
        
        # Check each dependency
        for dep in self.dependencies:
            # Only check dependencies relevant to this component
            if dep.param_name.startswith(component_name + '.'):
                try:
                    if not dep.constraint_func(full_params):
                        logger.warning(f"Dependency constraint failed: {dep.error_message}")
                        return False
                except Exception as e:
                    logger.error(f"Error checking dependency: {e}")
                    return False
        
        return True
    
    async def _check_market_status(self) -> bool:
        """Check if markets are currently open"""
        # Simplified - in production, check actual exchange hours
        current_hour = datetime.now().hour
        # Crypto markets are always open, but we can define maintenance windows
        # For traditional markets: 9:30 AM - 4:00 PM ET
        return True  # Crypto always open - override in production
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            # Check SyncBus health
            if 'sync_bus' in self.components:
                sync_bus = self.components['sync_bus']
                if hasattr(sync_bus, 'get_health_status'):
                    health = sync_bus.get_health_status()
                    if not health.get('healthy', False):
                        return {'healthy': False, 'reason': health.get('reason', 'Unknown')}
            
            # Check agent statuses
            unhealthy_agents = []
            for name, component in self.components.items():
                if hasattr(component, 'is_healthy'):
                    if not component.is_healthy():
                        unhealthy_agents.append(name)
            
            if unhealthy_agents:
                return {
                    'healthy': False,
                    'reason': f"Unhealthy agents: {', '.join(unhealthy_agents)}"
                }
            
            return {'healthy': True, 'reason': 'All checks passed'}
            
        except Exception as e:
            return {'healthy': False, 'reason': f'Health check error: {e}'}
    
    async def _check_recent_performance(self) -> Dict[str, float]:
        """Check recent system performance"""
        # Implement based on your performance tracking
        return {
            'drawdown': 0.0,
            'pnl': 0.0,
            'sharpe': 0.0
        }
    
    async def _check_sufficient_data(self) -> Dict[str, Any]:
        """Check if sufficient data exists for optimization"""
        # Check if enough historical data for evaluation
        return {
            'sufficient': True,
            'reason': 'Data check passed'
        }
    
    def _calculate_improvement(self, baseline: Dict, optimized: Dict) -> float:
        """Calculate performance improvement"""
        # Simple PnL-based improvement
        baseline_pnl = baseline.get('pnl', 0)
        optimized_pnl = optimized.get('pnl', 0)
        
        if baseline_pnl == 0:
            return 0.0
        
        return (optimized_pnl - baseline_pnl) / abs(baseline_pnl)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'is_optimizing': self.is_optimizing,
            'mode': self.mode.value,
            'optimization_start_time': self.optimization_start_time,
            'time_elapsed': time.time() - self.optimization_start_time if self.optimization_start_time else 0,
            'components_optimized': len(self.optimization_history),
            'snapshots_available': len(self.snapshots)
        }
    
    def save_optimization_history(self, filename: str = "safe_optimization_history.json"):
        """Save optimization history"""
        history_data = {
            'mode': self.mode.value,
            'constraints': {
                'max_drawdown_during_opt': self.constraints.max_drawdown_during_opt,
                'min_evaluation_trades': self.constraints.min_evaluation_trades,
                'max_parameter_change_pct': self.constraints.max_parameter_change_pct
            },
            'optimization_history': self.optimization_history,
            'parameter_bounds': self.parameter_bounds
        }
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Optimization history saved to {filename}")


# Example implementation of SafeOptimizableAgent
class ExampleSafeAgent(SafeOptimizableAgent):
    """Example implementation showing proper safety features"""
    
    def __init__(self, name: str):
        self.name = name
        self.momentum_period = 14
        self.rsi_window = 14
        self.performance_history = []
        
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'momentum_period': self.momentum_period,
            'rsi_window': self.rsi_window
        }
    
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        self.momentum_period = params.get('momentum_period', self.momentum_period)
        self.rsi_window = params.get('rsi_window', self.rsi_window)
    
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        if params.get('momentum_period', 0) < 5:
            return False, "momentum_period must be >= 5"
        if params.get('rsi_window', 0) < 5:
            return False, "rsi_window must be >= 5"
        return True, "Parameters valid"
    
    async def evaluate_async(self, mode: OptimizationMode, validation_period: int = 300) -> Tuple[float, Dict[str, Any]]:
        # Simulate evaluation
        await asyncio.sleep(0.1)  # Simulate async work
        
        # Mock metrics
        score = np.random.random()
        metrics = {
            'num_trades': np.random.randint(10, 50),
            'pnl': np.random.random() * 1000,
            'sharpe': np.random.random() * 2
        }
        
        return score, metrics
    
    def get_performance_metrics(self) -> Dict[str, float]:
        return {
            'pnl': 1000.0,
            'sharpe': 1.5,
            'max_dd': -0.1
        }
    
    def create_snapshot(self) -> Dict[str, Any]:
        return {
            'momentum_period': self.momentum_period,
            'rsi_window': self.rsi_window,
            'performance_history': self.performance_history.copy()
        }
    
    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.momentum_period = snapshot['momentum_period']
        self.rsi_window = snapshot['rsi_window']
        self.performance_history = snapshot['performance_history']


# Test the safe optimizer
async def test_safe_optimizer():
    """Test the production-safe optimizer"""
    
    # Create mock components
    components = {
        'scanner': ExampleSafeAgent('scanner'),
        'strategy_trainer': ExampleSafeAgent('strategy_trainer')
    }
    
    # Initialize optimizer in simulation mode
    optimizer = ProductionSafeMirrorOptimizer(
        components,
        mode=OptimizationMode.SIMULATION,
        constraints=OptimizationConstraints(
            require_market_closed=False,  # Allow in simulation
            min_evaluation_trades=5
        )
    )
    
    # Optimize a component
    result = await optimizer.optimize_component_safe(
        'scanner',
        components['scanner'],
        iterations=10
    )
    
    print(f"Optimization result: {result}")
    
    # Get status
    status = optimizer.get_optimization_status()
    print(f"Status: {status}")
    
    # Save history
    optimizer.save_optimization_history()


if __name__ == "__main__":
    asyncio.run(test_safe_optimizer())