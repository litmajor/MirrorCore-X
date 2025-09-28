
from bayes_opt import BayesianOptimization
from typing import Dict, Callable, Optional, Any
import numpy as np
import logging
from abc import ABC, abstractmethod
import json

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
    def evaluate(self) -> float:
        """Evaluate the agent's performance. Higher values are better."""
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Optional: Validate parameters before setting them."""
        return True

class MirrorOptimizerAgent:
    """Bayesian optimization engine for trading agents."""
    
    def __init__(self, agents: Dict[str, OptimizableAgent]):
        """
        Initialize the optimizer with a dictionary of agents.
        
        Args:
            agents: Dictionary mapping agent names to OptimizableAgent instances
        """
        self.agents = agents
        self.optimization_history = {}
        
    def optimize_agent(self, 
                      agent_name: str, 
                      bounds: Dict[str, tuple], 
                      iterations: int = 15,
                      init_points: int = 5,
                      acq: str = 'ucb',
                      kappa: float = 2.576,
                      xi: float = 0.0) -> Dict[str, Any]:
        """
        Optimize an agent's hyperparameters using Bayesian optimization.
        
        Args:
            agent_name: Name of the agent to optimize
            bounds: Dictionary of parameter bounds in format {'param': (min, max)}
            iterations: Number of optimization iterations
            init_points: Number of random initialization points
            acq: Acquisition function ('ucb', 'ei', or 'poi')
            kappa: Kappa parameter for UCB acquisition function
            xi: Xi parameter for EI and POI acquisition functions
            
        Returns:
            Dictionary containing the best parameters found
        """
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}")
        
        logger.info(f"Starting optimization for agent '{agent_name}'")
        logger.info(f"Bounds: {bounds}")
        logger.info(f"Iterations: {iterations}, Init points: {init_points}")
        
        # Store original parameters for potential rollback
        original_params = agent.get_hyperparameters()
        
        # Dynamically create a target function with the correct signature for BayesianOptimization
        param_names = list(bounds.keys())
        def target_function(**params):
            try:
                # Only keep parameters that are in bounds
                params = {k: params[k] for k in param_names if k in params}
                if not agent.validate_params(params):
                    logger.warning(f"Invalid parameters: {params}")
                    return -np.inf
                agent.set_hyperparameters(params)
                score = agent.evaluate()
                logger.info(f"Params: {params} -> Score: {score:.4f}")
                return score
            except Exception as e:
                logger.error(f"Error evaluating parameters {params}: {str(e)}")
                return -np.inf
        
        # Set up Bayesian optimization
        optimizer = BayesianOptimization(
            f=target_function,
            pbounds=bounds,
            random_state=42,
            verbose=1
        )
        
        # Configure acquisition function
        optimizer.set_gp_params(alpha=1e-3)
        
        try:
            # Run optimization
            optimizer.maximize(
                init_points=init_points,
                n_iter=iterations
            )
            
            # Get best parameters
            if optimizer.max is not None:
                best_params = optimizer.max.get("params", {})
                best_score = optimizer.max.get("target", float('-inf'))
            else:
                logger.error(f"No optimization results found for '{agent_name}'.")
                best_params = {}
                best_score = float('-inf')
            
            logger.info(f"Optimization complete for '{agent_name}'")
            logger.info(f"Best score: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            # Set the best parameters if available
            if best_params:
                agent.set_hyperparameters(best_params)
            
            # Store optimization history
            self.optimization_history[agent_name] = {
                'best_params': best_params,
                'best_score': best_score,
                'original_params': original_params,
                'bounds': bounds,
                'iterations': iterations,
                'all_results': [
                    {'params': dict(res['params']), 'target': res['target']} 
                    for res in optimizer.res
                ]
            }
            
            return best_params
            
        except Exception as e:
            logger.error(f"Optimization failed for '{agent_name}': {str(e)}")
            # Rollback to original parameters
            agent.set_hyperparameters(original_params)
            raise
    
    def optimize_all_agents(self, 
                           bounds_dict: Dict[str, Dict[str, tuple]], 
                           iterations: int = 15) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all agents sequentially.
        
        Args:
            bounds_dict: Dictionary mapping agent names to their parameter bounds
            iterations: Number of optimization iterations per agent
            
        Returns:
            Dictionary mapping agent names to their best parameters
        """
        results = {}
        
        for agent_name in self.agents.keys():
            if agent_name in bounds_dict:
                try:
                    best_params = self.optimize_agent(agent_name, bounds_dict[agent_name], iterations)
                    results[agent_name] = best_params
                except Exception as e:
                    logger.error(f"Failed to optimize agent '{agent_name}': {str(e)}")
                    results[agent_name] = None
            else:
                logger.warning(f"No bounds specified for agent '{agent_name}', skipping...")
        
        return results
    
    def get_optimization_history(self, agent_name: Optional[str] = None) -> Dict:
        """Get optimization history for a specific agent or all agents."""
        if agent_name:
            return self.optimization_history.get(agent_name, {})
        return self.optimization_history
    
    def save_results(self, filename: str) -> None:
        """Save optimization results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)
        logger.info(f"Results saved to {filename}")
    
    def load_results(self, filename: str) -> None:
        """Load optimization results from a JSON file."""
        with open(filename, 'r') as f:
            self.optimization_history = json.load(f)
        logger.info(f"Results loaded from {filename}")




class OracleAgent(OptimizableAgent):
    """Example oracle agent for demonstration."""
    
    def __init__(self):
        self.prediction_window = 5
        self.confidence_threshold = 0.7
        self.model_complexity = 10
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "prediction_window": self.prediction_window,
            "confidence_threshold": self.confidence_threshold,
            "model_complexity": self.model_complexity
        }
    
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        self.prediction_window = int(params.get("prediction_window", self.prediction_window))
        self.confidence_threshold = float(params.get("confidence_threshold", self.confidence_threshold))
        self.model_complexity = int(params.get("model_complexity", self.model_complexity))
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        pred_window = params.get("prediction_window", self.prediction_window)
        conf_thresh = params.get("confidence_threshold", self.confidence_threshold)
        complexity = params.get("model_complexity", self.model_complexity)
        
        return (1 <= pred_window <= 20 and 
                0.1 <= conf_thresh <= 0.95 and 
                1 <= complexity <= 50)
    
    def evaluate(self) -> float:
        """Mock oracle evaluation."""
        params = self.get_hyperparameters()
        
        # Simulate accuracy with optimal values around:
        # prediction_window: 7, confidence_threshold: 0.8, model_complexity: 15
        window_score = 90 - abs(params["prediction_window"] - 7) * 3
        conf_score = 90 - abs(params["confidence_threshold"] - 0.8) * 100
        complexity_score = 90 - abs(params["model_complexity"] - 15) * 2
        
        noise = np.random.normal(0, 1)
        return max(0, (window_score + conf_score + complexity_score) / 3 + noise)


# Usage example
# Usage example

if __name__ == "__main__":
    # Fully wired: Optimize MomentumScanner with real data
    import ccxt.pro as ccxt_async
    from scanner import MomentumScanner, get_dynamic_config
    import asyncio

    # Initialize exchange (choose your preferred exchange)
    exchange = ccxt_async.binance({
        'enableRateLimit': True,
        'rateLimit': 100,
        'timeout': 30000,
    })

    # Load config using the dynamic loader
    config = get_dynamic_config(market_type='crypto')

    # Initialize scanner
    scanner = MomentumScanner(exchange, config=config, market_type='crypto', quote_currency='USDT', min_volume_usd=1_000_000, top_n=30)
    agents = {"scanner": scanner}
    optimizer = MirrorOptimizerAgent(agents)

    bounds = {
        "momentum_period": (5, 50),
        "rsi_window": (5, 50),
        "volume_threshold": (0.5, 10.0)
    }

    print("=== Optimizing MomentumScanner with real backtest ===")
    best_params = optimizer.optimize_agent("scanner", bounds, iterations=60, init_points=12)
    print(f"Best scanner params: {best_params}")
