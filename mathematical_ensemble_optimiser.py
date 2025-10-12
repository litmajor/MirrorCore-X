"""
MirrorCore-X: Mathematical Ensemble Weight Optimizer
Production-ready QP solver with Ledoit-Wolf shrinkage, turnover penalty, and regime awareness.

Author: MirrorCore-X Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not available, falling back to scipy")

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


class OptimizationObjective(Enum):
    """Optimization objective types"""
    MEAN_VARIANCE = "mean_variance"
    MAX_SHARPE = "max_sharpe"
    SHARPE_RATIO = "sharpe_ratio"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    ROBUST_CVaR = "robust_cvar"


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    MIXED = "mixed"


@dataclass
class OptimizationConfig:
    """Configuration for ensemble optimization"""
    # Risk parameters
    lambda_risk: float = 100.0  # Risk aversion (higher = more conservative)
    eta_turnover: float = 0.05  # Turnover penalty
    gamma_reg: float = 1e-5  # Covariance regularization

    # Constraints
    max_weight: float = 0.25  # Maximum single strategy weight
    min_weight: float = 0.0  # Minimum weight (0 = can be zero)
    total_weight: float = 1.0  # Sum of all weights

    # Estimation parameters
    returns_window: int = 60  # Rolling window for returns estimation
    ewma_alpha: float = 0.1  # EWMA decay for returns
    use_shrinkage: bool = True  # Use Ledoit-Wolf shrinkage

    # Robustness
    max_condition_number: float = 1e6  # Maximum condition number for Sigma
    fallback_to_equal: bool = True  # Fallback to equal weights if solver fails

    # Other
    risk_free_rate: float = 0.0  # Per-period risk-free rate
    rebalance_threshold: float = 0.05  # Trigger rebalance if weight changes > threshold


@dataclass
class OptimizationResult:
    """Result from optimization"""
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    turnover: float
    condition_number: float
    solver_status: str
    regime: Optional[str] = None
    confidence: float = 1.0


class EnsembleOptimizer:
    """
    Mathematical optimization of ensemble strategy weights.

    Solves the convex quadratic program:
        max_w  w'μ - (λ/2)w'Σw - η||w - w_prev||²
        s.t.   sum(w) = 1, w_min <= w <= w_max

    Features:
    - Ledoit-Wolf covariance shrinkage
    - Turnover penalty for stability
    - Multiple objective functions
    - Regime-aware parameter adjustment
    - Robust fallback mechanisms
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.w_prev = None
        self.mu_history = []
        self.returns_history = []
        self.regime_params = self._initialize_regime_params()

    def _initialize_regime_params(self) -> Dict[str, Dict[str, float]]:
        """Initialize regime-specific parameter adjustments"""
        return {
            MarketRegime.TRENDING.value: {
                'lambda_multiplier': 0.8,  # Lower risk aversion in trends
                'eta_multiplier':
                1.2,  # Higher turnover penalty (stay in trend)
                'max_weight': 0.30
            },
            MarketRegime.RANGING.value: {
                'lambda_multiplier': 1.0,  # Normal risk aversion
                'eta_multiplier': 0.8,  # Lower turnover (rebalance more)
                'max_weight': 0.25
            },
            MarketRegime.VOLATILE.value: {
                'lambda_multiplier': 1.5,  # Higher risk aversion
                'eta_multiplier':
                1.5,  # Much higher turnover penalty (stability)
                'max_weight': 0.20
            },
            MarketRegime.MIXED.value: {
                'lambda_multiplier': 1.2,
                'eta_multiplier': 1.0,
                'max_weight': 0.25
            }
        }

    def estimate_parameters(
            self,
            returns_df: pd.DataFrame,
            method: str = 'ewma') -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate μ (expected returns) and Σ (covariance matrix).

        Args:
            returns_df: DataFrame with strategy returns (rows=time, cols=strategies)
            method: 'ewma', 'rolling', or 'shrinkage' for μ estimation

        Returns:
            (μ, Σ): Expected returns vector and covariance matrix
        """
        returns = returns_df.values

        # Estimate μ (expected returns)
        if method == 'ewma':
            mu = self._ewma_returns(returns, self.config.ewma_alpha)
        elif method == 'rolling':
            mu = np.mean(returns[-self.config.returns_window:], axis=0)
        elif method == 'shrinkage':
            mu_sample = np.mean(returns, axis=0)
            mu_prior = np.mean(mu_sample)  # Grand mean as prior
            mu = 0.7 * mu_sample + 0.3 * mu_prior  # Shrinkage
        else:
            raise ValueError(f"Unknown method: {method}")

        # Estimate Σ (covariance)
        if self.config.use_shrinkage:
            try:
                lw = LedoitWolf()
                Sigma = lw.fit(
                    returns[-self.config.returns_window:]).covariance_
            except Exception as e:
                warnings.warn(
                    f"Ledoit-Wolf failed: {e}, using sample covariance")
                Sigma = np.cov(returns[-self.config.returns_window:].T)
        else:
            Sigma = np.cov(returns[-self.config.returns_window:].T)

        # Regularization to ensure positive definiteness
        Sigma += self.config.gamma_reg * np.eye(len(mu))

        # Check condition number
        cond_num = np.linalg.cond(Sigma)
        if cond_num > self.config.max_condition_number:
            warnings.warn(
                f"High condition number: {cond_num:.2e}, increasing regularization"
            )
            Sigma += (self.config.gamma_reg * 10) * np.eye(len(mu))

        return mu, Sigma

    def _ewma_returns(self, returns: np.ndarray, alpha: float) -> np.ndarray:
        """Calculate EWMA of returns"""
        n_strategies = returns.shape[1]
        mu = np.zeros(n_strategies)

        for i in range(n_strategies):
            mu[i] = returns[0, i]
            for t in range(1, len(returns)):
                mu[i] = alpha * returns[t, i] + (1 - alpha) * mu[i]

        return mu

    def optimize(self,
                 mu: np.ndarray,
                 Sigma: np.ndarray,
                 regime: Optional[str] = None,
                 objective: OptimizationObjective = OptimizationObjective.
                 MEAN_VARIANCE,
                 solver: str = 'auto') -> OptimizationResult:
        """
        Optimize ensemble weights.

        Args:
            mu: Expected returns vector (n_strategies,)
            Sigma: Covariance matrix (n_strategies, n_strategies)
            regime: Current market regime for parameter adjustment
            objective: Optimization objective
            solver: 'cvxpy', 'scipy', or 'auto'

        Returns:
            OptimizationResult with optimal weights and diagnostics
        """
        n = len(mu)

        # Initialize previous weights if first run
        if self.w_prev is None:
            self.w_prev = np.ones(n) / n

        # Adjust parameters for regime
        lambda_risk = self.config.lambda_risk
        eta_turnover = self.config.eta_turnover
        max_weight = self.config.max_weight

        if regime:
            params = self.regime_params.get(regime, {})
            lambda_risk *= params.get('lambda_multiplier', 1.0)
            eta_turnover *= params.get('eta_multiplier', 1.0)
            max_weight = params.get('max_weight', self.config.max_weight)

        # Choose solver
        if solver == 'auto':
            solver = 'cvxpy' if CVXPY_AVAILABLE else 'scipy'

        try:
            if solver == 'cvxpy' and CVXPY_AVAILABLE:
                result = self._optimize_cvxpy(mu, Sigma, lambda_risk,
                                              eta_turnover, max_weight,
                                              objective)
            else:
                result = self._optimize_scipy(mu, Sigma, lambda_risk,
                                              eta_turnover, max_weight,
                                              objective)

            # Calculate diagnostics
            w_opt = result['weights']
            port_return = w_opt.T @ (mu - self.config.risk_free_rate)
            port_vol = np.sqrt(w_opt.T @ Sigma @ w_opt)
            sharpe = port_return / port_vol if port_vol > 0 else 0
            turnover = np.sum(np.abs(w_opt - self.w_prev))
            cond_num = np.linalg.cond(Sigma)

            # Update previous weights
            self.w_prev = w_opt

            return OptimizationResult(weights=w_opt,
                                      expected_return=port_return,
                                      expected_volatility=port_vol,
                                      sharpe_ratio=sharpe,
                                      turnover=turnover,
                                      condition_number=cond_num,
                                      solver_status=result['status'],
                                      regime=regime,
                                      confidence=self._calculate_confidence(
                                          Sigma, mu))

        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            if self.config.fallback_to_equal:
                return self._fallback_weights(mu, Sigma, regime)
            raise

    def _optimize_cvxpy(self, mu: np.ndarray, Sigma: np.ndarray,
                        lambda_risk: float, eta_turnover: float,
                        max_weight: float,
                        objective: OptimizationObjective) -> Dict:
        """Solve using CVXPY (preferred method)"""
        n = len(mu)
        w = cp.Variable(n)

        ret = mu - self.config.risk_free_rate

        if objective == OptimizationObjective.MEAN_VARIANCE:
            # Maximize: w'μ - (λ/2)w'Σw - η||w - w_prev||²
            obj = cp.Maximize(w @ ret -
                              0.5 * lambda_risk * cp.quad_form(w, Sigma) -
                              eta_turnover * cp.sum_squares(w - self.w_prev))
        elif objective == OptimizationObjective.MIN_VARIANCE:
            # Minimize variance only
            obj = cp.Minimize(cp.quad_form(w, Sigma))
        elif objective == OptimizationObjective.MAX_SHARPE or objective == OptimizationObjective.SHARPE_RATIO:
            # Approximate max Sharpe via regularized objective
            obj = cp.Maximize(w @ ret - 0.5 * cp.quad_form(w, Sigma))
        else:
            raise ValueError(
                f"Objective {objective} not implemented for cvxpy")

        constraints = [
            cp.sum(w) == self.config.total_weight, w >= self.config.min_weight,
            w <= max_weight
        ]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver status: {prob.status}")

        return {'weights': w.value, 'status': prob.status}

    def _optimize_scipy(self, mu: np.ndarray, Sigma: np.ndarray,
                        lambda_risk: float, eta_turnover: float,
                        max_weight: float,
                        objective: OptimizationObjective) -> Dict:
        """Solve using SciPy (fallback method)"""
        n = len(mu)
        ret = mu - self.config.risk_free_rate

        def objective_func(w):
            """Negative of objective to minimize"""
            portfolio_return = w @ ret
            portfolio_var = w @ Sigma @ w
            turnover_penalty = eta_turnover * np.sum((w - self.w_prev)**2)

            return -portfolio_return + 0.5 * lambda_risk * portfolio_var + turnover_penalty

        def objective_grad(w):
            """Gradient of objective"""
            grad_return = -ret
            grad_var = lambda_risk * (Sigma @ w)
            grad_turnover = 2 * eta_turnover * (w - self.w_prev)

            return grad_return + grad_var + grad_turnover

        constraints = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - self.config.total_weight
        }]

        bounds = [(self.config.min_weight, max_weight)] * n

        # Use previous weights as initial guess
        x0 = self.w_prev

        result = minimize(objective_func,
                          x0,
                          method='SLSQP',
                          jac=objective_grad,
                          bounds=bounds,
                          constraints=constraints,
                          options={
                              'maxiter': 1000,
                              'ftol': 1e-9
                          })

        if not result.success:
            warnings.warn(f"SciPy optimization warning: {result.message}")

        return {
            'weights': result.x,
            'status': 'optimal' if result.success else 'warning'
        }

    def _fallback_weights(self, mu: np.ndarray, Sigma: np.ndarray,
                          regime: Optional[str]) -> OptimizationResult:
        """Fallback to equal or risk-parity weights"""
        n = len(mu)

        # Try risk parity first
        try:
            w_opt = self._risk_parity_weights(Sigma)
        except:
            # Last resort: equal weights
            w_opt = np.ones(n) / n

        port_return = w_opt @ (mu - self.config.risk_free_rate)
        port_vol = np.sqrt(w_opt @ Sigma @ w_opt)
        sharpe = port_return / port_vol if port_vol > 0 else 0

        return OptimizationResult(weights=w_opt,
                                  expected_return=port_return,
                                  expected_volatility=port_vol,
                                  sharpe_ratio=sharpe,
                                  turnover=np.sum(np.abs(w_opt - self.w_prev))
                                  if self.w_prev is not None else 0,
                                  condition_number=np.linalg.cond(Sigma),
                                  solver_status='fallback',
                                  regime=regime,
                                  confidence=0.5)

    def _risk_parity_weights(self,
                             Sigma: np.ndarray,
                             max_iter: int = 100) -> np.ndarray:
        """
        Calculate risk parity weights (equal risk contribution).
        Uses iterative algorithm.
        """
        n = Sigma.shape[0]
        w = np.ones(n) / n  # Start with equal weights

        for _ in range(max_iter):
            # Calculate marginal risk contributions
            port_vol = np.sqrt(w @ Sigma @ w)
            mrc = (Sigma @ w) / port_vol

            # Update weights inversely proportional to marginal risk
            w_new = 1.0 / mrc
            w_new /= np.sum(w_new)  # Normalize

            # Check convergence
            if np.max(np.abs(w_new - w)) < 1e-6:
                break

            w = 0.5 * w + 0.5 * w_new  # Damped update

        return w

    def _calculate_confidence(self, Sigma: np.ndarray,
                              mu: np.ndarray) -> float:
        """
        Calculate confidence score for optimization results.
        Based on condition number and estimation uncertainty.
        """
        cond_num = np.linalg.cond(Sigma)

        # Penalize high condition numbers
        cond_score = 1.0 / (1.0 + np.log10(cond_num / 100))

        # Penalize low expected returns
        mu_score = np.tanh(np.mean(np.abs(mu)) * 100)

        return 0.7 * cond_score + 0.3 * mu_score

    def optimize_with_resampling(
            self,
            returns_df: pd.DataFrame,
            regime: Optional[str] = None,
            n_samples: int = 100,
            sample_fraction: float = 0.8) -> OptimizationResult:
        """
        Robust optimization via resampling (Monte Carlo).
        Averages weights across multiple bootstrap samples.
        """
        n_periods = len(returns_df)
        n_sample_periods = int(n_periods * sample_fraction)

        weights_samples = []

        for _ in range(n_samples):
            # Bootstrap sample
            sample_idx = np.random.choice(n_periods,
                                          n_sample_periods,
                                          replace=True)
            sample_returns = returns_df.iloc[sample_idx]

            # Estimate and optimize
            mu, Sigma = self.estimate_parameters(sample_returns)
            result = self.optimize(mu, Sigma, regime)

            if result.solver_status in ['optimal', 'optimal_inaccurate']:
                weights_samples.append(result.weights)

        # Average weights across samples
        if len(weights_samples) > 0:
            w_mean = np.mean(weights_samples, axis=0)
            w_std = np.std(weights_samples, axis=0)

            # Final estimation for diagnostics
            mu, Sigma = self.estimate_parameters(returns_df)

            port_return = w_mean @ (mu - self.config.risk_free_rate)
            port_vol = np.sqrt(w_mean @ Sigma @ w_mean)
            sharpe = port_return / port_vol if port_vol > 0 else 0

            return OptimizationResult(
                weights=w_mean,
                expected_return=port_return,
                expected_volatility=port_vol,
                sharpe_ratio=sharpe,
                turnover=np.sum(np.abs(w_mean - self.w_prev))
                if self.w_prev is not None else 0,
                condition_number=np.linalg.cond(Sigma),
                solver_status='resampled',
                regime=regime,
                confidence=1.0 -
                np.mean(w_std)  # Lower std = higher confidence
            )
        else:
            # Fallback if all samples failed
            return self._fallback_weights(mu, Sigma, regime)


# ============================================================================
# USAGE EXAMPLE & INTEGRATION
# ============================================================================


def example_usage():
    """Demonstrate usage with synthetic data"""

    # Simulate 19 strategies (11 core + 8 new) with 90 days of returns
    np.random.seed(42)
    n_strategies = 19
    n_days = 90

    strategy_names = [
        'UT_BOT', 'GRADIENT_TREND', 'VOLUME_SR', 'MEAN_REVERSION',
        'MOMENTUM_BREAKOUT', 'VOLATILITY_REGIME', 'PAIRS_TRADING',
        'ANOMALY_DETECTION', 'SENTIMENT_MOMENTUM', 'REGIME_CHANGE', 'RL_AGENT',
        'LIQUIDITY_FLOW', 'FRACTAL_GEOMETRY', 'CORRELATION_MATRIX',
        'MICROSTRUCTURE', 'OPTION_FLOW', 'ENTROPY_MEASURE', 'BAYESIAN_UPDATER',
        'CAUSALITY_DETECTOR'
    ]

    # Generate correlated returns
    true_mu = np.random.uniform(-0.001, 0.003, n_strategies)
    true_Sigma = np.random.randn(n_strategies, n_strategies) * 0.01
    true_Sigma = true_Sigma @ true_Sigma.T + 0.0001 * np.eye(n_strategies)

    returns = np.random.multivariate_normal(true_mu, true_Sigma, n_days)
    returns_df = pd.DataFrame(returns, columns=strategy_names)

    # Initialize optimizer
    config = OptimizationConfig(lambda_risk=100.0,
                                eta_turnover=0.05,
                                max_weight=0.25,
                                returns_window=60,
                                ewma_alpha=0.1)

    optimizer = EnsembleOptimizer(config)

    # Estimate parameters
    mu, Sigma = optimizer.estimate_parameters(returns_df, method='ewma')

    print("=" * 80)
    print("MIRRORCORE-X ENSEMBLE OPTIMIZATION")
    print("=" * 80)
    print(f"\nEstimated Parameters:")
    print(f"  Expected Returns (μ): {mu}")
    print(f"  Covariance condition number: {np.linalg.cond(Sigma):.2e}")

    # Optimize for different regimes
    for regime in [
            MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.VOLATILE
    ]:
        result = optimizer.optimize(
            mu,
            Sigma,
            regime=regime.value,
            objective=OptimizationObjective.MEAN_VARIANCE)

        print(f"\n{'='*80}")
        print(f"Regime: {regime.value.upper()}")
        print(f"{'='*80}")
        print(f"Expected Return: {result.expected_return*100:.3f}%")
        print(f"Expected Volatility: {result.expected_volatility*100:.3f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"Turnover: {result.turnover:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Solver Status: {result.solver_status}")
        print(f"\nTop 5 Weights:")

        top_idx = np.argsort(result.weights)[-5:][::-1]
        for idx in top_idx:
            print(
                f"  {strategy_names[idx]:20s}: {result.weights[idx]*100:6.2f}%"
            )

    # Robust optimization via resampling
    print(f"\n{'='*80}")
    print("ROBUST OPTIMIZATION (Resampling)")
    print(f"{'='*80}")

    result_robust = optimizer.optimize_with_resampling(
        returns_df, regime=MarketRegime.MIXED.value, n_samples=50)

    print(f"Sharpe Ratio: {result_robust.sharpe_ratio:.3f}")
    print(f"Confidence: {result_robust.confidence:.3f}")
    print(f"\nTop 5 Robust Weights:")

    top_idx = np.argsort(result_robust.weights)[-5:][::-1]
    for idx in top_idx:
        print(
            f"  {strategy_names[idx]:20s}: {result_robust.weights[idx]*100:6.2f}%"
        )


if __name__ == "__main__":
    example_usage()
