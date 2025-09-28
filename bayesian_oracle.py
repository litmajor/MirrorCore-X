
import numpy as np
from scipy.stats import beta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging" 
    VOLATILE = "volatile"
    LOW_VOLUME = "low_volume"

@dataclass
class MarketContext:
    """Current market conditions for belief updating"""
    regime: MarketRegime
    volatility_percentile: float
    volume_percentile: float
    trend_strength: float
    timestamp: datetime

class BeliefTracker:
    """Bayesian belief system for trading strategy components"""

    def __init__(self, prior_successes: float = 1.0, prior_failures: float = 1.0, decay_factor: float = 0.95):
        self.alpha = prior_successes  # Successes + prior
        self.beta = prior_failures    # Failures + prior
        self.total_observations = 0
        self.last_updated = None
        self.decay_factor = decay_factor

    def set_decay_factor(self, decay: float):
        self.decay_factor = decay

    def get_decay_factor(self) -> float:
        return self.decay_factor

    def update(self, success: bool, weight: float = 1.0):
        """Update beliefs with new evidence"""
        if success:
            self.alpha += weight
        else:
            self.beta += weight
        
        self.total_observations += 1
        self.last_updated = datetime.now()
    
    def probability(self) -> float:
        """Current belief probability (expected value)"""
        return self.alpha / (self.alpha + self.beta)
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Credible interval for the belief"""
        alpha_level = (1 - confidence) / 2
        dist = beta(self.alpha, self.beta)
        return dist.ppf(alpha_level), dist.ppf(1 - alpha_level)
    
    def uncertainty(self) -> float:
        """Measure of uncertainty (higher = more uncertain)"""
        return np.sqrt(self.alpha * self.beta / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1)))
    
    def effective_sample_size(self) -> float:
        """How much evidence we effectively have"""
        return self.alpha + self.beta - 2

class StrategyBelief:
    """Belief system for a specific trading strategy component"""
    
    def __init__(self, name: str, component_type: str):
        self.name = name
        self.component_type = component_type
        
        # Context-specific beliefs
        self.regime_beliefs: Dict[MarketRegime, BeliefTracker] = {
            regime: BeliefTracker() for regime in MarketRegime
        }
        
        # Overall belief (regime-agnostic)
        self.overall_belief = BeliefTracker()
        
        # Performance tracking
        self.recent_performance: List[Tuple[bool, MarketContext, datetime]] = []
        self.decay_factor = 0.95  # Exponential decay for older observations
        
    def update_belief(self, success: bool, context: MarketContext, confidence: float = 1.0):
        """Update beliefs given new market outcome"""
        
        # Weight based on confidence and recency
        weight = confidence
        
        # Update regime-specific belief
        self.regime_beliefs[context.regime].update(success, weight)
        
        # Update overall belief
        self.overall_belief.update(success, weight)
        
        # Track recent performance
        self.recent_performance.append((success, context, datetime.now()))
        
        # Keep only recent observations (last 100)
        if len(self.recent_performance) > 100:
            self.recent_performance = self.recent_performance[-100:]
    
    def get_contextual_probability(self, context: MarketContext) -> float:
        """Get probability of success given current market context"""
        regime_prob = self.regime_beliefs[context.regime].probability()
        overall_prob = self.overall_belief.probability()
        
        # Weight between regime-specific and overall based on regime evidence
        regime_evidence = self.regime_beliefs[context.regime].effective_sample_size()
        
        if regime_evidence < 5:  # Not enough regime-specific data
            return overall_prob
        else:
            # Blend regime-specific and overall beliefs
            regime_weight = min(0.8, regime_evidence / (regime_evidence + 10))
            return regime_weight * regime_prob + (1 - regime_weight) * overall_prob
    
    def get_uncertainty(self, context: MarketContext) -> float:
        """Get uncertainty for current context"""
        return self.regime_beliefs[context.regime].uncertainty()
    
    def decay_old_beliefs(self, days_threshold: int = 30):
        """Apply time decay to old beliefs"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        for regime_belief in self.regime_beliefs.values():
            if regime_belief.last_updated and regime_belief.last_updated < cutoff_date:
                # Gradually decay confidence in old beliefs
                total = regime_belief.alpha + regime_belief.beta
                regime_belief.alpha = 1 + (regime_belief.alpha - 1) * self.decay_factor
                regime_belief.beta = 1 + (regime_belief.beta - 1) * self.decay_factor

class MirrorCoreBayesianOracle:
    """Main belief system orchestrator for MirrorCore"""
    
    def __init__(self):
        self.strategy_beliefs: Dict[str, StrategyBelief] = {}
        self.correlation_beliefs: Dict[Tuple[str, str], BeliefTracker] = {}
        self.market_regime_detector = MarketRegimeDetector()
        
    def register_strategy_component(self, name: str, component_type: str):
        """Register a new strategy component for belief tracking"""
        self.strategy_beliefs[name] = StrategyBelief(name, component_type)
    
    def update_strategy_performance(self, strategy_name: str, success: bool, 
                                  market_data: pd.DataFrame, confidence: float = 1.0):
        """Update beliefs about a strategy's performance"""
        
        if strategy_name not in self.strategy_beliefs:
            self.register_strategy_component(strategy_name, "strategy")
        
        # Detect current market context
        context = self.market_regime_detector.detect_regime(market_data)
        
        # Update beliefs
        self.strategy_beliefs[strategy_name].update_belief(success, context, confidence)
    
    def rank_strategies(self, market_data: pd.DataFrame) -> List[Tuple[str, float, float]]:
        """Rank strategies by expected performance in current market context"""
        
        context = self.market_regime_detector.detect_regime(market_data)
        
        rankings = []
        for name, belief in self.strategy_beliefs.items():
            prob = belief.get_contextual_probability(context)
            uncertainty = belief.get_uncertainty(context)
            
            # Confidence-adjusted score (penalize high uncertainty)
            score = prob - 0.5 * uncertainty
            rankings.append((name, score, prob))
        
        # Sort by confidence-adjusted score
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_strategy_recommendation(self, market_data: pd.DataFrame) -> Dict:
        """Get strategy recommendations with confidence intervals"""
        
        context = self.market_regime_detector.detect_regime(market_data)
        rankings = self.rank_strategies(market_data)
        
        recommendations = {
            'market_context': {
                'regime': context.regime.value,
                'volatility_percentile': context.volatility_percentile,
                'volume_percentile': context.volume_percentile,
                'trend_strength': context.trend_strength
            },
            'top_strategies': [],
            'regime_analysis': {}
        }
        
        # Top 3 strategies with confidence intervals
        for name, score, prob in rankings[:3]:
            belief = self.strategy_beliefs[name]
            ci_low, ci_high = belief.regime_beliefs[context.regime].confidence_interval()
            
            recommendations['top_strategies'].append({
                'name': name,
                'probability': prob,
                'confidence_interval': [ci_low, ci_high],
                'uncertainty': belief.get_uncertainty(context),
                'sample_size': belief.regime_beliefs[context.regime].effective_sample_size()
            })
        
        return recommendations
    
    def adapt_to_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime):
        """Adapt beliefs when market regime changes"""
        
        print(f"Regime change detected: {old_regime.value} → {new_regime.value}")
        
        # Boost learning rate for new regime
        for belief in self.strategy_beliefs.values():
            # Reset recent performance tracking to focus on new regime
            belief.recent_performance = []
            
    def export_beliefs_summary(self) -> pd.DataFrame:
        """Export current beliefs for analysis"""
        
        summary_data = []
        for name, belief in self.strategy_beliefs.items():
            
            for regime, regime_belief in belief.regime_beliefs.items():
                summary_data.append({
                    'strategy': name,
                    'regime': regime.value,
                    'probability': regime_belief.probability(),
                    'uncertainty': regime_belief.uncertainty(),
                    'sample_size': regime_belief.effective_sample_size(),
                    'alpha': regime_belief.alpha,
                    'beta': regime_belief.beta
                })
        
        return pd.DataFrame(summary_data)

class MarketRegimeDetector:
    """Detect current market regime for contextual belief updating"""

    def __init__(self, regime_sensitivity: float = 0.3):
        self.regime_sensitivity = regime_sensitivity

    def set_regime_sensitivity(self, value: float):
        self.regime_sensitivity = value

    def get_regime_sensitivity(self) -> float:
        return self.regime_sensitivity

    def detect_regime(self, data: pd.DataFrame) -> MarketContext:
        """Detect market regime from recent price data"""
        if len(data) < 20:
            # Default context if insufficient data
            return MarketContext(
                regime=MarketRegime.RANGING,
                volatility_percentile=0.5,
                volume_percentile=0.5,
                trend_strength=0.0,
                timestamp=datetime.now()
            )

        # Calculate indicators
        returns = data['close'].pct_change().dropna()
        vol_20 = returns.rolling(20).std() * np.sqrt(252)

        # Volume analysis (if available)
        volume_series = data['volume'] if 'volume' in data.columns else pd.Series([1]*len(data))
        volume_ma = volume_series.rolling(20).mean()
        volume_current = volume_series.iloc[-1]

        # Trend strength (ADX-like calculation)
        high_low = (data['high'] - data['low']).rolling(14).mean()
        close_shift = abs(data['close'] - data['close'].shift(1)).rolling(14).mean()
        # Compute trend_strength as a float
        if high_low.iloc[-1] > 0:
            trend_strength_series = close_shift / high_low
            # Always use the last value if possible, else fallback to 0.0
            try:
                trend_val = float(trend_strength_series.iloc[-1])
            except Exception:
                trend_val = 0.0
        else:
            trend_val = 0.0

        # Current values
        current_vol = float(vol_20.iloc[-1]) if not vol_20.empty else 0.2
        vol_percentile = min(1.0, max(0.0, (current_vol - vol_20.quantile(0.1)) /
                                      (vol_20.quantile(0.9) - vol_20.quantile(0.1)) if (vol_20.quantile(0.9) - vol_20.quantile(0.1)) != 0 else 0.5))

        volume_percentile = min(1.0, max(0.0, volume_current / volume_ma.iloc[-1] - 0.5)) if volume_ma.iloc[-1] > 0 else 0.5

        # Regime classification
        if current_vol > vol_20.quantile(0.8):
            regime = MarketRegime.VOLATILE
        elif trend_val > 0.7:
            regime = MarketRegime.TRENDING
        elif volume_percentile < 0.3:
            regime = MarketRegime.LOW_VOLUME
        else:
            regime = MarketRegime.RANGING

        return MarketContext(
            regime=regime,
            volatility_percentile=vol_percentile,
            volume_percentile=volume_percentile,
            trend_strength=trend_val,
            timestamp=datetime.now()
        )


# --- BayesianOracleEnhancement: Advanced Wrapper for MirrorCoreBayesianOracle ---
class BayesianOracleEnhancement:
    """
    Enhancement layer for MirrorCoreBayesianOracle, providing advanced explainability,
    integration hooks, and meta-analytics for ensemble trading systems.
    """
    def __init__(self, base_oracle: Optional[MirrorCoreBayesianOracle] = None):
        if base_oracle is None:
            base_oracle = MirrorCoreBayesianOracle()
        self.base_oracle = base_oracle
        self.last_recommendations = None
        self.meta_log = []

    def update_strategy_performance(self, strategy_name: str, success: bool, market_data: pd.DataFrame, confidence: float = 1.0):
        """Proxy to base oracle, with meta-logging."""
        self.base_oracle.update_strategy_performance(strategy_name, success, market_data, confidence)
        self.meta_log.append({
            'event': 'update_strategy_performance',
            'strategy': strategy_name,
            'success': success,
            'confidence': confidence,
            'timestamp': datetime.now()
        })

    def get_strategy_recommendation(self, market_data: pd.DataFrame) -> Dict:
        """Proxy to base oracle, with explainability augmentation."""
        rec = self.base_oracle.get_strategy_recommendation(market_data)
        self.last_recommendations = rec
        # Add explainability: why top strategies were chosen
        for strat in rec.get('top_strategies', []):
            strat['explanation'] = self._explain_strategy(strat)
        return rec

    def _explain_strategy(self, strat: Dict) -> str:
        """Generate a human-readable explanation for a strategy's ranking."""
        prob = strat.get('probability', 0)
        unc = strat.get('uncertainty', 0)
        ci = strat.get('confidence_interval', [0, 1])
        sample = strat.get('sample_size', 0)
        if prob > 0.7 and unc < 0.1:
            return f"High confidence: strong historical performance in this regime (p={prob:.2f}, n={sample:.0f})."
        elif prob > 0.5:
            return f"Moderate confidence: above-average success, but some uncertainty remains (p={prob:.2f}, 95% CI=[{ci[0]:.2f}, {ci[1]:.2f}])."
        else:
            return f"Low confidence: limited or mixed evidence (p={prob:.2f}, n={sample:.0f})."

    def rank_strategies(self, market_data: pd.DataFrame) -> List[Tuple[str, float, float]]:
        return self.base_oracle.rank_strategies(market_data)

    def export_beliefs_summary(self) -> pd.DataFrame:
        return self.base_oracle.export_beliefs_summary()

    def adapt_to_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime):
        self.base_oracle.adapt_to_regime_change(old_regime, new_regime)
        self.meta_log.append({
            'event': 'regime_change',
            'from': old_regime.value,
            'to': new_regime.value,
            'timestamp': datetime.now()
        })

    def get_last_recommendations(self) -> Optional[Dict]:
        return self.last_recommendations

    def get_meta_log(self) -> List[Dict]:
        return self.meta_log

# --- EnhancedTradingOracleEngine: Integrates Bayesian Oracle with Trading System ---

# --- EnhancedTradingOracleEngine: Integrates Bayesian Oracle with Trading System ---
class EnhancedTradingOracleEngine:
    """
    Integrates BayesianOracleEnhancement with a trading system, providing advanced
    directive generation, explainability, and meta-analytics for ensemble trading.
    """
    def __init__(self, bayesian_oracle: Optional["BayesianOracleEnhancement"] = None):
        if bayesian_oracle is None:
            bayesian_oracle = BayesianOracleEnhancement()
        self.bayesian_oracle = bayesian_oracle
        self.last_directives = []
        self.explainability_log = []

    def generate_trading_directives(self, market_data: "pd.DataFrame", strategies: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate trading directives based on Bayesian recommendations and explainability.
        """
        rec = self.bayesian_oracle.get_strategy_recommendation(market_data)
        directives = []
        for strat in rec.get('top_strategies', []):
            if strategies is None or strat['name'] in strategies:
                action = self._map_probability_to_action(strat['probability'])
                directive = {
                    'strategy': strat['name'],
                    'action': action,
                    'probability': strat['probability'],
                    'confidence_interval': strat['confidence_interval'],
                    'uncertainty': strat['uncertainty'],
                    'explanation': strat.get('explanation', ''),
                }
                directives.append(directive)
                self.explainability_log.append({
                    'strategy': strat['name'],
                    'action': action,
                    'meta': strat
                })
        self.last_directives = directives
        return directives

    def _map_probability_to_action(self, prob: float) -> str:
        if prob > 0.7:
            return 'Strong Buy'
        elif prob > 0.55:
            return 'Buy'
        elif prob > 0.45:
            return 'Hold'
        elif prob > 0.3:
            return 'Sell'
        else:
            return 'Strong Sell'

    def get_last_directives(self) -> List[Dict]:
        return self.last_directives

    def get_explainability_log(self) -> List[Dict]:
        return self.explainability_log

# --- BayesianStrategyTrainerEnhancement: Adapter for Bayesian Belief-Driven Training ---
class BayesianStrategyTrainerEnhancement:
    """
    Adapter/enhancer for strategy trainers, using Bayesian beliefs to dynamically
    adjust strategy weights, provide regime-aware training, and explainability.
    """
    def __init__(self, bayesian_oracle: Optional["BayesianOracleEnhancement"] = None):
        if bayesian_oracle is None:
            bayesian_oracle = BayesianOracleEnhancement()
        self.bayesian_oracle = bayesian_oracle
        self.strategy_weights = {}
        self.last_market_context = None
        self.explainability = {}

    def update_weights(self, market_data: "pd.DataFrame"):
        """
        Update strategy weights based on Bayesian beliefs and current market context.
        """
        rec = self.bayesian_oracle.get_strategy_recommendation(market_data)
        self.last_market_context = rec.get('market_context', {})
        weights = {}
        explain = {}
        for strat in rec.get('top_strategies', []):
            name = strat['name']
            prob = strat['probability']
            # Map probability to weight (0.0-1.0)
            weight = min(max((prob - 0.3) / 0.7, 0.0), 1.0)
            weights[name] = weight
            explain[name] = strat.get('explanation', '')
        self.strategy_weights = weights
        self.explainability = explain
        return weights

    def get_weights(self) -> Dict[str, float]:
        return self.strategy_weights

    def get_explainability(self) -> Dict[str, str]:
        return self.explainability

    def get_last_market_context(self) -> Optional[Dict]:
        return self.last_market_context