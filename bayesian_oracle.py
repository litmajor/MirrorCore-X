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
    
    def __init__(self, prior_successes: float = 1.0, prior_failures: float = 1.0):
        self.alpha = prior_successes  # Successes + prior
        self.beta = prior_failures    # Failures + prior
        self.total_observations = 0
        self.last_updated = None
        
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
            
    def export_beliefs_summary(self) -> pd.DataFssrame:
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
        volume_ma = data.get('volume', pd.Series([1]*len(data))).rolling(20).mean()
        volume_current = data.get('volume', pd.Series([1]*len(data))).iloc[-1]
        
        # Trend strength (ADX-like calculation)
        high_low = (data['high'] - data['low']).rolling(14).mean()
        close_shift = abs(data['close'] - data['close'].shift(1)).rolling(14).mean()
        trend_strength = close_shift / high_low if high_low.iloc[-1] > 0 else 0
        
        # Current values
        current_vol = vol_20.iloc[-1] if not vol_20.empty else 0.2
        vol_percentile = min(1.0, max(0.0, (current_vol - vol_20.quantile(0.1)) / 
                                      (vol_20.quantile(0.9) - vol_20.quantile(0.1))))
        
        volume_percentile = min(1.0, max(0.0, volume_current / volume_ma.iloc[-1] - 0.5)) if volume_ma.iloc[-1] > 0 else 0.5
        
        trend_val = trend_strength.iloc[-1] if not trend_strength.empty else 0
        
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

# Usage Example
if __name__ == "__main__":
    
    # Initialize the Bayesian Oracle
    oracle = MirrorCoreBayesianOracle()
    
    # Register some trading strategies
    strategies = [
        "MACD_Crossover", 
        "RSI_Divergence", 
        "Volume_Breakout", 
        "Trend_Following",
        "Mean_Reversion"
    ]
    
    for strategy in strategies:
        oracle.register_strategy_component(strategy, "strategy")
    
    # Simulate some market data and strategy performance
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create synthetic market data
    price_base = 100
    prices = [price_base]
    volumes = []
    
    for i in range(99):
        # Random walk with some trending behavior
        change = np.random.normal(0, 0.02) + 0.001 * (i % 20 - 10) / 10
        prices.append(prices[-1] * (1 + change))
        volumes.append(np.random.lognormal(10, 0.5))
    
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'volume': volumes + [np.random.lognormal(10, 0.5)]  # Match length
    })
    
    # Simulate strategy performance over time
    for i in range(20, len(market_data)):
        current_data = market_data.iloc[:i]
        
        # Simulate different strategy success rates based on market conditions
        regime_context = oracle.market_regime_detector.detect_regime(current_data)
        
        for strategy in strategies:
            # Different strategies perform better in different regimes
            base_prob = {
                "MACD_Crossover": {MarketRegime.TRENDING: 0.7, MarketRegime.RANGING: 0.4, 
                                 MarketRegime.VOLATILE: 0.3, MarketRegime.LOW_VOLUME: 0.5},
                "RSI_Divergence": {MarketRegime.TRENDING: 0.8, MarketRegime.RANGING: 0.6,
                                 MarketRegime.VOLATILE: 0.4, MarketRegime.LOW_VOLUME: 0.5},
                "Volume_Breakout": {MarketRegime.TRENDING: 0.6, MarketRegime.RANGING: 0.3,
                                  MarketRegime.VOLATILE: 0.7, MarketRegime.LOW_VOLUME: 0.2},
                "Trend_Following": {MarketRegime.TRENDING: 0.8, MarketRegime.RANGING: 0.3,
                                  MarketRegime.VOLATILE: 0.5, MarketRegime.LOW_VOLUME: 0.4},
                "Mean_Reversion": {MarketRegime.TRENDING: 0.3, MarketRegime.RANGING: 0.7,
                                 MarketRegime.VOLATILE: 0.6, MarketRegime.LOW_VOLUME: 0.5}
            }
            
            success_prob = base_prob[strategy][regime_context.regime]
            success = np.random.random() < success_prob
            
            oracle.update_strategy_performance(strategy, success, current_data)
    
    # Get final recommendations
    final_recommendations = oracle.get_strategy_recommendation(market_data)
    
    print("=== MirrorCore Bayesian Oracle Recommendations ===\n")
    print(f"Market Context: {final_recommendations['market_context']}\n")
    
    print("Top Strategy Recommendations:")
    for i, strategy in enumerate(final_recommendations['top_strategies'], 1):
        ci_low, ci_high = strategy['confidence_interval']
        print(f"{i}. {strategy['name']}")
        print(f"   Success Probability: {strategy['probability']:.3f}")
        print(f"   95% Confidence Interval: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"   Uncertainty: {strategy['uncertainty']:.3f}")
        print(f"   Sample Size: {strategy['sample_size']:.1f}")
        print()
    
    # Export beliefs summary
    beliefs_df = oracle.export_beliefs_summary()
    print("Detailed Beliefs Summary:")
    print(beliefs_df.round(3))