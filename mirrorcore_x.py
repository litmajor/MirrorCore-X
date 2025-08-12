#This is the original version of mirrax
# --- MirrorCore-X (enhanced trading engine) ---
#currently not being used, but may be useful in the future for research purposes on cognitive self-awareness

# --- TradingOracleEngine (core logic) ---
class TradingOracleEngine:
    """Enhanced Oracle Engine specifically designed for trading integration"""
    def __init__(self, user_id: str, market_context: dict = None):
        self.user_id = user_id
        self.market_context = market_context or {}
        self.seed = self.generate_trading_seed(user_id)
        self.timeline_map = self.build_market_timeline_matrix()
        self.anchor = self.calculate_market_anchor_vector()
        self.confidence_history = []
        self.performance_feedback = []
    def generate_trading_seed(self, user_id):
        ts = str(datetime.utcnow().timestamp())
        market_hash = str(hash(str(self.market_context))) if self.market_context else "0"
        raw = f"{user_id}_{ts}_{market_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()
    def build_market_timeline_matrix(self):
        np.random.seed(int(self.seed[:8], 16))
        return np.random.rand(8, 6)
    def calculate_market_anchor_vector(self):
        hash_int = int(self.seed[:12], 16)
        base_anchor = np.array([
            (hash_int % 7) / 6,
            ((hash_int >> 3) % 9) / 8,
            ((hash_int >> 6) % 11) / 10,
            ((hash_int >> 9) % 13) / 12,
            ((hash_int >> 12) % 5) / 4,
            ((hash_int >> 15) % 17) / 16
        ])
        if self.market_context:
            market_multiplier = np.array([
                self.market_context.get('trend_strength', 1.0),
                self.market_context.get('volatility_factor', 1.0),
                self.market_context.get('volume_factor', 1.0),
                self.market_context.get('momentum_factor', 1.0),
                1.0 - self.market_context.get('fear_index', 0.5),
                self.market_context.get('opportunity_score', 1.0)
            ])
            base_anchor = base_anchor * market_multiplier
        return np.clip(base_anchor, 0, 1)
    def evaluate_trading_timelines(self):
        scores = []
        for idx, timeline in enumerate(self.timeline_map):
            alignment = np.dot(self.anchor, timeline)
            confidence_boost = np.mean(self.confidence_history[-10:]) if self.confidence_history else 0
            performance_boost = np.mean(self.performance_feedback[-5:]) if self.performance_feedback else 0
            adjusted_score = alignment + (confidence_boost * 0.1) + (performance_boost * 0.15)
            scores.append((idx, adjusted_score, timeline))
        return sorted(scores, key=lambda x: x[1], reverse=True)
    def get_trading_directive(self, signals_df=None, psych_state=None):
        ranked_timelines = self.evaluate_trading_timelines()
        primary_timeline = ranked_timelines[0]
        secondary_timeline = ranked_timelines[1]
        timeline_idx, stability_score, timeline_vector = primary_timeline
        trading_actions = [
            "AGGRESSIVE_BUY", "CONSERVATIVE_BUY", "SCALE_IN", 
            "HOLD_POSITION", "SCALE_OUT", "CONSERVATIVE_SELL", 
            "AGGRESSIVE_SELL", "WAIT_SIGNAL"
        ]
        risk_factor = timeline_vector[4]
        opportunity_factor = timeline_vector[5]
        action_index = int((stability_score * opportunity_factor * 10) % len(trading_actions))
        base_action = trading_actions[action_index]
        if risk_factor > 0.7:
            if "AGGRESSIVE" in base_action:
                base_action = base_action.replace("AGGRESSIVE", "CONSERVATIVE")
            elif base_action == "SCALE_IN":
                base_action = "WAIT_SIGNAL"
        regime = self._detect_market_regime(timeline_vector)
        position_size = self._calculate_position_size(stability_score, risk_factor)
        recommended_timeframe = self._recommend_timeframe(timeline_vector, stability_score)
        directive = {
            "primary_timeline": timeline_idx,
            "stability_score": round(stability_score, 4),
            "action": base_action,
            "market_regime": regime,
            "position_size_factor": position_size,
            "recommended_timeframe": recommended_timeframe,
            "risk_level": "HIGH" if risk_factor > 0.6 else "MEDIUM" if risk_factor > 0.3 else "LOW",
            "opportunity_level": "HIGH" if opportunity_factor > 0.7 else "MEDIUM" if opportunity_factor > 0.4 else "LOW",
            "confidence": round(min(stability_score * 100, 95), 1),
            "secondary_timeline": ranked_timelines[1][0],
            "timeline_consensus": len([t for t in ranked_timelines[:3] if t[1] > 0.5])
        }
        self.confidence_history.append(directive["confidence"] / 100)
        return directive
    def _detect_market_regime(self, timeline_vector):
        trend, volatility, volume, momentum, fear, opportunity = timeline_vector
        if trend > 0.7 and momentum > 0.6:
            return "STRONG_TREND"
        elif volatility > 0.8:
            return "HIGH_VOLATILITY"
        elif trend < 0.3 and momentum < 0.3:
            return "SIDEWAYS"
        elif fear > 0.7:
            return "FEAR_DRIVEN"
        elif opportunity > 0.8:
            return "OPPORTUNITY_RICH"
        else:
            return "MIXED"
    def _calculate_position_size(self, stability_score, risk_factor):
        base_size = stability_score * 0.8
        risk_adjustment = 1.0 - (risk_factor * 0.5)
        return round(np.clip(base_size * risk_adjustment, 0.1, 1.0), 2)
    def _recommend_timeframe(self, timeline_vector, stability_score):
        trend, volatility, volume, momentum, fear, opportunity = timeline_vector
        if volatility > 0.8 and momentum > 0.7:
            return "scalping"
        elif stability_score > 0.6 and trend > 0.5:
            return "7day"
        elif trend > 0.7 and stability_score > 0.7:
            return "long_term"
        else:
            return "7day"
    def provide_performance_feedback(self, pnl, win_rate):
        normalized_pnl = np.tanh(pnl / 1000)
        normalized_winrate = (win_rate - 50) / 50
        feedback_score = (normalized_pnl * 0.7) + (normalized_winrate * 0.3)
        self.performance_feedback.append(feedback_score)
        if len(self.performance_feedback) > 20:
            self.performance_feedback = self.performance_feedback[-20:]

# --- OracleEnhancedMirrorCore (integration class) ---
class OracleEnhancedMirrorCore:
    """Integration class that enhances MirrorCore-X with Oracle Engine"""
    def __init__(self, mirrorcore_instance, user_id="TRADER_001"):
        self.mirrorcore = mirrorcore_instance
        self.oracle = None
        self.user_id = user_id
        self.oracle_decisions = []
    def _update_market_context(self, signals_df, psych_state, fear_level, scanner_instance=None):
        if signals_df is None or signals_df.empty:
            return {}
        momentum_cols = [col for col in signals_df.columns if col.startswith('momentum_')]
        primary_momentum_col = momentum_cols[0] if momentum_cols else 'momentum_7day'
        trend_strength = signals_df[primary_momentum_col].mean() if primary_momentum_col in signals_df.columns else 0.5
        momentum_std = signals_df[primary_momentum_col].std() if primary_momentum_col in signals_df.columns else 0.1
        rsi_volatility = signals_df['rsi'].std() / 20 if 'rsi' in signals_df.columns else 1.0
        volume_factor = 1.0
        if 'average_volume_usd' in signals_df.columns:
            volume_mean = signals_df['average_volume_usd'].mean()
            volume_factor = min(volume_mean / 1_000_000, 3.0)
        macd_bullish_ratio = (signals_df['macd'] > 0).mean() if 'macd' in signals_df.columns else 0.5
        strong_signals = ['Consistent Uptrend', 'New Spike', 'MACD Bullish', 'Moderate Uptrend']
        weak_signals = ['Topping Out', 'Lagging', 'Potential Reversal']
        signal_strength = 0.5
        if 'signal' in signals_df.columns:
            strong_count = signals_df['signal'].isin(strong_signals).sum()
            weak_count = signals_df['signal'].isin(weak_signals).sum()
            total_signals = len(signals_df)
            if total_signals > 0:
                signal_strength = (strong_count - weak_count) / total_signals + 0.5
                signal_strength = max(0, min(1, signal_strength))
        fear_greed_value = 50
        if scanner_instance and hasattr(scanner_instance, 'fear_greed_history') and scanner_instance.fear_greed_history:
            fear_greed_value = scanner_instance.fear_greed_history[-1] or 50
        fear_greed_opportunity = 1.0 - (fear_greed_value / 100)
        context = {
            "trend_strength": max(0, min(2, abs(trend_strength) * 10)),
            "volatility_factor": max(0.1, min(3.0, rsi_volatility + momentum_std)),
            "volume_factor": volume_factor,
            "momentum_factor": macd_bullish_ratio,
            "fear_index": fear_level,
            "opportunity_score": (signal_strength + fear_greed_opportunity) / 2,
            "market_fear_greed": fear_greed_value,
            "signal_distribution": {
                "strong_signals": strong_count if 'signal' in signals_df.columns else 0,
                "weak_signals": weak_count if 'signal' in signals_df.columns else 0,
                "total_signals": len(signals_df)
            }
        }
        return context
    def enhanced_tick(self, timeframe="7day"):
        try:
            scanned_signals = self.mirrorcore.scanner.scan_market(timeframe)
            strong_signals = self.mirrorcore.scanner.get_strong_signals(timeframe)
            market_context = self._update_market_context(
                strong_signals, 
                getattr(self.mirrorcore.arch_ctrl, 'psych_state', {}),
                getattr(self.mirrorcore.arch_ctrl, 'fear', 0.5),
                self.mirrorcore.scanner
            )
            self.oracle = TradingOracleEngine(self.user_id, market_context)
            oracle_directive = self.oracle.get_trading_directive(strong_signals, getattr(self.mirrorcore.arch_ctrl, 'psych_state', {}))
            self.oracle_decisions.append(oracle_directive)
            return {
                "signals": strong_signals,
                "oracle": oracle_directive,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"Error in Oracle-enhanced tick: {e}")
            return {"signals": None, "oracle": {"error": str(e)}, "timestamp": datetime.utcnow().isoformat()}

# === Strategy Trainer Integration ===
from strategy_trainer_agent import (
    StrategyTrainerAgent,
    UTSignalAgent,
    GradientTrendAgent,
    SupportResistanceAgent
)
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from momentum_scanner import MomentumScanner
import ccxt
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from arch_ctrl import ARCH_CTRL
from collections import Counter
import matplotlib.gridspec as gridspec

# --- Trade Analyzer Integration ---
from trade_analyzer_agent import TradeAnalyzerAgent

# Register
strategy_trainer = StrategyTrainerAgent()
strategy_trainer.learn_new_strategy("UT_BOT", UTSignalAgent)
strategy_trainer.learn_new_strategy("GRADIENT_TREND", GradientTrendAgent)
strategy_trainer.learn_new_strategy("VBSR", SupportResistanceAgent)


# Enhanced Base Agent Class
class MirrorAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.last_update = time.time()
        self.memory = []
        self.max_memory = 1000

    @abstractmethod
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and return agent's output"""
        pass

    def store_memory(self, item: Any):
        """Store item in agent's memory with timestamp"""
        self.memory.append({
            'timestamp': time.time(),
            'data': item
        })
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def get_recent_memory(self, seconds: int = 300) -> List[Any]:
        """Get memories from the last N seconds"""
        cutoff = time.time() - seconds
        return [m['data'] for m in self.memory if m['timestamp'] > cutoff]

    def debug(self) -> Dict[str, Any]:
        """Return internal debug information (can be overridden)"""
        return {
            'name': self.name,
            'memory_count': len(self.memory),
            'last_update': self.last_update
        }

    def reset(self):
        """Reset agent's state"""
        self.last_update = time.time()
        self.memory = []
        self.max_memory = 1000

    def __str__(self):
        return f"MirrorAgent(name={self.name}, memory_count={len(self.memory)}, last_update={datetime.fromtimestamp(self.last_update)})"
    
# Core Data Structures
class MarketStructure(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    VOLATILE = "volatile"

class EmotionalState(Enum):
    CONFIDENT = "confident"
    FEARFUL = "fearful"
    GREEDY = "greedy"
    UNCERTAIN = "uncertain"
    EUPHORIC = "euphoric"
    PANIC = "panic"

@dataclass
class MarketData:
    timestamp: float
    price: float
    volume: int
    high: float
    low: float
    open: float
    volatility: float = 0.0
    structure: MarketStructure = MarketStructure.RANGING

@dataclass
class TradeSignal:
    direction: str  # "long", "short", "neutral"
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    risk_level: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class PsychProfile:
    emotional_state: EmotionalState = EmotionalState.UNCERTAIN
    stress_level: float = 0.5
    confidence_level: float = 0.5
    risk_tolerance: float = 0.5
    recent_pnl: float = 0.0
    win_streak: int = 0
    loss_streak: int = 0


# Extend existing enums for self-awareness
class ConsistencyLevel(Enum):
    HIGHLY_CONSISTENT = "highly_consistent"
    CONSISTENT = "consistent"
    MODERATELY_INCONSISTENT = "moderately_inconsistent"
    HIGHLY_INCONSISTENT = "highly_inconsistent"
    ERRATIC = "erratic"

class DeviationType(Enum):
    RISK_TOLERANCE_SHIFT = "risk_tolerance_shift"
    DIRECTIONAL_BIAS_CHANGE = "directional_bias_change"
    EMOTIONAL_VOLATILITY = "emotional_volatility"
    CONFIDENCE_PATTERN_BREAK = "confidence_pattern_break"
    EXECUTION_TIMING_DRIFT = "execution_timing_drift"
    POSITION_SIZING_ANOMALY = "position_sizing_anomaly"
    FEAR_THRESHOLD_SHIFT = "fear_threshold_shift"

@dataclass
class BehavioralBaseline:
    """Represents the agent's normal behavioral patterns"""
    avg_risk_tolerance: float = 0.5
    typical_confidence_range: Tuple[float, float] = (0.3, 0.7)
    dominant_emotional_states: List[str] = field(default_factory=list)
    directional_bias_ratio: float = 0.5  # long/short preference
    avg_position_hold_time: float = 604800.0 # seconds
    typical_position_size_range: Tuple[float, float] = (500.0, 1500.0)
    fear_sensitivity: float = 0.5
    decision_consistency_score: float = 0.8
    last_updated: float = field(default_factory=time.time)

@dataclass
class BehavioralDeviation:
    """Represents a detected deviation from baseline behavior"""
    deviation_type: DeviationType
    severity: float  # 0.0 to 1.0
    description: str
    timestamp: float
    current_value: Any
    baseline_value: Any
    confidence: float = 0.8
    recommendation: str = ""

@dataclass
class SelfAwarenessState:
    """Current self-awareness status"""
    consistency_level: ConsistencyLevel = ConsistencyLevel.CONSISTENT
    active_deviations: List[BehavioralDeviation] = field(default_factory=list)
    behavioral_drift_score: float = 0.0  # How much behavior has drifted overall
    self_trust_level: float = 0.8  # How much the agent trusts its own decisions
    metacognitive_confidence: float = 0.7  # Confidence in self-assessment
    introspection_depth: int = 50  # How many past actions to analyze


# Enhanced Perception Layer
class PerceptionLayer(MirrorAgent):
    def __init__(self, scanner):
        super().__init__('Perception')
        self.scanner = scanner
        self.price_history = []
        self.volume_history = []
        self.volatility_window = 20
        self.pattern_memory = []  # Stores recent signal patterns for replay or anomaly detection

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        market_data = data.get('market_data')
        if not market_data:
            return {}

        top_momentum = self.scanner.scan_market()

        self.price_history.append(market_data.price)
        self.volume_history.append(market_data.volume)

        if len(self.price_history) > 100:
            self.price_history.pop(0)
            self.volume_history.pop(0)

        signals = self._extract_signals(market_data)

        # Store last N signal patterns
        if len(self.pattern_memory) >= 50:
            self.pattern_memory.pop(0)
        self.pattern_memory.append(signals.copy())

        for _, row in top_momentum.iterrows():
            signals[f"momentum_{row['symbol']}"] = row['momentum_7d']
            signals[f"signal_{row['symbol']}"] = row['signal']

        print(f"[{self.name}] Processing market data: Price={market_data.price:.2f}, "
              f"Volume={market_data.volume}, Structure={market_data.structure.value}")

        self.store_memory(signals)
        return {'perception_signals': signals, 'market_data': market_data}

    def _extract_signals(self, market_data: MarketData) -> Dict[str, float]:
        signals = {}
        if len(self.price_history) < 2:
            return signals

        signals['momentum'] = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]

        if len(self.volume_history) >= 2:
            signals['volume_trend'] = (self.volume_history[-1] - self.volume_history[-2]) / max(self.volume_history[-2], 1)

        if len(self.price_history) >= self.volatility_window:
            recent_prices = self.price_history[-self.volatility_window:]
            signals['volatility'] = np.std(recent_prices) / np.mean(recent_prices)

        if len(self.price_history) >= 10:
            recent_high = max(self.price_history[-10:])
            recent_low = min(self.price_history[-10:])
            current_price = self.price_history[-1]
            signals['resistance_proximity'] = (recent_high - current_price) / (recent_high - recent_low + 0.001)
            signals['support_proximity'] = (current_price - recent_low) / (recent_high - recent_low + 0.001)

        # Regime Change Detection
        if len(self.price_history) >= 21:
            short_term = np.mean(self.price_history[-5:])
            long_term = np.mean(self.price_history[-20:])
            signals['regime_shift'] = short_term - long_term

        return signals

# EgoProcessor updated with Reflexive Confidence Feedback and Ego Drift Model
class EgoProcessor(MirrorAgent):
    def __init__(self):
        super().__init__('EgoProcessor')
        self.beliefs = {
            'trend_bias': 0.5,
            'breakout_bias': 0.5,
            'reversal_bias': 0.5,
            'momentum_bias': 0.5,
        }
        self.psych_profile = PsychProfile()
        self.recent_trades = []
        self.ego_drift = 0.0  # Subtle bias accumulator

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self_trust_multiplier = data.get('self_trust_multiplier', 1.0)
        market_data = data.get('market_data')
        perception_signals = data.get('perception_signals', {})
        trade_results = data.get('trade_results', [])

        self._update_psychology(trade_results)
        self._adjust_beliefs(perception_signals, market_data, self_trust_multiplier)

        print(f"[{self.name}] Emotional State: {self.psych_profile.emotional_state.value}, "
              f"Confidence: {self.psych_profile.confidence_level:.2f}, "
              f"Stress: {self.psych_profile.stress_level:.2f}, Drift: {self.ego_drift:.3f}")

        return {
            'beliefs': self.beliefs.copy(),
            'psych_profile': self.psych_profile,
            'ego_bias': self._calculate_ego_bias()
        }

    def _update_psychology(self, trade_results: List[Dict]):
        if not trade_results:
            return

        recent_pnl = sum(r.get('pnl', 0) for r in trade_results[-10:])
        self.psych_profile.recent_pnl = recent_pnl

        wins = losses = 0
        for result in reversed(trade_results[-10:]):
            if result.get('pnl', 0) > 0:
                if losses == 0:
                    wins += 1
                else:
                    break
            else:
                if wins == 0:
                    losses += 1
                else:
                    break

        self.psych_profile.win_streak = wins
        self.psych_profile.loss_streak = losses

        if recent_pnl > 0 and wins >= 3:
            self.psych_profile.emotional_state = EmotionalState.CONFIDENT
            self.psych_profile.confidence_level = min(0.95, self.psych_profile.confidence_level + 0.1)
            self.ego_drift += 0.01
        elif recent_pnl < 0 and losses >= 2:
            self.psych_profile.emotional_state = EmotionalState.FEARFUL
            self.psych_profile.confidence_level = max(0.05, self.psych_profile.confidence_level - 0.1)
            self.psych_profile.stress_level = min(0.95, self.psych_profile.stress_level + 0.2)
            self.ego_drift -= 0.01
        else:
            self.psych_profile.emotional_state = EmotionalState.UNCERTAIN

    def _adjust_beliefs(self, signals: Dict[str, float], market_data: Optional[MarketData], self_trust_multiplier: float = 1.0):
        if not signals or not market_data:
            return

        momentum = signals.get('momentum', 0)
        volatility = signals.get('volatility', 0)

        if abs(momentum) > 0.02:
            self.beliefs['momentum_bias'] = min(0.9, self.beliefs['momentum_bias'] + 0.05)

        if volatility > 0.05:
            self.beliefs['breakout_bias'] = min(0.85, self.beliefs['breakout_bias'] + 0.03)

        for key in self.beliefs:
            if self.psych_profile.emotional_state == EmotionalState.FEARFUL:
                self.beliefs[key] = max(0.15, self.beliefs[key] - 0.02)
                # Reduce beliefs further if self-trust is low
                self.beliefs[key] = max(0.1, self.beliefs[key] * self_trust_multiplier)
            elif self.psych_profile.emotional_state == EmotionalState.CONFIDENT:
                self.beliefs[key] = min(0.85, self.beliefs[key] + 0.01)

    def _calculate_ego_bias(self) -> float:
        confidence = self.psych_profile.confidence_level
        stress = 1.0 - self.psych_profile.stress_level
        return (confidence + stress + self.ego_drift) / 3.0

class FearAnalyzer(MirrorAgent):
    def __init__(self):
        super().__init__('FearAnalyzer')
        self.volatility_threshold = 0.05
        self.panic_threshold = 0.1 # Panic threshold for extreme volatility
        # Initialize fear level and decay rate
        self.fear_level = 0.0 # Initial fear level
        self.decay_rate = 0.1 # Decay rate for fear level
        # Initialize fear tags
        self.fear_tags = []

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        market_data = data.get('market_data')
        signals = data.get('perception_signals', {})
        psych_profile = data.get('psych_profile') # Psych profile for emotional state and stress level

        fear_metrics = self._calculate_fear_metrics(signals, market_data, psych_profile)
        self.fear_level = (1 - self.decay_rate) * self.fear_level + self.decay_rate * np.mean(list(fear_metrics.values()))
        self_trust_multiplier = data.get('self_trust_multiplier', 1.0)
        if self_trust_multiplier < 0.5: 
            self.fear_level = min(1.0, self.fear_level * (1.2 - self_trust_multiplier))  # amplify fear

        # Detect volatility regime
        regime = self._detect_volatility_regime(signals.get("volatility", 0))
        self.fear_tags = self._generate_fear_tags(fear_metrics)

        print(f"[{self.name}] Fear: {self.fear_level:.2f}, Regime: {regime}, Tags: {', '.join(self.fear_tags)}")

        return {
            'fear_level': self.fear_level,
            'fear_metrics': fear_metrics,
            'risk_assessment': self._assess_risk(fear_metrics),
            'volatility_regime': regime,
            'fear_tags': self.fear_tags
        }

    def _calculate_fear_metrics(self, signals, market_data, psych_profile):
        metrics = {}
        volatility = signals.get('volatility', 0)
        metrics['volatility_risk'] = min(1.0, volatility / self.volatility_threshold)
        metrics['momentum_risk'] = min(1.0, abs(signals.get('momentum', 0)) * 10)
        volume_trend = signals.get('volume_trend', 0)
        metrics['volume_anomaly'] = min(1.0, abs(volume_trend) / 3.0) if abs(volume_trend) > 2.0 else 0.0

        if psych_profile:
            stress_multiplier = 1.0 + psych_profile.stress_level
            for k in metrics:
                metrics[k] *= stress_multiplier

        return metrics

    def _assess_risk(self, fear_metrics):
        avg = np.mean(list(fear_metrics.values()))
        if avg > 0.7: return "HIGH_RISK"
        if avg > 0.4: return "MEDIUM_RISK"
        return "LOW_RISK"

    def _detect_volatility_regime(self, volatility):
        if volatility > 0.08:
            return "turbulent"
        elif volatility > 0.04:
            return "normal"
        else:
            return "calm"

    def _generate_fear_tags(self, metrics):
        tags = []
        if metrics.get('volatility_risk', 0) > 1.2: tags.append("panic_spike")
        if metrics.get('volume_anomaly', 0) > 0.5: tags.append("liquidity_shock")
        if metrics.get('momentum_risk', 0) > 0.7: tags.append("price_whiplash")
        if not tags: tags.append("resilient")
        return tags
    

class SelfAwarenessAgent(MirrorAgent):
    def __init__(self, baseline_window: int = 100, deviation_threshold: float = 0.3):
        super().__init__('SelfAwarenessAgent')
        
        # Core configuration
        self.baseline_window = baseline_window
        self.deviation_threshold = deviation_threshold
        
        # Behavioral tracking
        self.behavioral_baseline = BehavioralBaseline()
        self.behavioral_history = deque(maxlen=1000)  # Long-term memory
        self.action_patterns = defaultdict(list)
        self.consistency_tracker = deque(maxlen=50)
        
        # Self-awareness state
        self.awareness_state = SelfAwarenessState()
        self.deviation_history = deque(maxlen=200)
        
        # Metacognitive metrics
        self.self_prediction_accuracy = deque(maxlen=100)
        self.behavioral_coherence_score = 0.8
        self.last_introspection = time.time()
        
        # Pattern recognition
        self.decision_clusters = {}  # Groups similar decisions
        self.anomaly_detector_sensitivity = 0.7
        
        print(f"[{self.name}] Initialized with baseline window: {baseline_window}, "
              f"deviation threshold: {deviation_threshold}")

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main update cycle - performs self-awareness analysis"""
        # Extract relevant data
        current_behavior = self._extract_current_behavior(data)
        # Store in behavioral history
        self.behavioral_history.append({
            'timestamp': time.time(),
            'behavior': current_behavior,
            'market_context': data.get('market_data'),
            'decision': data.get('selected_scenario'),
            'execution': data.get('execution_result'),
            'psychology': data.get('psych_profile')
        })
        # Update baseline if we have enough data
        if len(self.behavioral_history) >= self.baseline_window:
            if time.time() - self.behavioral_baseline.last_updated > 300:  # Update every 5 minutes
                self._update_behavioral_baseline()
        # Detect deviations from baseline
        deviations = self._detect_behavioral_deviations(current_behavior)
        # Update consistency tracking
        consistency_score = self._calculate_consistency_score()
        self.consistency_tracker.append(consistency_score)
        # Perform introspection
        introspection_insights = self._perform_introspection()
        # Update self-awareness state
        self._update_awareness_state(deviations, consistency_score)
        # Generate self-awareness output
        output = self._generate_awareness_output(deviations, introspection_insights)
        print(f"[{self.name}] Consistency: {self.awareness_state.consistency_level.value}, "
              f"Drift: {self.awareness_state.behavioral_drift_score:.3f}, "
              f"Trust: {self.awareness_state.self_trust_level:.2f}, "
              f"Active Deviations: {len(self.awareness_state.active_deviations)}")
        return output

    def _extract_current_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        behavior = {}
        fear_level = data.get('fear_level', 0.5)
        psych_profile = data.get('psych_profile')
        if psych_profile:
            behavior['risk_tolerance'] = (1.0 - fear_level) * psych_profile.confidence_level
            behavior['confidence'] = psych_profile.confidence_level
            behavior['stress_level'] = psych_profile.stress_level
            behavior['emotional_state'] = psych_profile.emotional_state.value
        selected_scenario = data.get('selected_scenario')
        if selected_scenario:
            behavior['decision_direction'] = selected_scenario.direction
            behavior['decision_confidence'] = selected_scenario.confidence
            behavior['decision_risk_level'] = selected_scenario.risk_level
        execution_result = data.get('execution_result')
        if execution_result and execution_result.get('action') == 'opened':
            position = execution_result.get('position', {})
            behavior['position_size'] = position.get('size', 0)
            behavior['executed_direction'] = position.get('direction')
        fear_metrics = data.get('fear_metrics', {})
        if fear_metrics:
            behavior['fear_sensitivity'] = np.mean(list(fear_metrics.values()))
        behavior['timestamp'] = time.time()
        return behavior

    def _update_behavioral_baseline(self):
        """Update behavioral baseline from recent history"""
        if len(self.behavioral_history) < self.baseline_window:
            return
        
        recent_behaviors = [entry['behavior'] for entry in list(self.behavioral_history)[-self.baseline_window:]]
        
        # Calculate baseline metrics
        risk_tolerances = [b.get('risk_tolerance', 0.5) for b in recent_behaviors if 'risk_tolerance' in b]
        confidences = [b.get('confidence', 0.5) for b in recent_behaviors if 'confidence' in b]
        emotional_states = [b.get('emotional_state') for b in recent_behaviors if 'emotional_state' in b]
        position_sizes = [b.get('position_size', 1000) for b in recent_behaviors if 'position_size' in b]
        fear_sensitivities = [b.get('fear_sensitivity', 0.5) for b in recent_behaviors if 'fear_sensitivity' in b]
        
        # Update baseline
        if risk_tolerances:
            self.behavioral_baseline.avg_risk_tolerance = float(np.mean(risk_tolerances))
        
        if confidences:
            conf_low = float(np.percentile(confidences, 25))
            conf_high = float(np.percentile(confidences, 75))
            self.behavioral_baseline.typical_confidence_range = (conf_low, conf_high)

        
        if emotional_states:
            # Find most common emotional states
            state_counts = defaultdict(int)
            for state in emotional_states:
                if state:
                    state_counts[state] += 1
            self.behavioral_baseline.dominant_emotional_states = [
                state for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            ]
        if position_sizes:
           size_min = float(np.percentile(position_sizes, 25))
           size_max = float(np.percentile(position_sizes, 75))
           self.behavioral_baseline.typical_position_size_range = (size_min, size_max)

        
        if fear_sensitivities:
            self.behavioral_baseline.fear_sensitivity = float(np.mean(fear_sensitivities))
        
        # Calculate directional bias
        directions = [b.get('decision_direction') for b in recent_behaviors if 'decision_direction' in b]
        if directions:
            long_count = directions.count('long')
            short_count = directions.count('short')
            total_directional = long_count + short_count
            if total_directional > 0:
                self.behavioral_baseline.directional_bias_ratio = long_count / total_directional
        
        self.behavioral_baseline.last_updated = time.time()
        
        print(f"[{self.name}] Baseline updated - Risk Tolerance: {self.behavioral_baseline.avg_risk_tolerance:.3f}, "
              f"Confidence Range: {self.behavioral_baseline.typical_confidence_range}, "
              f"Directional Bias: {self.behavioral_baseline.directional_bias_ratio:.3f}")

    def _detect_behavioral_deviations(self, current_behavior: Dict[str, Any]) -> List[BehavioralDeviation]:
        """Detect deviations from established behavioral baseline"""
        deviations = []
        
        # Risk tolerance deviation
        current_risk = current_behavior.get('risk_tolerance')
        if current_risk is not None:
            baseline_risk = self.behavioral_baseline.avg_risk_tolerance
            risk_deviation = abs(current_risk - baseline_risk)
            
            if risk_deviation > self.deviation_threshold:
                severity = min(1.0, risk_deviation / 0.5)
                deviations.append(BehavioralDeviation(
                    deviation_type=DeviationType.RISK_TOLERANCE_SHIFT,
                    severity=severity,
                    description=f"Risk tolerance shifted from {baseline_risk:.3f} to {current_risk:.3f}",
                    timestamp=time.time(),
                    current_value=current_risk,
                    baseline_value=baseline_risk,
                    recommendation="Review risk management parameters or market conditions"
                ))
        
        # Confidence pattern deviation
        current_confidence = current_behavior.get('confidence')
        if current_confidence is not None:
            conf_min, conf_max = self.behavioral_baseline.typical_confidence_range
            if current_confidence < conf_min - 0.2 or current_confidence > conf_max + 0.2:
                severity = min(1.0, max(
                    abs(current_confidence - conf_min), 
                    abs(current_confidence - conf_max)
                ) / 0.3)
                
                deviations.append(BehavioralDeviation(
                    deviation_type=DeviationType.CONFIDENCE_PATTERN_BREAK,
                    severity=severity,
                    description=f"Confidence {current_confidence:.3f} outside typical range [{conf_min:.3f}, {conf_max:.3f}]",
                    timestamp=time.time(),
                    current_value=current_confidence,
                    baseline_value=(conf_min, conf_max),
                    recommendation="Monitor for overconfidence or loss of conviction"
                ))
        
        # Emotional volatility detection
        current_emotional_state = current_behavior.get('emotional_state')
        if current_emotional_state and current_emotional_state not in self.behavioral_baseline.dominant_emotional_states:
            # Check frequency of emotional state changes in recent history
            recent_states = [
                entry['behavior'].get('emotional_state') 
                for entry in list(self.behavioral_history)[-10:] 
                if 'emotional_state' in entry['behavior']
            ]
            
            if len(set(recent_states)) > 4:  # Too many different emotional states recently
                deviations.append(BehavioralDeviation(
                    deviation_type=DeviationType.EMOTIONAL_VOLATILITY,
                    severity=0.7,
                    description=f"Unusual emotional volatility - current: {current_emotional_state}",
                    timestamp=time.time(),
                    current_value=current_emotional_state,
                    baseline_value=self.behavioral_baseline.dominant_emotional_states,
                    recommendation="Consider emotional regulation or stress management"
                ))
        
        # Position sizing anomaly
        current_position_size = current_behavior.get('position_size')
        if current_position_size is not None:
            size_min, size_max = self.behavioral_baseline.typical_position_size_range
            if current_position_size < size_min * 0.5 or current_position_size > size_max * 2.0:
                severity = min(1.0, max(
                    abs(current_position_size - size_min) / size_min,
                    abs(current_position_size - size_max) / size_max
                ))
                
                deviations.append(BehavioralDeviation(
                    deviation_type=DeviationType.POSITION_SIZING_ANOMALY,
                    severity=severity,
                    description=f"Position size {current_position_size:.0f} deviates from typical range [{size_min:.0f}, {size_max:.0f}]",
                    timestamp=time.time(),
                    current_value=current_position_size,
                    baseline_value=(size_min, size_max),
                    recommendation="Review position sizing logic and risk parameters"
                ))
        
        # Directional bias shift
        recent_directions = [
            entry['behavior'].get('decision_direction') 
            for entry in list(self.behavioral_history)[-20:] 
            if entry['behavior'].get('decision_direction') in ['long', 'short']
        ]
        
        if len(recent_directions) >= 10:
            recent_long_ratio = recent_directions.count('long') / len(recent_directions)
            baseline_ratio = self.behavioral_baseline.directional_bias_ratio
            bias_shift = abs(recent_long_ratio - baseline_ratio)
            
            if bias_shift > 0.4:  # Significant directional shift
                deviations.append(BehavioralDeviation(
                    deviation_type=DeviationType.DIRECTIONAL_BIAS_CHANGE,
                    severity=min(1.0, bias_shift / 0.5),
                    description=f"Directional bias shifted from {baseline_ratio:.3f} to {recent_long_ratio:.3f}",
                    timestamp=time.time(),
                    current_value=recent_long_ratio,
                    baseline_value=baseline_ratio,
                    recommendation="Analyze market conditions causing directional preference change"
                ))
        
        return deviations

    def _perform_introspection(self) -> List[str]:
        """Perform deep introspection on behavioral patterns"""
        insights = []
        
        if len(self.behavioral_history) < 20:
            return ["Insufficient behavioral history for deep introspection"]
        
        # Analyze decision-outcome correlation
        recent_entries = list(self.behavioral_history)[-50:]
        decision_outcomes = []
        
        for i, entry in enumerate(recent_entries):
            if i < len(recent_entries) - 1:  # Has a next entry to check outcome
                current_decision = entry['behavior'].get('decision_confidence', 0)
                next_entry = recent_entries[i + 1]
                # Simple outcome proxy - could be enhanced with actual P&L
                outcome_proxy = next_entry.get('psychology', {})
                if hasattr(outcome_proxy, 'confidence_level'):
                    decision_outcomes.append((current_decision, outcome_proxy.confidence_level))
        
        if len(decision_outcomes) >= 10:
            decisions, outcomes = zip(*decision_outcomes)
            correlation = np.corrcoef(decisions, outcomes)[0, 1]
            
            if abs(correlation) > 0.3:
                insights.append(f"Decision confidence correlates {correlation:.2f} with subsequent outcomes")
                if correlation < -0.3:
                    insights.append("Negative correlation: high confidence leads to poor outcomes — review conviction bias.")
        
        # Pattern recognition in emotional states
        emotional_transitions = []
        for i in range(1, len(recent_entries)):
            prev_emotion = recent_entries[i-1]['behavior'].get('emotional_state')
            curr_emotion = recent_entries[i]['behavior'].get('emotional_state')
            if prev_emotion and curr_emotion:
                emotional_transitions.append((prev_emotion, curr_emotion))
        
        if emotional_transitions:
            transition_counts = defaultdict(int)
            for transition in emotional_transitions:
                transition_counts[transition] += 1
            
            most_common = max(transition_counts.items(), key=lambda x: x[1])
            if most_common[1] >= 3:
                insights.append(f"Most common emotional pattern: {most_common[0][0]} → {most_common[0][1]}")
        
        # Self-prediction accuracy
        if len(self.self_prediction_accuracy) >= 10:
            avg_accuracy = np.mean(self.self_prediction_accuracy)
            if avg_accuracy < 0.6:
                insights.append(f"Low self-prediction accuracy ({avg_accuracy:.2f}) - introspection may be unreliable")
            elif avg_accuracy > 0.8:
                insights.append(f"High self-prediction accuracy ({avg_accuracy:.2f}) - strong self-awareness")
        
        return insights

    def _update_awareness_state(self, deviations: List[BehavioralDeviation], consistency_score: float):
        """Update the agent's self-awareness state"""
        
        # Update active deviations (keep only recent ones)
        current_time = time.time()
        self.awareness_state.active_deviations = [
            d for d in self.awareness_state.active_deviations 
            if current_time - d.timestamp < 600  # Keep for 10 minutes
        ]
        self.awareness_state.active_deviations.extend(deviations)
        
        # Update consistency level
        if consistency_score > 0.8:
            self.awareness_state.consistency_level = ConsistencyLevel.HIGHLY_CONSISTENT
        elif consistency_score > 0.6:
            self.awareness_state.consistency_level = ConsistencyLevel.CONSISTENT
        elif consistency_score > 0.4:
            self.awareness_state.consistency_level = ConsistencyLevel.MODERATELY_INCONSISTENT
        elif consistency_score > 0.2:
            self.awareness_state.consistency_level = ConsistencyLevel.HIGHLY_INCONSISTENT
        else:
            self.awareness_state.consistency_level = ConsistencyLevel.ERRATIC
        
        # Calculate behavioral drift score
        if deviations:
            avg_deviation_severity = float(np.mean([d.severity for d in deviations]))
            self.awareness_state.behavioral_drift_score = 0.7 * self.awareness_state.behavioral_drift_score + 0.3 * avg_deviation_severity
        else:
            self.awareness_state.behavioral_drift_score *= 0.95  # Gradual decay
        
        # Update self-trust level
        trust_adjustment = 0.0
        if consistency_score > 0.7:
            trust_adjustment += 0.02
        if len(deviations) == 0:
            trust_adjustment += 0.01
        if len([d for d in deviations if d.severity > 0.7]) > 0:
            trust_adjustment -= 0.05
        
        self.awareness_state.self_trust_level = np.clip(
            self.awareness_state.self_trust_level + trust_adjustment, 0.0, 1.0
        )
        
        # Update metacognitive confidence based on prediction accuracy
        if len(self.self_prediction_accuracy) >= 5:
            recent_accuracy = np.mean(list(self.self_prediction_accuracy)[-5:])
            self.awareness_state.metacognitive_confidence =float(0.8 * self.awareness_state.metacognitive_confidence + 0.2 * recent_accuracy)

    def _calculate_consistency_score(self) -> float:
        if len(self.behavioral_history) < 10:
            return 0.8  # Default for insufficient data
        recent_behaviors = [entry['behavior'] for entry in list(self.behavioral_history)[-20:]]
        metrics = ['risk_tolerance', 'confidence', 'fear_sensitivity']
        consistency_scores = []
        for metric in metrics:
            values = [b.get(metric) for b in recent_behaviors if metric in b]
            if len(values) >= 5:
                variance = float(np.var(values))
                consistency = max(0.0, 1.0 - variance * 4)
                consistency_scores.append(consistency)
        decisions = [b.get('decision_direction') for b in recent_behaviors[-10:] if 'decision_direction' in b]
        if len(decisions) >= 5:
            switches = sum(1 for i in range(1, len(decisions)) if decisions[i] != decisions[i-1])
            switch_consistency = max(0.0, 1.0 - switches / len(decisions))
            consistency_scores.append(switch_consistency)
        return float(np.mean(consistency_scores) if consistency_scores else 0.8)

    def _generate_awareness_output(self, deviations: List[BehavioralDeviation], introspection_insights: List[str]) -> Dict[str, Any]:
        deviation_summary = {}
        for deviation in deviations:
            dev_type = deviation.deviation_type.value
            if dev_type not in deviation_summary:
                deviation_summary[dev_type] = []
            deviation_summary[dev_type].append({
                'severity': deviation.severity,
                'description': deviation.description,
                'recommendation': deviation.recommendation
            })
        awareness_flags = {
            'behavioral_consistency_warning': self.awareness_state.consistency_level in [
                ConsistencyLevel.HIGHLY_INCONSISTENT, ConsistencyLevel.ERRATIC
            ],
            'high_behavioral_drift': self.awareness_state.behavioral_drift_score > 0.7,
            'low_self_trust': self.awareness_state.self_trust_level < 0.4,
            'metacognitive_uncertainty': self.awareness_state.metacognitive_confidence < 0.5
        }
        return {
            'self_awareness_state': self.awareness_state,
            'behavioral_deviations': deviation_summary,
            'introspection_insights': introspection_insights,
            'consistency_score': self.consistency_tracker[-1] if self.consistency_tracker else 0.8,
            'behavioral_baseline': self.behavioral_baseline,
            'awareness_flags': awareness_flags,
            'self_trust_multiplier': self.awareness_state.self_trust_level,
            'behavioral_drift_score': self.awareness_state.behavioral_drift_score
        }

# Enhanced SyncBus
class SyncBus:
    def __init__(self, agents: List[MirrorAgent]):
        self.agents = agents
        self.tick_count = 0
        self.global_state = {}
        self.state_history = []  # For batch mode
        self.debug_logs = []

    def tick(self, market_data: MarketData):
        """Process one market tick through all agents"""
        self.tick_count += 1

        print(f"\n[SyncBus] Tick #{self.tick_count} - Broadcasting market update...")
        print(f"Market Data: {market_data}")

        data_package = {
            'market_data': market_data,
            'tick_count': self.tick_count,
            'timestamp': time.time()
        }

        for agent in self.agents:
            try:
                agent_output = agent.update(data_package)
                if agent_output:
                    data_package.update(agent_output)

                if hasattr(agent, 'debug'):
                    self.debug_logs.append({agent.name: agent.debug()})

            except Exception as e:
                print(f"[SyncBus] Error in {agent.name}: {str(e)}")
                continue

        self.global_state = data_package.copy()
        self.state_history.append(self.global_state)

        self._print_tick_summary()
        return self.global_state

    def tick_batch(self, data_list: List[MarketData]) -> List[Dict[str, Any]]:
        """Run a batch of ticks for backtesting"""
        for data in data_list:
            self.tick(data)
        return self.state_history

    def _print_tick_summary(self):
        """Print a summary of the current tick"""
        print(f"\n[SyncBus] Tick #{self.tick_count} Summary:")

        market_data = self.global_state.get('market_data')
        if market_data:
            print(f"  Market: ${market_data.price:.2f} | Vol: {market_data.volume} | Structure: {market_data.structure.value}")

        psych_profile = self.global_state.get('psych_profile')
        if psych_profile:
            print(f"  Psychology: {psych_profile.emotional_state.value} | Confidence: {psych_profile.confidence_level:.2f}")

        fear_level = self.global_state.get('fear_level', 0)
        print(f"  Fear Level: {fear_level:.2f}")

        selected_scenario = self.global_state.get('selected_scenario')
        if selected_scenario:
            print(f"  Decision: {selected_scenario.direction.upper()} | Strength: {selected_scenario.strength:.2f}")

        execution_result = self.global_state.get('execution_result')
        if execution_result:
            print(f"  Execution: {execution_result.get('action', 'None')}")

        performance_metrics = self.global_state.get('performance_metrics')
        if performance_metrics and performance_metrics['total_trades'] > 0:
            print(f"  Performance: {performance_metrics['total_trades']} trades | "
                  f"Win Rate: {performance_metrics['win_rate']:.1%} | "
                  f"PnL: {performance_metrics['total_pnl']:.2f}")

        print("-" * 60)

    def plot_performance(self):
        """Visualize performance metrics over time"""
        if not self.state_history:
            print("[SyncBus] No data to plot.")
            return

        timestamps = []
        pnl_curve = []
        fear = []
        confidence = []
        win_rate = []

        cumulative_pnl = 0.0

        for state in self.state_history:
            # Timestamp
            timestamps.append(state.get('timestamp', time.time()))

            # PnL (safely)
            trades = state.get('trade_results')
            if trades and isinstance(trades, list) and trades[-1].get('pnl') is not None:
                pnl = trades[-1].get('pnl', 0)
            else:
                pnl = 0
                cumulative_pnl += pnl
            pnl_curve.append(cumulative_pnl)

            # Emotion state
            fear.append(state.get('fear_level', 0))
            confidence.append(state.get('psych_profile').confidence_level if state.get('psych_profile') else 0.5)

            # Win rate
            win = state.get('performance_metrics', {}).get('win_rate', 0)
            win_rate.append(win)

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax[0].plot(timestamps, pnl_curve, label="PnL", color="green")
        ax[0].set_ylabel("Cumulative PnL")
        ax[0].legend()

        ax[1].plot(timestamps, fear, label="Fear", color="red")
        ax[1].plot(timestamps, confidence, label="Confidence", color="blue")
        ax[1].set_ylabel("Emotion State")
        ax[1].legend()

        ax[2].plot(timestamps, win_rate, label="Win Rate", color="purple")
        ax[2].set_ylabel("Win Rate")
        ax[2].set_xlabel("Time")
        ax[2].legend()

        plt.tight_layout()
        plt.show()


    def get_debug_log(self) -> List[Dict[str, Any]]:
        """Return collected debug logs from agents"""
        return self.debug_logs

# --- Minimal agent stubs for missing classes ---
class DecisionMirror(MirrorAgent):
    def __init__(self):
        super().__init__('DecisionMirror')
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {}

class ExecutionDaemon(MirrorAgent):
    DRY_RUN = True  # Set to False to enable live trading

    def __init__(self, exchange=None, capital=1000, risk_pct=0.01, dry_run=True):
        super().__init__('ExecutionDaemon')
        self.exchange = exchange
        self.capital = capital
        self.risk_pct = risk_pct
        self.trade_log = []
        self.virtual_balance = capital
        self.open_positions = {}
        self.DRY_RUN = dry_run

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        scanner_df = data.get('momentum_df')
        arch_ctrl = data.get('arch_ctrl')
        trades_executed = []
        if scanner_df is not None:
            for _, row in scanner_df.iterrows():
                symbol = row['symbol']
                price = row['price']
                risk_amount = self.capital * self.risk_pct
                position_size = risk_amount / price
                # Emotional gating
                if arch_ctrl and (arch_ctrl.fear > 0.3 or arch_ctrl.stress > 0.3):
                    continue  # Block trade
                allow_high_risk_trade = arch_ctrl and arch_ctrl.confidence > 0.7
                # DRY_RUN logic
                if self.DRY_RUN:
                    print(f"[🧪 DRY-RUN] BUY {symbol} | Size: {position_size:.4f} | Price: {price}")
                    # Simulate position and balance
                    self.open_positions[symbol] = {"entry": price, "size": position_size}
                    self.virtual_balance -= price * position_size
                else:
                    if self.exchange:
                        try:
                            self.exchange.create_market_buy_order(symbol, position_size)
                            print(f"Buy order placed for {symbol} at {price} size {position_size:.4f}")
                        except Exception as e:
                            print(f"Order error for {symbol}: {e}")
                trades_executed.append({
                    "symbol": symbol,
                    "price": price,
                    "size": position_size,
                    "risk": risk_amount,
                    "high_risk": allow_high_risk_trade
                })
                self.trade_log.append({
                    "symbol": symbol,
                    "entry": price,
                    "size": position_size,
                    "risk": risk_amount,
                    "timestamp": datetime.now().isoformat()
                })
        # Simulate closing positions and PnL (for DRY_RUN)
        closed_trades = []
        if self.DRY_RUN and self.open_positions:
            for symbol, pos in list(self.open_positions.items()):
                # Simulate random exit after some ticks (for demo)
                if np.random.rand() < 0.2:  # 20% chance to close
                    exit_price = pos["entry"] * (1 + np.random.normal(0, 0.01))
                    profit = (exit_price - pos["entry"]) * pos["size"]
                    self.virtual_balance += exit_price * pos["size"]
                    print(f"[🧾 DRY-RUN PROFIT] {symbol}: ${profit:.2f} | New Balance: ${self.virtual_balance:.2f}")
                    closed_trades.append({
                        "symbol": symbol,
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "size": pos["size"],
                        "pnl": profit
                    })
                    del self.open_positions[symbol]
        return {
            "execution_result": {"action": "opened", "trades": trades_executed, "closed_trades": closed_trades},
            "trade_log": self.trade_log,
            "virtual_balance": self.virtual_balance,
            "open_positions": self.open_positions
        }

    def execute(self, signal, symbol="BTC/USDT"):
        if self.DRY_RUN:
            print(f"[DRY_RUN] Executing {signal} for {symbol}")
        else:
            print(f"[LIVE] Executing {signal} for {symbol}")

class ReflectionCore(MirrorAgent):
    def __init__(self):
        super().__init__('ReflectionCore')
        self.trade_log = []
        self.pnl_curve = []

    def record_trade(self, symbol, entry_price, exit_price):
        pnl = (exit_price - entry_price) / entry_price
        self.trade_log.append({
            "symbol": symbol,
            "entry": entry_price,
            "exit": exit_price,
            "pnl": pnl
        })
        self.pnl_curve.append(pnl)

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        trades = data.get('execution_result', {}).get('trades', [])
        for trade in trades:
            # Simulate exit price for demo
            exit_price = trade['price'] * (1 + np.random.normal(0, 0.01))
            self.record_trade(trade['symbol'], trade['price'], exit_price)
        return {"trade_log": self.trade_log, "pnl_curve": self.pnl_curve}

# --- Integration helper function stub ---
def integrate_selfawareness_into_mirrorcore(mirrorcore_system: SyncBus) -> SyncBus:
    # Already included in agent list, so just return system
    return mirrorcore_system
class MarketDataGenerator:
    def __init__(self, initial_price: float = 3337.0):
        self.current_price = initial_price
        self.trend = 0.0
        self.volatility = 0.02
        self.tick_count = 0
        self.sentiment_bias = 0.0  # Optional external control (e.g., from ego)

    def inject_sentiment(self, bias: float):
        """Inject external sentiment bias: -1.0 (bearish) to +1.0 (bullish)"""
        self.sentiment_bias = max(-1.0, min(1.0, bias))

    def generate_tick(self) -> MarketData:
        self.tick_count += 1

        # Adjust volatility regime randomly every 30 ticks
        if self.tick_count % 30 == 0:
            self.volatility *= np.random.choice([0.8, 1.2])

        # Apply trend + sentiment + noise
        news_shock = np.random.normal(0, 0.0015)
        trend_bias = self.trend + self.sentiment_bias * 0.001
        price_change = np.random.normal(trend_bias, self.volatility) * self.current_price + news_shock

        self.current_price = max(0.01, self.current_price + price_change)

        # Generate OHLC (simplified)
        high = self.current_price * (1 + abs(np.random.normal(0, 0.001)))
        low = self.current_price * (1 - abs(np.random.normal(0, 0.001)))
        open_price = self.current_price + np.random.normal(0, 0.001) * self.current_price

        # Volume simulation with spikes
        base_volume = 1000
        volatility_factor = min(2.0, abs(price_change / self.current_price) * 10)
        volume = int(base_volume * (1 + np.random.normal(volatility_factor, 0.2)))

        # Market structure classification
        move_ratio = abs(price_change / self.current_price)
        if move_ratio > 0.03:
            structure = MarketStructure.VOLATILE
        elif price_change > 0.01:
            structure = MarketStructure.TRENDING_UP
        elif price_change < -0.01:
            structure = MarketStructure.TRENDING_DOWN
        else:
            structure = MarketStructure.RANGING

        # Change underlying trend every 25 ticks
        if self.tick_count % 25 == 0:
            self.trend = np.random.normal(0, 0.002)

        return MarketData(
            timestamp=time.time(),
            price=self.current_price,
            volume=volume,
            high=high,
            low=low,
            open=open_price,
            volatility=self.volatility,
            structure=structure
        )



def create_mirrorcore_system(scanner) -> Tuple[SyncBus, MarketDataGenerator]:
    """Create and initialize the MirrorCore-X system with sentiment feedback loop"""
    print("🔮 Initializing MirrorCore-X: Cognitive Trading Organism")
    print("=" * 60)

    # Initialize all agents
    ego = EgoProcessor()
    agents = [
        PerceptionLayer(scanner),
        ego,
        FearAnalyzer(),
        SelfAwarenessAgent(),
        DecisionMirror(),
        ExecutionDaemon(),
        ReflectionCore(),
        MirrorMindMetaAgent(),
        MetaAgent(),
        RiskSentinel()
    ]

    # Create the sync bus
    mirrorcore = SyncBus(agents)

    # Create market data generator
    market_generator = MarketDataGenerator(initial_price=3337.0)

    # Attach hook to inject sentiment from ego into the market generator
    original_tick = mirrorcore.tick

    def sentiment_aware_tick(market_data: MarketData):
        # Inject emotional bias from ego into generator before tick
        psych_state = getattr(ego, "last_profile", None)
        if psych_state:
            # Convert emotion + confidence into sentiment: [-1, +1]
            emotion_score = {
                'CALM': 0.0, 'FEARFUL': -0.7, 'GREEDY': 0.7,
                'ANXIOUS': -0.4, 'CONFIDENT': 0.6
            }.get(psych_state.emotional_state.name.upper(), 0.0)
            sentiment_bias = emotion_score * psych_state.confidence_level
            market_generator.inject_sentiment(sentiment_bias)
        return original_tick(market_data)

    mirrorcore.tick = sentiment_aware_tick

    print("✅ System initialized with the following cognitive agents:")
    for agent in agents:
        print(f"   • {agent.name}")

    print("\n🧠 MirrorCore-X is now wired for emotional market feedback...")
    print("=" * 60)

    return mirrorcore, market_generator

def run_demo_session(mirrorcore: SyncBus, market_generator: MarketDataGenerator, num_ticks: int = 25, save_report: bool = False):
    """Run a demonstration session of MirrorCore-X"""
    print(f"\n🚀 Starting demo session with {num_ticks} market ticks...")
    print("=" * 60)

    session_log = []

    for i in range(num_ticks):
        # Generate market data
        market_data = market_generator.generate_tick()

        # Process through MirrorCore-X
        try:
            global_state = mirrorcore.tick(market_data)
            execution = global_state.get('execution_result', {}).get('action', 'None')
            scenario = global_state.get('selected_scenario')
            decision_dir = scenario.direction if scenario else "none"

            print(f"Tick #{i+1:02d} | Action: {execution.upper()} | Direction: {decision_dir.upper()} | Price: {market_data.price:.2f}")
            session_log.append({
                'tick': i + 1,
                'price': market_data.price,
                'action': execution,
                'direction': decision_dir,
                'pnl': global_state.get('performance_metrics', {}).get('total_pnl', 0)
            })

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n⏹️  Demo session interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error during tick {i + 1}: {str(e)}")
            continue

    # Final summary
    print("\n📊 FINAL SESSION SUMMARY")
    print("=" * 60)
    final_state = mirrorcore.global_state
    performance = final_state.get('performance_metrics', {})

    if performance.get('total_trades', 0) > 0:
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.1%}")
        print(f"Total P&L: ${performance['total_pnl']:.2f}")
        print(f"Average Win: ${performance['avg_win']:.2f}")
        print(f"Average Loss: ${performance['avg_loss']:.2f}")
        print(f"Max Drawdown: ${performance['max_drawdown']:.2f}")
    else:
        print("No trades executed during this session")

    # Psychological state
    psych_profile = final_state.get('psych_profile')
    if psych_profile:
        print(f"\nFinal Psychological State:")
        print(f"  Emotional State: {psych_profile.emotional_state.value}")
        print(f"  Confidence Level: {psych_profile.confidence_level:.2f}")
        print(f"  Stress Level: {psych_profile.stress_level:.2f}")
        print(f"  Risk Tolerance: {psych_profile.risk_tolerance:.2f}")

    # Insights
    insights = final_state.get('insights', [])
    if insights:
        print(f"\n💡 System Insights:")
        for insight in insights:
            print(f"  • {insight}")

    print("\n🔮 MirrorCore-X session complete. The cognitive organism has evolved.")
    print("=" * 60)

    # Plot performance
    mirrorcore.plot_performance()

    # Save session log (optional)
    if save_report:
        import json
        with open("mirrorcore_session_log.json", "w") as f:
            json.dump(session_log, f, indent=2)
        print("📝 Session log saved to 'mirrorcore_session_log.json'")

# Meta Agent for Oversight
class MirrorMindMetaAgent(MirrorAgent):
    def __init__(self):
        super().__init__('MirrorMindMetaAgent')
        self.history = []

    def generate_insights(self, ego_state, fear_state, self_state, decision_count, execution_count):
        insights = []
        # --- Emotional State Analysis ---
        if ego_state["confidence"] > 0.7 and fear_state["fear"] < 0.2:
            insights.append("System is confident and fearless — optimal conditions for assertive trading.")
        elif ego_state["stress"] > 0.7:
            insights.append("High stress detected — consider reducing position size or pausing decisions.")
        elif ego_state["confidence"] < 0.3:
            insights.append("Low confidence — system is unsure. Might be missing market clarity.")
        # --- Self Awareness ---
        if self_state["drift"] > 0.05:
            insights.append("Agent drift detected — internal coherence weakening.")
        if self_state["trust"] > 0.9:
            insights.append("High self-trust indicates consistent behavior.")
        elif self_state["trust"] < 0.6:
            insights.append("Trust is falling — agents may be diverging in behavior.")
        # --- Action Analysis ---
        if decision_count == 0:
            insights.append("No decisions made — market may be flat or filters too strict.")
        if execution_count > 0 and fear_state["fear"] > 0.5:
            insights.append("Executing trades in high-fear regime — risky behavior.")
        # --- Grade the Session ---
        grade = "A"
        if self_state["trust"] < 0.6 or ego_state["stress"] > 0.8:
            grade = "C"
        elif decision_count == 0 and ego_state["confidence"] < 0.5:
            grade = "B"
        elif fear_state["fear"] > 0.7:
            grade = "B-"
        # Store + Return
        self.history.append({"grade": grade, "insights": insights})
        return grade, insights

    def summarize_session(self):
        grades = [h["grade"] for h in self.history[-10:]]
        insights = sum((h["insights"] for h in self.history[-10:]), [])
        most_common = Counter(insights).most_common(3)
        print("\n[MirrorMindMetaAgent] 🔁 Insight Summary (Last 10 Ticks):")
        print(" - Frequent Patterns:")
        for insight, count in most_common:
            print(f"   • {insight} ({count} times)")
        if grades:
            avg_grade = sum(ord(g[0]) for g in grades) / len(grades)
            print(" - Average Grade:", avg_grade)
        else:
            print(" - No grades to summarize.")

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract states from other agents
        ego_state = data.get('ego_state', {"confidence": 0.5, "stress": 0.5})
        fear_state = data.get('fear_state', {"fear": 0.0, "regime": "calm"})
        self_state = data.get('self_state', {"drift": 0.0, "trust": 0.8, "deviations": 0})
        decision_count = data.get('decision_count', 0)
        execution_count = data.get('execution_count', 0)
        grade, insights = self.generate_insights(ego_state, fear_state, self_state, decision_count, execution_count)
        print(f"[MirrorMindMetaAgent] Session Grade: {grade} | Insights: {insights}")
        return {
            'meta_insights': insights,
            'session_grade': grade
        }

class MetaAgent(MirrorAgent):
    def __init__(self):
        super().__init__('MetaAgent')
        self.agent_scores = {}

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        performance = data.get('performance_metrics', {})
        fear_level = data.get('fear_level', 0.0)
        psych_profile = data.get('psych_profile')
        confidence = psych_profile.confidence_level if psych_profile else 0.5


        execution_weight = 1.0 - fear_level
        decision_weight = confidence * (1.0 - fear_level)

        print(f"[{self.name}] Adjusting weights — Execution: {execution_weight:.2f}, Decision: {decision_weight:.2f}")

        return {
            'meta_weights': {
                'execution_weight': execution_weight,
                'decision_weight': decision_weight
            }
        }

class RiskSentinel(MirrorAgent):
    def __init__(self, max_drawdown: float = -500.0, max_open_positions: int = 10):
        super().__init__('RiskSentinel')
        self.max_drawdown = max_drawdown
        self.max_open_positions = max_open_positions

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        performance = data.get('performance_metrics', {})
        active_positions = data.get('active_positions', 0)

        pnl = performance.get('total_pnl', 0.0)
        if pnl <= self.max_drawdown:
            print(f"[{self.name}] Max drawdown reached — Disabling new trades")
            return {'risk_block': True}

        if active_positions >= self.max_open_positions:
            print(f"[{self.name}] Position limit reached — Holding execution")
            return {'risk_block': True}

        return {'risk_block': False}

import time
import numpy as np
import ccxt
from arch_ctrl import ARCH_CTRL

if __name__ == "__main__":
    # --- MirrorCore-X + Oracle Integration Demo ---
    exchange = ccxt.kucoinfutures({'enableRateLimit': True})
    scanner = MomentumScanner(exchange)
    mirrorcore_system, market_gen = create_mirrorcore_system(scanner)
    mirrorcore_system = integrate_selfawareness_into_mirrorcore(mirrorcore_system)
    # Attach scanner to mirrorcore for Oracle integration
    mirrorcore_system.scanner = scanner
    # Attach arch_ctrl for Oracle context
    arch_ctrl = ARCH_CTRL()
    mirrorcore_system.arch_ctrl = arch_ctrl
    # Attach risk sentinel stub if needed
    if not hasattr(mirrorcore_system, 'risk_sentinel'):
        mirrorcore_system.risk_sentinel = RiskSentinel()
    # Attach self-awareness stub if needed
    if not hasattr(mirrorcore_system, 'self_awareness'):
        mirrorcore_system.self_awareness = SelfAwarenessAgent()
    # Attach trade analyzer stub if needed
    if not hasattr(mirrorcore_system, 'trade_analyzer'):
        mirrorcore_system.trade_analyzer = TradeAnalyzerAgent()
    # --- Oracle Integration ---
    oracle_core = OracleEnhancedMirrorCore(mirrorcore_system, user_id="TRADER_001")
    # Example: Run an Oracle-enhanced tick
    print("\n=== Oracle-Enhanced MirrorCore-X Demo ===")
    oracle_result = oracle_core.enhanced_tick(timeframe="7day")
    print(f"Oracle Result: {oracle_result}")
