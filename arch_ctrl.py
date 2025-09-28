
import time
import logging
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)

class ARCH_CTRL:
    """Enhanced ARCH_CTRL compatible with high-performance SyncBus"""
    
    def __init__(self):
        self.fear = 0.0
        self.stress = 0.0
        self.confidence = 0.5
        self.trust = 0.8
        self.override = False
        self.history = []
        self.risk_appetite = 0.5  # Dynamic risk appetite
        self.architect_injected = False
        
        # Enhanced SyncBus compatibility
        self.name = "ARCH_CTRL"
        self.data_interests = ['emotional_state', 'market_data', 'risk']
        self.is_paused = False
        self.command_queue = []
        self.last_update = time.time()
        
        # Enhanced emotional state tracking
        self.emotional_momentum = 0.0
        self.stress_triggers = []
        self.confidence_history = []
        self.intervention_count = 0
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced update method compatible with SyncBus architecture"""
        # Process commands first
        await self._process_commands()
        
        if self.is_paused:
            return {'status': 'paused', 'emotional_state': self.get_emotional_state()}
        
        try:
            # Extract relevant data for emotional processing
            market_volatility = 0.0
            loss_occurred = False
            architect_override = self.override
            
            # Process market data for volatility
            if 'market_data' in data:
                market_data = data['market_data']
                if isinstance(market_data, list) and market_data:
                    latest_data = market_data[-1]
                    market_volatility = latest_data.get('volatility', 0.0)
                elif isinstance(market_data, dict):
                    market_volatility = market_data.get('volatility', 0.0)
            
            # Check for losses in trade data
            if 'trades' in data:
                trades = data['trades']
                if isinstance(trades, list):
                    for trade in trades:
                        if isinstance(trade, dict) and trade.get('pnl', 0) < 0:
                            loss_occurred = True
                            break
                elif isinstance(trades, dict) and trades.get('pnl', 0) < 0:
                    loss_occurred = True
            
            # Process risk signals
            if 'risk' in data:
                risk_data = data['risk']
                if isinstance(risk_data, dict):
                    volatility_anomaly = risk_data.get('volume_anomaly', False)
                    if volatility_anomaly:
                        market_volatility = max(market_volatility, 0.05)
            
            # Update emotional state
            self.update_emotions(market_volatility, loss_occurred, architect_override)
            
            self.last_update = time.time()
            
            return {
                'emotional_state': self.get_emotional_state(),
                'insights': self.generate_insights(),
                'trade_allowed': self.allow_action(),
                'status': 'active',
                'last_update': self.last_update,
                'confidence': self.confidence,
                'risk_appetite': self.risk_appetite
            }
            
        except Exception as e:
            logger.error(f"ARCH_CTRL update failed: {e}")
            return {
                'status': 'error', 
                'error': str(e),
                'emotional_state': self.get_emotional_state()
            }
    
    async def _process_commands(self):
        """Process commands from SyncBus"""
        while self.command_queue:
            command_msg = self.command_queue.pop(0)
            command = command_msg.get('command')
            params = command_msg.get('params', {})
            
            if command == 'pause':
                self.is_paused = True
                logger.info("ARCH_CTRL paused")
            elif command == 'resume':
                self.is_paused = False
                logger.info("ARCH_CTRL resumed")
            elif command == 'emergency_stop':
                self.is_paused = True
                self.override = True  # Emergency override
                self.fear = 1.0  # Maximum fear
                self.confidence = 0.1  # Minimum confidence
                logger.warning("ARCH_CTRL emergency stopped - emotional lockdown activated")
            elif command == 'reset_emotions':
                self._reset_emotional_state()
                logger.info("ARCH_CTRL emotional state reset")
            elif command == 'set_override':
                override_state = params.get('state', False)
                self.set_override(override_state)
                logger.info(f"ARCH_CTRL override set to {override_state}")
            elif command == 'inject_emotion':
                # Allow external emotional injection
                if 'fear' in params:
                    self.inject_emotion(fear=params['fear'])
                if 'confidence' in params:
                    self.inject_emotion(confidence=params['confidence'])
                if 'stress' in params:
                    self.inject_emotion(stress=params['stress'])
                logger.info("ARCH_CTRL emotions injected")
    
    def update_emotions(self, market_volatility: float, loss_occurred: bool = False, architect_override: bool = False):
        """Enhanced emotional update with better tracking"""
        # Store previous state for momentum calculation
        prev_fear = self.fear
        prev_confidence = self.confidence
        
        # Adjust emotion state based on outcomes and conditions
        if loss_occurred:
            self.fear += 0.1
            self.stress += 0.1
            self.confidence -= 0.1
            self.stress_triggers.append(time.time())
        else:
            self.fear = max(self.fear - 0.02, 0)
            self.stress = max(self.stress - 0.01, 0)
            self.confidence = min(self.confidence + 0.02, 1)
        
        # Market volatility impact
        if market_volatility > 0.03:
            self.fear += 0.05
            self.stress += 0.05
            self.stress_triggers.append(time.time())
        
        # Dynamic risk appetite calculation
        self.risk_appetite = max(0.01, min(1.0, self.confidence * (1.0 - self.fear)))
        
        # Architect override effects
        if architect_override or self.override:
            self.fear *= 0.25
            self.stress *= 0.25
            self.confidence = max(self.confidence, 0.9)
            self.architect_injected = True
            self.intervention_count += 1
        else:
            self.architect_injected = False
        
        # Clamp values to [0, 1]
        self.fear = min(max(self.fear, 0.0), 1.0)
        self.stress = min(max(self.stress, 0.0), 1.0)
        self.confidence = min(max(self.confidence, 0.0), 1.0)
        
        # Calculate emotional momentum
        self.emotional_momentum = (self.fear - prev_fear) + (self.confidence - prev_confidence)
        
        # Store confidence history for trend analysis
        self.confidence_history.append(self.confidence)
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]
        
        # Clean old stress triggers (keep only last hour)
        current_time = time.time()
        self.stress_triggers = [t for t in self.stress_triggers if current_time - t < 3600]
        
        # Log to internal state history
        self.history.append({
            "fear": self.fear,
            "stress": self.stress,
            "confidence": self.confidence,
            "trust": self.trust,
            "override": self.override,
            "risk_appetite": self.risk_appetite,
            "architect_injected": self.architect_injected,
            "emotional_momentum": self.emotional_momentum,
            "timestamp": current_time
        })
        
        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
    
    def update(self, market_volatility: float, loss_occurred: bool = False, architect_override: bool = False):
        """Legacy update method for backward compatibility"""
        self.update_emotions(market_volatility, loss_occurred, architect_override)
    
    def allow_action(self, max_fear: float = 0.7, min_confidence: float = 0.5, allow_high_risk: bool = False):
        """Enhanced emotional gating for trade execution"""
        # Use attributes directly
        if getattr(self, 'override', False) or getattr(self, 'architect_injected', False):
            return True
        
        if self.fear > max_fear or self.confidence < min_confidence:
            return False
            
        if allow_high_risk and self.confidence > 0.8:
            return True
        
        # Additional checks for extreme conditions
        if len(self.stress_triggers) > 5:  # Too many recent stress events
            return False
            
        if self.emotional_momentum < -0.3:  # Rapidly declining emotional state
            return False
        
        # Default logic
        return self.confidence > 0.5 and self.fear < 0.7
    
    def set_override(self, state: bool):
        """Set override state with logging"""
        self.override = state
        logger.info(f"[ARCH_CTRL] Override set to {state} by Architect.")
    
    def get_emotional_state(self):
        """Enhanced emotional state with additional metrics"""
        return {
            "fear": self.fear,
            "stress": self.stress,
            "confidence": self.confidence,
            "trust": self.trust,
            "risk_appetite": self.risk_appetite,
            "override": self.override,
            "architect_injected": self.architect_injected,
            "emotional_momentum": self.emotional_momentum,
            "stress_event_count": len(self.stress_triggers),
            "intervention_count": self.intervention_count,
            "confidence_trend": self._calculate_confidence_trend()
        }
    
    def _calculate_confidence_trend(self) -> str:
        """Calculate confidence trend from recent history"""
        if len(self.confidence_history) < 10:
            return "insufficient_data"
        
        recent = self.confidence_history[-10:]
        older = self.confidence_history[-20:-10] if len(self.confidence_history) >= 20 else recent
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    def generate_insights(self):
        """Enhanced insights with more conditions"""
        insights = []
        
        if self.fear > 0.8 and not self.override:
            insights.append("🚨 Agent paralyzed by extreme fear. Architect intervention recommended.")
        elif self.fear > 0.6:
            insights.append("⚠️ Elevated fear levels detected. Trading caution advised.")
        
        if self.override:
            insights.append("🎯 Architect override active — emotional gates bypassed.")
        
        if self.confidence > 0.85 and self.fear < 0.2:
            insights.append("🚀 Emotionally stable and confident — primed for bold action.")
        elif self.confidence < 0.3:
            insights.append("📉 Low confidence detected. Conservative approach recommended.")
        
        if self.stress > 0.75:
            insights.append("🔥 Elevated stress — high volatility environment detected.")
        
        if len(self.stress_triggers) > 3:
            insights.append(f"⚡ {len(self.stress_triggers)} stress events in last hour. System under pressure.")
        
        if self.trust < 0.6:
            insights.append("🔍 Trust integrity at risk — agent divergence possible.")
        
        if self.emotional_momentum < -0.2:
            insights.append("📉 Negative emotional momentum detected. System stability declining.")
        elif self.emotional_momentum > 0.2:
            insights.append("📈 Positive emotional momentum. System confidence building.")
        
        insights.append(f"💪 Current risk appetite: {self.risk_appetite:.2f}")
        
        if self.architect_injected:
            insights.append("👑 Architect has injected override this tick.")
        
        # Add confidence trend insight
        trend = self._calculate_confidence_trend()
        if trend == "improving":
            insights.append("📈 Confidence trend improving over recent activity.")
        elif trend == "declining":
            insights.append("📉 Confidence trend declining. Monitor closely.")
        
        return insights
    
    def get_insights(self):
        """Legacy method for backward compatibility"""
        return self.generate_insights()
    
    def inject_emotion(self, fear=None, stress=None, confidence=None, trust=None):
        """Enhanced emotional injection with validation"""
        if fear is not None:
            self.fear = min(max(float(fear), 0.0), 1.0)
        if stress is not None:
            self.stress = min(max(float(stress), 0.0), 1.0)
        if confidence is not None:
            self.confidence = min(max(float(confidence), 0.0), 1.0)
        if trust is not None:
            self.trust = min(max(float(trust), 0.0), 1.0)
        
        # Recalculate risk appetite
        self.risk_appetite = max(0.01, min(1.0, self.confidence * (1.0 - self.fear)))
        
        # Log injection
        self.history.append({
            "fear": self.fear,
            "stress": self.stress,
            "confidence": self.confidence,
            "trust": self.trust,
            "override": self.override,
            "risk_appetite": self.risk_appetite,
            "architect_injected": True,  # Mark as injected
            "timestamp": time.time(),
            "injection": True
        })
        
        logger.info(f"Emotional injection: fear={fear}, stress={stress}, confidence={confidence}, trust={trust}")
    
    def get_grade(self):
        """Enhanced grading system"""
        grade = 'B'  # Default grade
        
        if self.fear > 0.8:
            grade = 'F'  # Extreme fear
        elif self.fear > 0.6:
            grade = 'D'  # High fear
        elif self.confidence < 0.3:
            grade = 'D'  # Very low confidence
        elif self.confidence < 0.5:
            grade = 'C'  # Low confidence
        elif self.stress > 0.8:
            grade = 'D'  # Extreme stress
        elif self.trust < 0.4:
            grade = 'F'  # Very low trust
        elif self.trust < 0.6:
            grade = 'D'  # Low trust
        elif self.fear < 0.2 and self.confidence > 0.8 and self.trust > 0.8:
            grade = 'A+'  # Excellent state
        elif self.fear < 0.3 and self.confidence > 0.7:
            grade = 'A'   # Very good state
        elif len(self.stress_triggers) > 5:
            grade = 'D'  # Too many stress events
        
        return grade
    
    def _reset_emotional_state(self):
        """Reset emotional state to defaults"""
        self.fear = 0.0
        self.stress = 0.0
        self.confidence = 0.5
        self.trust = 0.8
        self.override = False
        self.risk_appetite = 0.5
        self.architect_injected = False
        self.emotional_momentum = 0.0
        self.stress_triggers.clear()
        self.confidence_history.clear()
        self.intervention_count = 0
        self.history.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status for monitoring"""
        return {
            'name': self.name,
            'is_paused': self.is_paused,
            'emotional_state': self.get_emotional_state(),
            'grade': self.get_grade(),
            'trade_allowed': self.allow_action(),
            'last_update': self.last_update,
            'data_interests': self.data_interests,
            'total_history_entries': len(self.history)
        }


class ArchitectConsole:
    """Enhanced Architect Console with SyncBus integration capabilities"""
    
    def __init__(self, arch_ctrl: ARCH_CTRL):
        self.arch_ctrl = arch_ctrl
        self.command_history = []
        
    def set_override(self, state: bool):
        """Set override state"""
        self.arch_ctrl.set_override(state)
        self.command_history.append({
            'command': 'set_override',
            'params': {'state': state},
            'timestamp': time.time()
        })
    
    def inject_emotions(self, fear=None, stress=None, confidence=None, trust=None):
        """Inject emotional states"""
        self.arch_ctrl.inject_emotion(fear=fear, stress=stress, confidence=confidence, trust=trust)
        self.command_history.append({
            'command': 'inject_emotions',
            'params': {'fear': fear, 'stress': stress, 'confidence': confidence, 'trust': trust},
            'timestamp': time.time()
        })
    
    def emergency_shutdown(self):
        """Emergency emotional shutdown"""
        self.arch_ctrl.fear = 1.0
        self.arch_ctrl.confidence = 0.1
        self.arch_ctrl.override = True
        self.command_history.append({
            'command': 'emergency_shutdown',
            'timestamp': time.time()
        })
        logger.warning("Emergency emotional shutdown activated by Architect")
    
    def reset_system(self):
        """Reset emotional system"""
        self.arch_ctrl._reset_emotional_state()
        self.command_history.append({
            'command': 'reset_system',
            'timestamp': time.time()
        })
        logger.info("Emotional system reset by Architect")
    
    def get_command_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history"""
        return self.command_history[-last_n:]
    
    def display_emotional_dashboard(self):
        """Display real-time emotional dashboard"""
        state = self.arch_ctrl.get_emotional_state()
        
        print("\n" + "="*60)
        print("🧠 ARCH_CTRL EMOTIONAL DASHBOARD")
        print("="*60)
        print(f"😨 Fear:       {state['fear']:.2f} {'🔴' if state['fear'] > 0.7 else '🟡' if state['fear'] > 0.4 else '🟢'}")
        print(f"😰 Stress:     {state['stress']:.2f} {'🔴' if state['stress'] > 0.7 else '🟡' if state['stress'] > 0.4 else '🟢'}")
        print(f"💪 Confidence: {state['confidence']:.2f} {'🟢' if state['confidence'] > 0.7 else '🟡' if state['confidence'] > 0.4 else '🔴'}")
        print(f"🤝 Trust:      {state['trust']:.2f} {'🟢' if state['trust'] > 0.7 else '🟡' if state['trust'] > 0.5 else '🔴'}")
        print(f"🎯 Risk App:   {state['risk_appetite']:.2f}")
        print(f"👑 Override:   {'✅ ACTIVE' if state['override'] else '❌ INACTIVE'}")
        print(f"📈 Momentum:   {state['emotional_momentum']:+.2f}")
        print(f"⚠️ Stress Events: {state['stress_event_count']}")
        print(f"📊 Grade:      {self.arch_ctrl.get_grade()}")
        print(f"✅ Trade OK:   {'YES' if self.arch_ctrl.allow_action() else 'NO'}")
        print("="*60)
        
        # Show insights
        insights = self.arch_ctrl.generate_insights()
        if insights:
            print("💭 Current Insights:")
            for insight in insights:
                print(f"   • {insight}")
            print("="*60)


# Example usage
if __name__ == "__main__":
    # Test the enhanced ARCH_CTRL
    async def test_arch_ctrl():
        arch = ARCH_CTRL()
        console = ArchitectConsole(arch)
        
        # Test update with various scenarios
        test_scenarios = [
            {'market_data': {'volatility': 0.05}, 'trades': []},  # High volatility
            {'trades': [{'pnl': -50}]},  # Loss occurred
            {'market_data': {'volatility': 0.01}},  # Low volatility
        ]
        
        for i, scenario in enumerate(test_scenarios):
            result = await arch.update(scenario)
            print(f"Scenario {i+1} result: {result}")
            console.display_emotional_dashboard()
        
        # Test commands
        arch.command_queue.append({'command': 'set_override', 'params': {'state': True}})
        result = await arch.update({})
        print(f"Override result: {result}")
        
        # Test emergency stop
        arch.command_queue.append({'command': 'emergency_stop'})
        result = await arch.update({})
        print(f"Emergency stop result: {result}")
        
        # Test reset
        arch.command_queue.append({'command': 'reset_emotions'})
        result = await arch.update({})
        print(f"Reset result: {result}")
        
        console.display_emotional_dashboard()
    
    import asyncio
    asyncio.run(test_arch_ctrl())
