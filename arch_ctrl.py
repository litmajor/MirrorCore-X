class ARCH_CTRL:
    def allow_action(self):
        # Emotional gating for trade execution
        return self.override or (self.fear < 0.7 and self.confidence > 0.5)
    def generate_insights(self):
        insights = []
        if self.fear > 0.8 and not self.override:
            insights.append("Agent paralyzed by fear. Awaiting Architect's intervention.")
        if self.override:
            insights.append("Architect override active — emotional gates bypassed.")
        if self.confidence > 0.85 and self.fear < 0.2:
            insights.append("Emotionally stable and confident — primed for bold action.")
        if self.stress > 0.75:
            insights.append("Elevated stress — caution advised.")
        if self.trust < 0.6:
            insights.append("Trust integrity at risk — agent divergence possible.")
        insights.append(f"Current risk appetite: {self.risk_appetite:.2f}")
        if self.architect_injected:
            insights.append("Architect has injected override this tick.")
        return insights
    def __init__(self):
        self.fear = 0.0
        self.stress = 0.0
        self.confidence = 0.5
        self.trust = 0.8
        self.override = False
        self.history = []
        self.risk_appetite = 0.5  # Dynamic risk appetite
        self.architect_injected = False
    def update(self, market_volatility: float, loss_occurred: bool = False, architect_override: bool = False):
        # Adjust emotion state based on outcomes and conditions
        if loss_occurred:
            self.fear += 0.1
            self.stress += 0.1
            self.confidence -= 0.1
        else:
            self.fear = max(self.fear - 0.02, 0)
            self.stress = max(self.stress - 0.01, 0)
            self.confidence = min(self.confidence + 0.02, 1)

        if market_volatility > 0.03:
            self.fear += 0.05
            self.stress += 0.05

        # Dynamic risk appetite
        self.risk_appetite = max(0.01, min(1.0, self.confidence * (1.0 - self.fear)))

        # Architect override
        if architect_override or self.override:
            self.fear *= 0.25
            self.stress *= 0.25
            self.confidence = max(self.confidence, 0.9)
            self.architect_injected = True
        else:
            self.architect_injected = False

        # Clamp values to [0, 1]
        self.fear = min(max(self.fear, 0.0), 1.0)
        self.stress = min(max(self.stress, 0.0), 1.0)
        self.confidence = min(max(self.confidence, 0.0), 1.0)

        # Log to internal state history
        if hasattr(self, 'history'):
            self.history.append({
                "fear": self.fear,
                "stress": self.stress,
                "confidence": self.confidence,
                "override": self.override,
                "risk_appetite": self.risk_appetite,
                "architect_injected": self.architect_injected
            })

    def get_insights(self):
        return self.generate_insights()
        """Advanced emotional throttle for trade execution."""
        # Use attributes directly
        if getattr(self, 'override', False) or getattr(self, 'architect_injected', False):
            return True
        if self.fear > max_fear or self.confidence < min_confidence:
            return False
        if allow_high_risk and self.confidence > 0.8:
            return True
        # Use method via getattr
        allow_action_method = getattr(self, 'allow_action', None)
        if callable(allow_action_method):
            return allow_action_method()
        return self.confidence > 0.5 and self.fear < 0.7

    def inject_emotion(self, fear=None, stress=None, confidence=None, trust=None):
        """Directly inject emotional state (for Architect Console or agent feedback)."""
        if fear is not None:
            self.fear = min(max(fear, 0.0), 1.0)
        if stress is not None:
            self.stress = min(max(stress, 0.0), 1.0)
        if confidence is not None:
            self.confidence = min(max(confidence, 0.0), 1.0)
        if trust is not None:
            self.trust = min(max(trust, 0.0), 1.0)
        self.risk_appetite = max(0.01, min(1.0, self.confidence * (1.0 - self.fear)))
        if hasattr(self, 'history'):
            self.history.append({
                "fear": self.fear,
                "stress": self.stress,
                "confidence": self.confidence,
                "trust": self.trust,
                "override": getattr(self, 'override', False),
                "risk_appetite": self.risk_appetite,
                "architect_injected": getattr(self, 'architect_injected', False)
            })

    def get_grade(self):
        """Calculate a grade based on emotional state.""" 
        grade = 'B'  # Default grade
        if self.fear > 0.8:
            grade = 'C'
        elif self.confidence < 0.5:
            grade = 'D'
        elif self.fear < 0.2 and self.confidence > 0.8:
            grade = 'A'
        elif self.stress > 0.75:
            grade = 'C'
        elif self.trust < 0.6:
            grade = 'D'
        return grade


class ArchitectConsole:
    def __init__(self, arch_ctrl):
        self.arch_ctrl = arch_ctrl
        self.fear = 0.0
        self.stress = 0.0
        self.confidence = 0.5
        self.trust = 0.8
        self.override = False
        self.history = []
        self.risk_appetite = 0.5
        self.architect_injected = False

    def update(self, market_volatility: float, loss_occurred: bool = False, architect_override: bool = False):
        # Adjust emotion state based on outcomes and conditions
        if loss_occurred:
            self.fear += 0.1
            self.stress += 0.1
            self.confidence -= 0.1
        else:
            self.fear = max(self.fear - 0.02, 0)
            self.stress = max(self.stress - 0.01, 0)
            self.confidence = min(self.confidence + 0.02, 1)

        if market_volatility > 0.03:
            self.fear += 0.05
            self.stress += 0.05

        # Dynamic risk appetite
        self.risk_appetite = max(0.01, min(1.0, self.confidence * (1.0 - self.fear)))

        # Architect override
        if architect_override or self.override:
            self.fear *= 0.25
            self.stress *= 0.25
            self.confidence = max(self.confidence, 0.9)
            self.architect_injected = True
        else:
            self.architect_injected = False

        # Clamp values to [0, 1]
        self.fear = min(max(self.fear, 0.0), 1.0)
        self.stress = min(max(self.stress, 0.0), 1.0)
        self.confidence = min(max(self.confidence, 0.0), 1.0)

        # Log to internal state history
        self.history.append({
            "fear": self.fear,
            "stress": self.stress,
            "confidence": self.confidence,
            "override": self.override,
            "risk_appetite": self.risk_appetite,
            "architect_injected": self.architect_injected
        })

    def set_override(self, state: bool):
        self.override = state
        print(f"[ARCH_CTRL] Override set to {state} by Architect.")

    def allow_action(self):
        # Emotional gating for trade execution
        return self.override or (self.fear < 0.7 and self.confidence > 0.5)

    def get_emotional_state(self):
        # Expose current emotional state for agent access
        return {
            "fear": self.fear,
            "stress": self.stress,
            "confidence": self.confidence,
            "trust": self.trust,
            "risk_appetite": self.risk_appetite,
            "override": self.override,
            "architect_injected": self.architect_injected
        }

    def generate_insights(self):
        insights = []
        if self.fear > 0.8 and not self.override:
            insights.append("Agent paralyzed by fear. Awaiting Architect's intervention.")
        if self.override:
            insights.append("Architect override active — emotional gates bypassed.")
        if self.confidence > 0.85 and self.fear < 0.2:
            insights.append("Emotionally stable and confident — primed for bold action.")
        if self.stress > 0.75:
            insights.append("Elevated stress — caution advised.")
        if self.trust < 0.6:
            insights.append("Trust integrity at risk — agent divergence possible.")
        insights.append(f"Current risk appetite: {self.risk_appetite:.2f}")
        if self.architect_injected:
            insights.append("Architect has injected override this tick.")
        return insights
