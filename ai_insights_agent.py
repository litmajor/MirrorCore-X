
"""
AI Insights Agent - Intelligent Trading Assistant
Provides natural language explanations of trading data, strategies, and system state
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class InsightContext:
    """Context for generating insights"""
    user_query: str
    system_state: Dict[str, Any]
    market_data: List[Dict]
    performance_metrics: Dict[str, float]
    active_signals: List[Dict]
    timestamp: float

class AIInsightsAgent:
    """AI-powered insights and chat assistant"""
    
    def __init__(self):
        self.conversation_history = []
        self.insight_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load response templates for common queries"""
        return {
            "performance": "Based on current metrics, your system shows a Sharpe ratio of {sharpe:.2f} with {win_rate:.1f}% win rate. {interpretation}",
            "strategy": "The {strategy_name} strategy is currently {status} with {confidence:.1f}% confidence. {explanation}",
            "market": "Market analysis shows {trend} trend with {volatility} volatility. {recommendation}",
            "risk": "Risk assessment: {risk_level}. Current exposure is {exposure:.1f}% with max drawdown at {drawdown:.1f}%. {action}",
            "signals": "There are {count} active signals. Top opportunity: {top_signal} with {score:.1f}% composite score. {details}"
        }
    
    async def process_query(self, query: str, context: InsightContext) -> Dict[str, Any]:
        """Process user query and generate intelligent response"""
        query_lower = query.lower()
        
        # Intent classification
        intent = self._classify_intent(query_lower)
        
        # Generate response based on intent
        if intent == "performance":
            response = await self._explain_performance(context)
        elif intent == "strategy":
            response = await self._explain_strategies(context)
        elif intent == "market":
            response = await self._explain_market(context)
        elif intent == "risk":
            response = await self._explain_risk(context)
        elif intent == "signals":
            response = await self._explain_signals(context)
        elif intent == "oracle":
            response = await self._explain_oracle(context)
        else:
            response = await self._general_assistance(query, context)
        
        # Store in conversation history
        self.conversation_history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _classify_intent(self, query: str) -> str:
        """Classify user query intent"""
        intent_keywords = {
            "performance": ["performance", "pnl", "profit", "sharpe", "returns", "win rate"],
            "strategy": ["strategy", "strategies", "bayesian", "ensemble", "trading"],
            "market": ["market", "trend", "volatility", "price", "btc", "eth"],
            "risk": ["risk", "drawdown", "exposure", "position", "stop loss"],
            "signals": ["signal", "opportunity", "buy", "sell", "alert"],
            "oracle": ["oracle", "directive", "imagination", "prediction"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(kw in query for kw in keywords):
                return intent
        
        return "general"
    
    async def _explain_performance(self, context: InsightContext) -> Dict[str, Any]:
        """Explain current performance metrics"""
        metrics = context.performance_metrics
        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)
        pnl = metrics.get('total_pnl', 0)
        
        # Generate interpretation
        if sharpe > 2.0:
            interpretation = "Excellent risk-adjusted returns! The system is performing exceptionally well."
        elif sharpe > 1.0:
            interpretation = "Strong performance with good risk management."
        else:
            interpretation = "Performance is moderate. Consider reviewing strategy weights."
        
        suggestion = ""
        if win_rate < 50:
            suggestion = " Tip: Your win rate could be improved by tightening entry criteria on lower-confidence signals."
        
        return {
            "type": "performance_analysis",
            "summary": f"System P&L: ${pnl:.2f} | Sharpe: {sharpe:.2f} | Win Rate: {win_rate:.1f}%",
            "interpretation": interpretation + suggestion,
            "metrics": {
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "total_pnl": pnl,
                "max_drawdown": metrics.get('max_drawdown', 0)
            },
            "recommendations": self._generate_performance_recommendations(metrics)
        }
    
    async def _explain_strategies(self, context: InsightContext) -> Dict[str, Any]:
        """Explain active strategies and their performance"""
        state = context.system_state
        strategy_grades = state.get('strategy_grades', {})
        
        top_strategies = sorted(
            strategy_grades.items(),
            key=lambda x: ord(x[1]) if isinstance(x[1], str) else 0,
            reverse=True
        )[:5]
        
        analysis = []
        for strategy, grade in top_strategies:
            status = "performing excellently" if grade == 'A' else "performing well" if grade == 'B' else "underperforming"
            analysis.append(f"• {strategy}: Grade {grade} - {status}")
        
        return {
            "type": "strategy_analysis",
            "summary": f"Tracking {len(strategy_grades)} strategies. Top 5 performers:",
            "strategy_breakdown": "\n".join(analysis),
            "top_strategy": top_strategies[0][0] if top_strategies else "None",
            "recommendations": [
                "Focus capital on A-graded strategies",
                "Review C-graded strategies for optimization",
                "Monitor B-graded strategies for trend changes"
            ]
        }
    
    async def _explain_market(self, context: InsightContext) -> Dict[str, Any]:
        """Explain current market conditions"""
        market_data = context.market_data
        if not market_data:
            return {"type": "market_analysis", "summary": "No market data available"}
        
        # Calculate market metrics
        recent_data = market_data[-20:] if len(market_data) > 20 else market_data
        
        avg_volatility = np.mean([d.get('volatility', 0) for d in recent_data])
        avg_momentum = np.mean([d.get('momentum_7d', 0) for d in recent_data])
        
        # Determine trend
        if avg_momentum > 0.03:
            trend = "strong bullish"
            recommendation = "Favor long positions and momentum strategies"
        elif avg_momentum > 0:
            trend = "moderate bullish"
            recommendation = "Balanced approach with slight long bias"
        elif avg_momentum > -0.03:
            trend = "moderate bearish"
            recommendation = "Reduce long exposure, consider mean reversion plays"
        else:
            trend = "strong bearish"
            recommendation = "Focus on short strategies and risk management"
        
        # Volatility assessment
        if avg_volatility > 0.05:
            vol_desc = "high"
            vol_rec = "Widen stop losses and reduce position sizes"
        elif avg_volatility > 0.02:
            vol_desc = "moderate"
            vol_rec = "Normal position sizing recommended"
        else:
            vol_desc = "low"
            vol_rec = "Tighter stops acceptable, consider larger positions"
        
        return {
            "type": "market_analysis",
            "summary": f"Market showing {trend} trend with {vol_desc} volatility",
            "trend_direction": trend,
            "volatility_level": vol_desc,
            "recommendation": f"{recommendation}. {vol_rec}",
            "key_metrics": {
                "avg_momentum": avg_momentum,
                "avg_volatility": avg_volatility
            }
        }
    
    async def _explain_risk(self, context: InsightContext) -> Dict[str, Any]:
        """Explain current risk profile"""
        metrics = context.performance_metrics
        drawdown = metrics.get('max_drawdown', 0)
        
        if drawdown > 25:
            risk_level = "HIGH"
            action = "Immediate action required: Reduce positions and review risk parameters"
        elif drawdown > 15:
            risk_level = "ELEVATED"
            action = "Monitor closely and consider reducing exposure"
        else:
            risk_level = "NORMAL"
            action = "Risk profile is healthy, continue normal operations"
        
        return {
            "type": "risk_analysis",
            "risk_level": risk_level,
            "max_drawdown": drawdown,
            "action": action,
            "risk_metrics": {
                "var_95": metrics.get('var_95', 0),
                "portfolio_volatility": metrics.get('portfolio_volatility', 0)
            }
        }
    
    async def _explain_signals(self, context: InsightContext) -> Dict[str, Any]:
        """Explain active trading signals"""
        signals = context.active_signals
        
        if not signals:
            return {
                "type": "signal_analysis",
                "summary": "No active signals currently",
                "recommendation": "System is monitoring markets for opportunities"
            }
        
        # Find top signal
        top_signal = max(signals, key=lambda x: x.get('composite_score', 0))
        
        signal_breakdown = []
        for sig in signals[:5]:
            signal_breakdown.append(
                f"• {sig.get('symbol', 'Unknown')}: {sig.get('signal', 'N/A')} "
                f"({sig.get('composite_score', 0):.1f}% confidence)"
            )
        
        return {
            "type": "signal_analysis",
            "summary": f"{len(signals)} active signals detected",
            "top_opportunity": {
                "symbol": top_signal.get('symbol'),
                "action": top_signal.get('signal'),
                "score": top_signal.get('composite_score'),
                "reason": self._explain_signal_reason(top_signal)
            },
            "signal_breakdown": "\n".join(signal_breakdown)
        }
    
    async def _explain_oracle(self, context: InsightContext) -> Dict[str, Any]:
        """Explain Oracle directives and imagination insights"""
        state = context.system_state
        directives = state.get('oracle_directives', [])
        
        if not directives:
            return {
                "type": "oracle_analysis",
                "summary": "Oracle is analyzing market conditions",
                "status": "No directives issued yet"
            }
        
        latest = directives[-1] if directives else {}
        
        return {
            "type": "oracle_analysis",
            "summary": "Oracle has issued trading directives based on multi-agent analysis",
            "latest_directive": {
                "symbol": latest.get('symbol'),
                "action": latest.get('action'),
                "confidence": latest.get('confidence', 0) * 100,
                "reasoning": "Based on ensemble strategy consensus and market regime analysis"
            },
            "imagination_insights": "Counterfactual scenarios analyzed for robustness"
        }
    
    async def _general_assistance(self, query: str, context: InsightContext) -> Dict[str, Any]:
        """Provide general assistance"""
        return {
            "type": "general_assistance",
            "summary": "I can help you understand:",
            "topics": [
                "📊 System Performance - P&L, Sharpe ratio, win rates",
                "🎯 Trading Strategies - Active strategies and their grades", 
                "📈 Market Analysis - Trends, volatility, opportunities",
                "⚠️ Risk Management - Drawdowns, exposure, safety metrics",
                "🔔 Trading Signals - Active alerts and recommendations",
                "🔮 Oracle Insights - AI-driven directives and predictions"
            ],
            "suggestion": "Try asking: 'How is my performance?' or 'What are the top signals?'"
        }
    
    def _explain_signal_reason(self, signal: Dict) -> str:
        """Generate explanation for why a signal was generated"""
        reasons = []
        
        if signal.get('momentum_7d', 0) > 0.05:
            reasons.append("strong momentum")
        if signal.get('volume_ratio', 1) > 1.5:
            reasons.append("high volume")
        if signal.get('rsi', 50) < 30:
            reasons.append("oversold RSI")
        elif signal.get('rsi', 50) > 70:
            reasons.append("overbought RSI")
        
        if reasons:
            return "Triggered by: " + ", ".join(reasons)
        return "Multiple technical indicators aligned"
    
    def _generate_performance_recommendations(self, metrics: Dict) -> List[str]:
        """Generate performance improvement recommendations"""
        recs = []
        
        if metrics.get('win_rate', 0) < 55:
            recs.append("Improve entry criteria - current win rate below optimal")
        
        if metrics.get('max_drawdown', 0) > 20:
            recs.append("Tighten risk management - drawdown exceeds comfort zone")
        
        if metrics.get('sharpe_ratio', 0) < 1.5:
            recs.append("Review strategy allocation - Sharpe ratio can be improved")
        
        if not recs:
            recs.append("Performance is strong - maintain current approach")
        
        return recs

# Singleton instance
ai_insights = AIInsightsAgent()
