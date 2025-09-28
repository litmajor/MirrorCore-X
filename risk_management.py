
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    DRAWDOWN = "drawdown"
    POSITION_SIZE = "position_size"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    EXPOSURE = "exposure"
    LIQUIDITY = "liquidity"

@dataclass
class RiskAlert:
    timestamp: float
    alert_type: AlertType
    level: RiskLevel
    message: str
    value: float
    threshold: float
    symbol: Optional[str] = None
    strategy: Optional[str] = None

@dataclass
class RiskLimits:
    max_drawdown: float = 0.15  # 15% max drawdown
    max_position_size: float = 0.1  # 10% of portfolio per position
    max_portfolio_volatility: float = 0.25  # 25% annualized volatility
    max_correlation: float = 0.8  # 80% max correlation between positions
    max_sector_exposure: float = 0.4  # 40% max exposure to any sector
    min_liquidity_ratio: float = 0.2  # 20% minimum cash ratio
    max_leverage: float = 2.0  # 2x maximum leverage
    var_confidence: float = 0.95  # 95% VaR confidence level
    stress_test_scenarios: int = 1000  # Number of Monte Carlo scenarios

class AdvancedRiskManager:
    """
    Comprehensive risk management system with multiple safety layers
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.alerts: List[RiskAlert] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.position_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.var_estimate: float = 0.0
        self.expected_shortfall: float = 0.0
        self.circuit_breakers_active: bool = False
        
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main risk management update function"""
        try:
            # Extract relevant data
            portfolio_value = data.get('portfolio_value', 100000.0)
            positions = data.get('positions', {})
            market_data = data.get('market_data', [])
            trades = data.get('trades', [])
            
            # Update portfolio and position history
            self._update_history(portfolio_value, positions, trades)
            
            # Perform risk checks
            risk_alerts = []
            
            # 1. Drawdown monitoring
            drawdown_alert = self._check_drawdown(portfolio_value)
            if drawdown_alert:
                risk_alerts.append(drawdown_alert)
            
            # 2. Position size limits
            position_alerts = self._check_position_sizes(positions, portfolio_value)
            risk_alerts.extend(position_alerts)
            
            # 3. Portfolio volatility
            volatility_alert = self._check_portfolio_volatility()
            if volatility_alert:
                risk_alerts.append(volatility_alert)
            
            # 4. Correlation monitoring
            correlation_alerts = self._check_correlations(positions, market_data)
            risk_alerts.extend(correlation_alerts)
            
            # 5. Liquidity monitoring
            liquidity_alert = self._check_liquidity(positions, portfolio_value)
            if liquidity_alert:
                risk_alerts.append(liquidity_alert)
            
            # 6. VaR and Expected Shortfall
            self._calculate_var_es()
            
            # 7. Stress testing
            stress_results = self._run_stress_tests(positions, market_data)
            
            # Update alerts
            self.alerts.extend(risk_alerts)
            self._cleanup_old_alerts()
            
            # Determine if circuit breakers should be active
            self._update_circuit_breakers(risk_alerts)
            
            # Calculate performance metrics
            self._update_performance_metrics()
            
            return {
                'risk_alerts': [alert.__dict__ for alert in risk_alerts],
                'risk_metrics': {
                    'current_drawdown': self._calculate_current_drawdown(),
                    'portfolio_volatility': self._calculate_portfolio_volatility(),
                    'var_95': self.var_estimate,
                    'expected_shortfall': self.expected_shortfall,
                    'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0.0),
                    'max_drawdown': self.performance_metrics.get('max_drawdown', 0.0),
                    'circuit_breakers_active': self.circuit_breakers_active
                },
                'stress_test_results': stress_results,
                'position_limits': self._calculate_position_limits(portfolio_value),
                'risk_score': self._calculate_overall_risk_score()
            }
            
        except Exception as e:
            logger.error(f"Risk management update failed: {e}")
            return {'error': str(e)}
    
    def _update_history(self, portfolio_value: float, positions: Dict, trades: List):
        """Update historical data for risk calculations"""
        timestamp = datetime.now().timestamp()
        
        # Portfolio history
        self.portfolio_history.append({
            'timestamp': timestamp,
            'value': portfolio_value,
            'positions_count': len(positions),
            'total_exposure': sum(abs(pos.get('value', 0)) for pos in positions.values())
        })
        
        # Keep only last 1000 entries
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
        
        # Position history
        for symbol, position in positions.items():
            self.position_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'size': position.get('size', 0),
                'value': position.get('value', 0),
                'pnl': position.get('pnl', 0)
            })
        
        # Keep only last 5000 position entries
        if len(self.position_history) > 5000:
            self.position_history = self.position_history[-5000:]
    
    def _check_drawdown(self, current_value: float) -> Optional[RiskAlert]:
        """Check for excessive drawdown"""
        if len(self.portfolio_history) < 2:
            return None
        
        # Calculate peak and current drawdown
        peak_value = max(entry['value'] for entry in self.portfolio_history)
        current_drawdown = (peak_value - current_value) / peak_value
        
        if current_drawdown > self.risk_limits.max_drawdown:
            return RiskAlert(
                timestamp=datetime.now().timestamp(),
                alert_type=AlertType.DRAWDOWN,
                level=RiskLevel.CRITICAL if current_drawdown > self.risk_limits.max_drawdown * 1.5 else RiskLevel.HIGH,
                message=f"Portfolio drawdown of {current_drawdown:.2%} exceeds limit of {self.risk_limits.max_drawdown:.2%}",
                value=current_drawdown,
                threshold=self.risk_limits.max_drawdown
            )
        
        return None
    
    def _check_position_sizes(self, positions: Dict, portfolio_value: float) -> List[RiskAlert]:
        """Check individual position sizes"""
        alerts = []
        
        for symbol, position in positions.items():
            position_value = abs(position.get('value', 0))
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_pct > self.risk_limits.max_position_size:
                level = RiskLevel.CRITICAL if position_pct > self.risk_limits.max_position_size * 2 else RiskLevel.HIGH
                alerts.append(RiskAlert(
                    timestamp=datetime.now().timestamp(),
                    alert_type=AlertType.POSITION_SIZE,
                    level=level,
                    message=f"Position {symbol} size of {position_pct:.2%} exceeds limit of {self.risk_limits.max_position_size:.2%}",
                    value=position_pct,
                    threshold=self.risk_limits.max_position_size,
                    symbol=symbol
                ))
        
        return alerts
    
    def _check_portfolio_volatility(self) -> Optional[RiskAlert]:
        """Check portfolio volatility"""
        if len(self.portfolio_history) < 30:  # Need at least 30 data points
            return None
        
        # Calculate returns
        values = [entry['value'] for entry in self.portfolio_history[-30:]]
        returns = np.diff(values) / values[:-1]
        
        # Annualized volatility (assuming daily data)
        volatility = np.std(returns) * np.sqrt(252)
        
        if volatility > self.risk_limits.max_portfolio_volatility:
            return RiskAlert(
                timestamp=datetime.now().timestamp(),
                alert_type=AlertType.VOLATILITY,
                level=RiskLevel.HIGH if volatility < self.risk_limits.max_portfolio_volatility * 1.5 else RiskLevel.CRITICAL,
                message=f"Portfolio volatility of {volatility:.2%} exceeds limit of {self.risk_limits.max_portfolio_volatility:.2%}",
                value=volatility,
                threshold=self.risk_limits.max_portfolio_volatility
            )
        
        return None
    
    def _check_correlations(self, positions: Dict, market_data: List) -> List[RiskAlert]:
        """Check correlations between positions"""
        alerts = []
        
        if len(positions) < 2:
            return alerts
        
        # This is a simplified correlation check
        # In practice, you'd calculate actual price correlations
        symbols = list(positions.keys())
        
        # Placeholder correlation matrix (would be calculated from actual price data)
        n_symbols = len(symbols)
        correlation_matrix = np.random.rand(n_symbols, n_symbols)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Check for high correlations
        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                correlation = correlation_matrix[i, j]
                if correlation > self.risk_limits.max_correlation:
                    alerts.append(RiskAlert(
                        timestamp=datetime.now().timestamp(),
                        alert_type=AlertType.CORRELATION,
                        level=RiskLevel.MEDIUM,
                        message=f"High correlation {correlation:.2f} between {symbols[i]} and {symbols[j]}",
                        value=correlation,
                        threshold=self.risk_limits.max_correlation
                    ))
        
        return alerts
    
    def _check_liquidity(self, positions: Dict, portfolio_value: float) -> Optional[RiskAlert]:
        """Check portfolio liquidity"""
        # Calculate total position value
        total_position_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
        cash_ratio = 1.0 - (total_position_value / portfolio_value) if portfolio_value > 0 else 1.0
        
        if cash_ratio < self.risk_limits.min_liquidity_ratio:
            return RiskAlert(
                timestamp=datetime.now().timestamp(),
                alert_type=AlertType.LIQUIDITY,
                level=RiskLevel.MEDIUM,
                message=f"Cash ratio of {cash_ratio:.2%} below minimum of {self.risk_limits.min_liquidity_ratio:.2%}",
                value=cash_ratio,
                threshold=self.risk_limits.min_liquidity_ratio
            )
        
        return None
    
    def _calculate_var_es(self):
        """Calculate Value at Risk and Expected Shortfall"""
        if len(self.portfolio_history) < 30:
            return
        
        # Calculate returns
        values = [entry['value'] for entry in self.portfolio_history[-252:]]  # Last year of data
        returns = np.diff(values) / values[:-1]
        
        # VaR calculation
        var_percentile = (1 - self.risk_limits.var_confidence) * 100
        self.var_estimate = np.percentile(returns, var_percentile)
        
        # Expected Shortfall (average of returns below VaR)
        tail_returns = returns[returns <= self.var_estimate]
        self.expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else self.var_estimate
    
    def _run_stress_tests(self, positions: Dict, market_data: List) -> Dict[str, Any]:
        """Run Monte Carlo stress tests"""
        if not positions:
            return {'stress_scenarios': 0, 'worst_case_loss': 0.0, 'probability_of_loss': 0.0}
        
        # Simplified stress testing
        scenarios = self.risk_limits.stress_test_scenarios
        losses = []
        
        for _ in range(scenarios):
            # Generate random market shock
            market_shock = np.random.normal(-0.02, 0.05)  # Mean -2%, std 5%
            
            # Calculate portfolio loss under this scenario
            portfolio_loss = 0.0
            for symbol, position in positions.items():
                position_value = position.get('value', 0)
                # Assume all positions move with market (simplified)
                position_loss = position_value * market_shock
                portfolio_loss += position_loss
            
            losses.append(portfolio_loss)
        
        worst_case_loss = min(losses)
        probability_of_loss = len([l for l in losses if l < 0]) / len(losses)
        
        return {
            'stress_scenarios': scenarios,
            'worst_case_loss': worst_case_loss,
            'probability_of_loss': probability_of_loss,
            'average_loss': np.mean([l for l in losses if l < 0]) if any(l < 0 for l in losses) else 0.0
        }
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        peak_value = max(entry['value'] for entry in self.portfolio_history)
        current_value = self.portfolio_history[-1]['value']
        return (peak_value - current_value) / peak_value
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.portfolio_history) < 30:
            return 0.0
        
        values = [entry['value'] for entry in self.portfolio_history[-30:]]
        returns = np.diff(values) / values[:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _update_circuit_breakers(self, risk_alerts: List[RiskAlert]):
        """Update circuit breaker status based on risk alerts"""
        critical_alerts = [alert for alert in risk_alerts if alert.level == RiskLevel.CRITICAL]
        high_alerts = [alert for alert in risk_alerts if alert.level == RiskLevel.HIGH]
        
        # Activate circuit breakers if we have critical alerts or too many high alerts
        self.circuit_breakers_active = len(critical_alerts) > 0 or len(high_alerts) >= 3
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if len(self.portfolio_history) < 30:
            return
        
        values = [entry['value'] for entry in self.portfolio_history]
        returns = np.diff(values) / values[:-1]
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        excess_returns = returns - 0.02/252  # Daily risk-free rate
        self.performance_metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / peak
        self.performance_metrics['max_drawdown'] = np.max(drawdowns)
        
        # Win rate (simplified)
        winning_days = len([r for r in returns if r > 0])
        self.performance_metrics['win_rate'] = winning_days / len(returns) if returns else 0
    
    def _calculate_position_limits(self, portfolio_value: float) -> Dict[str, float]:
        """Calculate dynamic position limits"""
        current_volatility = self._calculate_portfolio_volatility()
        volatility_adjustment = max(0.5, min(1.5, 1.0 - (current_volatility - 0.15) * 2))
        
        return {
            'max_position_size': self.risk_limits.max_position_size * volatility_adjustment,
            'max_sector_exposure': self.risk_limits.max_sector_exposure,
            'max_correlation': self.risk_limits.max_correlation,
            'recommended_cash_ratio': max(self.risk_limits.min_liquidity_ratio, current_volatility * 0.5)
        }
    
    def _calculate_overall_risk_score(self) -> float:
        """Calculate overall risk score (0-100)"""
        risk_factors = []
        
        # Drawdown component
        current_drawdown = self._calculate_current_drawdown()
        drawdown_score = min(100, (current_drawdown / self.risk_limits.max_drawdown) * 50)
        risk_factors.append(drawdown_score)
        
        # Volatility component
        current_volatility = self._calculate_portfolio_volatility()
        volatility_score = min(100, (current_volatility / self.risk_limits.max_portfolio_volatility) * 30)
        risk_factors.append(volatility_score)
        
        # Alert-based component
        recent_alerts = [alert for alert in self.alerts if alert.timestamp > datetime.now().timestamp() - 3600]  # Last hour
        alert_score = min(100, len(recent_alerts) * 10)
        risk_factors.append(alert_score)
        
        return min(100, sum(risk_factors))
    
    def _cleanup_old_alerts(self):
        """Remove alerts older than 24 hours"""
        cutoff_time = datetime.now().timestamp() - 86400  # 24 hours
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be halted"""
        if self.circuit_breakers_active:
            return True, "Circuit breakers active due to critical risk alerts"
        
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.risk_limits.max_drawdown * 1.5:
            return True, f"Excessive drawdown: {current_drawdown:.2%}"
        
        risk_score = self._calculate_overall_risk_score()
        if risk_score > 90:
            return True, f"Risk score too high: {risk_score:.1f}/100"
        
        return False, ""
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        halt_trading, halt_reason = self.should_halt_trading()
        
        return {
            'risk_score': self._calculate_overall_risk_score(),
            'current_drawdown': self._calculate_current_drawdown(),
            'portfolio_volatility': self._calculate_portfolio_volatility(),
            'var_95': self.var_estimate,
            'expected_shortfall': self.expected_shortfall,
            'circuit_breakers_active': self.circuit_breakers_active,
            'halt_trading': halt_trading,
            'halt_reason': halt_reason,
            'active_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a.level == RiskLevel.CRITICAL]),
            'performance_metrics': self.performance_metrics,
            'last_updated': datetime.now().isoformat()
        }

# Example usage and integration
class RiskManagedTradingSystem:
    """Trading system with integrated risk management"""
    
    def __init__(self, sync_bus, initial_capital: float = 100000.0):
        self.sync_bus = sync_bus
        self.risk_manager = AdvancedRiskManager()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update with risk management checks"""
        # Get current portfolio state
        portfolio_data = {
            'portfolio_value': self.current_capital,
            'positions': data.get('positions', {}),
            'market_data': data.get('market_data', []),
            'trades': data.get('trades', [])
        }
        
        # Run risk management
        risk_results = await self.risk_manager.update(portfolio_data)
        
        # Check if trading should be halted
        halt_trading, halt_reason = self.risk_manager.should_halt_trading()
        
        if halt_trading:
            logger.warning(f"Trading halted: {halt_reason}")
            await self.sync_bus.update_state('trading_halted', True)
            await self.sync_bus.update_state('halt_reason', halt_reason)
        
        # Update sync bus with risk data
        await self.sync_bus.update_state('risk_alerts', risk_results.get('risk_alerts', []))
        await self.sync_bus.update_state('risk_metrics', risk_results.get('risk_metrics', {}))
        await self.sync_bus.update_state('risk_score', self.risk_manager._calculate_overall_risk_score())
        
        return risk_results
