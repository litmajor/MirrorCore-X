
import asyncio
import logging
from typing import Optional
import ccxt

logger = logging.getLogger(__name__)

class EmergencyKillSwitch:
    """Global emergency stop for all trading operations"""
    
    def __init__(self, exchange: ccxt.Exchange, max_drawdown_pct: float = 15.0):
        self.exchange = exchange
        self.max_drawdown_pct = max_drawdown_pct
        self.is_emergency = False
        self.initial_balance = None
        
    async def check_and_trigger(self, current_pnl: float, current_latency_ms: float):
        """Check conditions and trigger emergency stop if needed"""
        
        # Drawdown check
        if self.initial_balance is None:
            self.initial_balance = 10000.0  # Set from actual balance
            
        drawdown_pct = (current_pnl / self.initial_balance) * 100
        
        if drawdown_pct <= -self.max_drawdown_pct:
            await self.emergency_stop(f"Max drawdown reached: {drawdown_pct:.2f}%")
            
        # Latency check
        if current_latency_ms > 500:
            await self.emergency_stop(f"High latency: {current_latency_ms}ms")
    
    async def emergency_stop(self, reason: str):
        """Immediately halt all trading and close positions"""
        if self.is_emergency:
            return
            
        self.is_emergency = True
        logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
        
        try:
            # Cancel all open orders
            await self.exchange.cancel_all_orders()
            logger.info("✅ All orders cancelled")
            
            # Close all positions (for futures)
            # Implement based on your exchange
            logger.warning("⚠️ Manual position closing required")
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            
    def reset(self):
        """Reset emergency state (manual intervention required)"""
        self.is_emergency = False
        logger.info("Emergency state reset")
