
"""
Emergency Controls and Kill Switch System for MirrorCore-X
Implements critical safety mechanisms to prevent catastrophic losses
"""

import os
import time
import logging
import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import aiofiles
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EmergencyReason(Enum):
    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    LATENCY_EXCEEDED = "latency_exceeded"
    API_ERRORS = "api_errors"
    MANUAL_TRIGGER = "manual_trigger"
    POSITION_LIMIT = "position_limit"
    CORRELATION_RISK = "correlation_risk"
    DATA_DIVERGENCE = "data_divergence"

@dataclass
class EmergencyConfig:
    max_drawdown_pct: float = 15.0  # 15% max drawdown
    max_latency_ms: float = 500.0   # 500ms max latency
    max_api_errors: int = 5         # 5 consecutive API errors
    position_limit_usd: float = 50000.0  # $50k max position
    correlation_threshold: float = 0.7    # 70% correlation limit
    price_divergence_pct: float = 0.2     # 0.2% price divergence limit

class EmergencyController:
    """Global emergency controller with kill switch capability"""
    
    def __init__(self, config: EmergencyConfig, exchange, syncbus):
        self.config = config
        self.exchange = exchange
        self.syncbus = syncbus
        
        # Emergency state
        self.emergency_active = False
        self.emergency_reason = None
        self.emergency_timestamp = None
        
        # Monitoring state
        self.initial_balance = 100000.0  # Set from config
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.api_error_count = 0
        self.last_api_success = time.time()
        self.latency_samples = []
        
        # Position tracking
        self.open_positions = {}
        self.total_exposure = 0.0
        
        # Data integrity
        self.price_sources = {}
        self.last_price_check = {}
        
        # Audit log
        self.audit_log = []
        
        logger.info("Emergency Controller initialized with safety limits")
    
    async def check_emergency_conditions(self) -> bool:
        """Check all emergency conditions and trigger if necessary"""
        
        # Check drawdown
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > (self.config.max_drawdown_pct / 100):
            await self.trigger_emergency(EmergencyReason.DRAWDOWN_EXCEEDED, 
                                        f"Drawdown {current_drawdown:.2%} exceeds limit {self.config.max_drawdown_pct}%")
            return True
        
        # Check API latency
        if len(self.latency_samples) > 10:
            avg_latency = sum(self.latency_samples[-10:]) / 10
            if avg_latency > self.config.max_latency_ms:
                await self.trigger_emergency(EmergencyReason.LATENCY_EXCEEDED,
                                            f"Average latency {avg_latency:.1f}ms exceeds limit {self.config.max_latency_ms}ms")
                return True
        
        # Check API errors
        if self.api_error_count >= self.config.max_api_errors:
            await self.trigger_emergency(EmergencyReason.API_ERRORS,
                                        f"API error count {self.api_error_count} exceeds limit {self.config.max_api_errors}")
            return True
        
        # Check position limits
        if self.total_exposure > self.config.position_limit_usd:
            await self.trigger_emergency(EmergencyReason.POSITION_LIMIT,
                                        f"Total exposure ${self.total_exposure:,.2f} exceeds limit ${self.config.position_limit_usd:,.2f}")
            return True
        
        return False
    
    async def trigger_emergency(self, reason: EmergencyReason, message: str):
        """Trigger emergency stop - flatten all positions immediately"""
        if self.emergency_active:
            return  # Already in emergency mode
        
        self.emergency_active = True
        self.emergency_reason = reason
        self.emergency_timestamp = time.time()
        
        logger.critical(f"EMERGENCY TRIGGERED: {reason.value} - {message}")
        
        # Log emergency event
        await self.log_emergency_event(reason, message)
        
        # Flatten all positions immediately
        await self.flatten_all_positions()
        
        # Stop all trading activity via SyncBus
        await self.syncbus.broadcast_command('emergency_stop')
        
        # Send emergency notification
        await self.send_emergency_notification(reason, message)
    
    async def flatten_all_positions(self):
        """Emergency position flattening - close all positions immediately"""
        logger.warning("Flattening all positions - EMERGENCY STOP")
        
        flattened_positions = []
        
        try:
            # Get current positions from exchange
            positions = await self.exchange.fetch_positions()
            
            for position in positions:
                if position['contracts'] != 0:  # Has open position
                    symbol = position['symbol']
                    size = abs(position['contracts'])
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    
                    try:
                        # Close position with market order for immediate execution
                        order = await self.exchange.create_market_order(
                            symbol=symbol,
                            type='market',
                            side=side,
                            amount=size
                        )
                        
                        flattened_positions.append({
                            'symbol': symbol,
                            'size': size,
                            'side': side,
                            'order_id': order['id'],
                            'timestamp': time.time()
                        })
                        
                        logger.warning(f"Emergency close: {symbol} {side} {size}")
                        
                    except Exception as e:
                        logger.error(f"Failed to close position {symbol}: {e}")
            
            # Log all flattened positions
            await self.log_flattened_positions(flattened_positions)
            
        except Exception as e:
            logger.error(f"Error during emergency flatten: {e}")
    
    async def update_balance(self, new_balance: float):
        """Update balance and check for new peak"""
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
    
    async def record_api_latency(self, latency_ms: float):
        """Record API latency sample"""
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]  # Keep last 100 samples
        
        # Reset error count on successful API call
        self.api_error_count = 0
        self.last_api_success = time.time()
    
    async def record_api_error(self):
        """Record API error"""
        self.api_error_count += 1
        logger.warning(f"API error recorded. Count: {self.api_error_count}")
    
    async def update_positions(self, positions: Dict[str, Any]):
        """Update position tracking"""
        self.open_positions = positions
        self.total_exposure = sum(abs(pos.get('notional', 0)) for pos in positions.values())
    
    async def check_price_divergence(self, symbol: str, price1: float, price2: float) -> bool:
        """Check for price divergence between sources"""
        if price1 == 0 or price2 == 0:
            return False
        
        divergence = abs(price1 - price2) / ((price1 + price2) / 2)
        
        if divergence > (self.config.price_divergence_pct / 100):
            await self.trigger_emergency(EmergencyReason.DATA_DIVERGENCE,
                                        f"Price divergence {divergence:.2%} for {symbol}: {price1} vs {price2}")
            return True
        
        return False
    
    async def log_emergency_event(self, reason: EmergencyReason, message: str):
        """Log emergency event to audit trail"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'EMERGENCY_TRIGGER',
            'reason': reason.value,
            'message': message,
            'balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'drawdown_pct': (self.peak_balance - self.current_balance) / self.peak_balance * 100,
            'total_exposure': self.total_exposure,
            'api_error_count': self.api_error_count
        }
        
        self.audit_log.append(event)
        
        # Write to persistent log file
        async with aiofiles.open('emergency_audit.log', 'a') as f:
            await f.write(json.dumps(event) + '\n')
    
    async def log_flattened_positions(self, positions: list):
        """Log all flattened positions"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'POSITIONS_FLATTENED',
            'positions': positions,
            'total_positions': len(positions)
        }
        
        async with aiofiles.open('emergency_audit.log', 'a') as f:
            await f.write(json.dumps(event) + '\n')
    
    async def send_emergency_notification(self, reason: EmergencyReason, message: str):
        """Send emergency notification (implement based on your notification system)"""
        notification = {
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL',
            'system': 'MirrorCore-X',
            'reason': reason.value,
            'message': message,
            'balance': self.current_balance,
            'drawdown': (self.peak_balance - self.current_balance) / self.peak_balance * 100
        }
        
        # Log notification (extend this to send actual notifications)
        logger.critical(f"EMERGENCY NOTIFICATION: {json.dumps(notification, indent=2)}")
    
    def is_emergency_active(self) -> bool:
        """Check if emergency mode is active"""
        return self.emergency_active
    
    async def reset_emergency(self):
        """Reset emergency state (use with caution)"""
        logger.warning("Resetting emergency state - manual override")
        self.emergency_active = False
        self.emergency_reason = None
        self.emergency_timestamp = None
        self.api_error_count = 0

class WatchdogProcess:
    """Independent watchdog to monitor main process health"""
    
    def __init__(self, heartbeat_file: str = 'heartbeat.txt', max_stale_seconds: int = 30):
        self.heartbeat_file = heartbeat_file
        self.max_stale_seconds = max_stale_seconds
        self.running = True
    
    async def start_monitoring(self):
        """Start watchdog monitoring loop"""
        logger.info("Watchdog process started")
        
        while self.running:
            try:
                # Check heartbeat file
                if os.path.exists(self.heartbeat_file):
                    stat = os.stat(self.heartbeat_file)
                    last_heartbeat = stat.st_mtime
                    time_since_heartbeat = time.time() - last_heartbeat
                    
                    if time_since_heartbeat > self.max_stale_seconds:
                        logger.critical(f"Heartbeat stale for {time_since_heartbeat:.1f}s - triggering emergency")
                        await self.trigger_watchdog_emergency()
                        break
                else:
                    logger.warning("Heartbeat file not found - main process may be down")
                    await asyncio.sleep(5)
                    continue
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(5)
    
    async def trigger_watchdog_emergency(self):
        """Trigger emergency from watchdog"""
        logger.critical("WATCHDOG EMERGENCY: Main process appears stalled")
        
        # Log watchdog emergency
        emergency_log = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'WATCHDOG_EMERGENCY',
            'message': 'Main process heartbeat stale',
            'stale_duration': self.max_stale_seconds
        }
        
        async with aiofiles.open('watchdog_emergency.log', 'a') as f:
            await f.write(json.dumps(emergency_log) + '\n')
        
        # Additional emergency actions can be added here
        # For example: send notifications, attempt graceful shutdown, etc.
    
    def stop(self):
        """Stop watchdog monitoring"""
        self.running = False

# Heartbeat utility for main process
class HeartbeatManager:
    """Manages heartbeat for watchdog monitoring"""
    
    def __init__(self, heartbeat_file: str = 'heartbeat.txt'):
        self.heartbeat_file = heartbeat_file
        self.running = True
    
    async def start_heartbeat(self):
        """Start heartbeat loop"""
        while self.running:
            try:
                # Update heartbeat file
                with open(self.heartbeat_file, 'w') as f:
                    f.write(str(time.time()))
                
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)
    
    def stop(self):
        """Stop heartbeat"""
        self.running = False
        if os.path.exists(self.heartbeat_file):
            os.remove(self.heartbeat_file)

# Integration functions
async def create_emergency_system(exchange, syncbus, config: Optional[EmergencyConfig] = None):
    """Create and initialize emergency control system"""
    if config is None:
        config = EmergencyConfig()
    
    emergency_controller = EmergencyController(config, exchange, syncbus)
    heartbeat_manager = HeartbeatManager()
    
    # Start heartbeat
    asyncio.create_task(heartbeat_manager.start_heartbeat())
    
    return emergency_controller, heartbeat_manager

# Example usage
if __name__ == "__main__":
    async def test_emergency_system():
        # Mock objects for testing
        class MockExchange:
            async def fetch_positions(self): return []
            async def create_market_order(self, **kwargs): return {'id': 'test_order'}
        
        class MockSyncBus:
            async def broadcast_command(self, cmd): print(f"Broadcast: {cmd}")
        
        config = EmergencyConfig(max_drawdown_pct=10.0)
        controller = EmergencyController(config, MockExchange(), MockSyncBus())
        
        # Test drawdown trigger
        await controller.update_balance(90000.0)  # 10% loss
        await controller.check_emergency_conditions()
        
        print(f"Emergency active: {controller.is_emergency_active()}")
    
    asyncio.run(test_emergency_system())
