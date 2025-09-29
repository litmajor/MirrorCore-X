
"""
Comprehensive Audit Logging System for MirrorCore-X
Provides structured, immutable audit trail for all trading activities
"""

import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiofiles
from collections import deque

logger = logging.getLogger(__name__)

class EventType(Enum):
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_LIMIT_HIT = "risk_limit_hit"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SIGNAL_GENERATED = "signal_generated"
    BALANCE_UPDATE = "balance_update"

@dataclass
class AuditEvent:
    """Structured audit event"""
    event_id: str
    timestamp: float
    event_type: EventType
    symbol: Optional[str]
    data: Dict[str, Any]
    checksum: Optional[str] = None
    previous_checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            'event_type': self.event_type.value,
            'symbol': self.symbol,
            'data': self.data,
            'checksum': self.checksum,
            'previous_checksum': self.previous_checksum
        }

class AuditLogger:
    """Immutable audit logger with cryptographic integrity"""
    
    def __init__(self, log_dir: str = "audit_logs", buffer_size: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.buffer_size = buffer_size
        self.event_buffer = deque(maxlen=buffer_size)
        self.last_checksum = None
        self.event_counter = 0
        
        # Create daily log file
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.log_file = self.log_dir / f"audit_{self.current_date}.log"
        
        # Initialize audit chain
        self._initialize_audit_chain()
        
        logger.info(f"Audit Logger initialized: {self.log_file}")
    
    def _initialize_audit_chain(self):
        """Initialize the audit chain with genesis event"""
        if not self.log_file.exists():
            genesis_event = AuditEvent(
                event_id=self._generate_event_id(),
                timestamp=time.time(),
                event_type=EventType.SYSTEM_START,
                symbol=None,
                data={'message': 'Audit chain initialized', 'version': '1.0'}
            )
            genesis_event.checksum = self._calculate_checksum(genesis_event)
            self.last_checksum = genesis_event.checksum
            
            # Write genesis event
            asyncio.create_task(self._write_event_to_file(genesis_event))
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        self.event_counter += 1
        timestamp_ms = int(time.time() * 1000000)  # Microsecond precision
        return f"EVT_{timestamp_ms}_{self.event_counter:06d}"
    
    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate SHA-256 checksum for event integrity"""
        # Create deterministic string representation
        event_str = json.dumps({
            'event_id': event.event_id,
            'timestamp': event.timestamp,
            'event_type': event.event_type.value,
            'symbol': event.symbol,
            'data': event.data,
            'previous_checksum': event.previous_checksum
        }, sort_keys=True, separators=(',', ':'))
        
        return hashlib.sha256(event_str.encode()).hexdigest()
    
    async def log_order_event(self, order_data: Dict[str, Any], event_type: EventType):
        """Log order-related events"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=time.time(),
            event_type=event_type,
            symbol=order_data.get('symbol'),
            data={
                'order_id': order_data.get('id'),
                'type': order_data.get('type'),
                'side': order_data.get('side'),
                'amount': order_data.get('amount'),
                'price': order_data.get('price'),
                'status': order_data.get('status'),
                'filled': order_data.get('filled', 0),
                'remaining': order_data.get('remaining', 0),
                'cost': order_data.get('cost', 0),
                'fee': order_data.get('fee'),
                'timestamp_exchange': order_data.get('timestamp'),
                'strategy': order_data.get('strategy', 'unknown'),
                'signal_strength': order_data.get('signal_strength'),
                'risk_metrics': order_data.get('risk_metrics')
            }
        )
        
        await self._log_event(event)
    
    async def log_position_event(self, position_data: Dict[str, Any], event_type: EventType):
        """Log position-related events"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=time.time(),
            event_type=event_type,
            symbol=position_data.get('symbol'),
            data={
                'position_size': position_data.get('size', 0),
                'entry_price': position_data.get('entry_price'),
                'exit_price': position_data.get('exit_price'),
                'pnl': position_data.get('pnl', 0),
                'pnl_percentage': position_data.get('pnl_percentage', 0),
                'duration_seconds': position_data.get('duration_seconds'),
                'strategy': position_data.get('strategy', 'unknown'),
                'stop_loss': position_data.get('stop_loss'),
                'take_profit': position_data.get('take_profit'),
                'risk_reward_ratio': position_data.get('risk_reward_ratio')
            }
        )
        
        await self._log_event(event)
    
    async def log_signal_event(self, signal_data: Dict[str, Any]):
        """Log trading signal generation"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=time.time(),
            event_type=EventType.SIGNAL_GENERATED,
            symbol=signal_data.get('symbol'),
            data={
                'signal_type': signal_data.get('signal'),
                'confidence': signal_data.get('confidence_score', 0),
                'composite_score': signal_data.get('composite_score', 0),
                'rsi': signal_data.get('rsi'),
                'macd': signal_data.get('macd'),
                'momentum_7d': signal_data.get('momentum_7d'),
                'volume_ratio': signal_data.get('volume_ratio'),
                'price': signal_data.get('price'),
                'timeframe': signal_data.get('timeframe', 'daily'),
                'scanner_version': signal_data.get('scanner_version', '1.0')
            }
        )
        
        await self._log_event(event)
    
    async def log_balance_update(self, balance_data: Dict[str, Any]):
        """Log balance/equity updates"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=time.time(),
            event_type=EventType.BALANCE_UPDATE,
            symbol=None,
            data={
                'total_balance': balance_data.get('total', 0),
                'free_balance': balance_data.get('free', 0),
                'used_balance': balance_data.get('used', 0),
                'equity': balance_data.get('equity', 0),
                'margin_level': balance_data.get('margin_level'),
                'unrealized_pnl': balance_data.get('unrealized_pnl', 0),
                'realized_pnl': balance_data.get('realized_pnl', 0),
                'positions_count': balance_data.get('positions_count', 0),
                'total_exposure': balance_data.get('total_exposure', 0)
            }
        )
        
        await self._log_event(event)
    
    async def log_risk_event(self, risk_data: Dict[str, Any]):
        """Log risk management events"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=time.time(),
            event_type=EventType.RISK_LIMIT_HIT,
            symbol=risk_data.get('symbol'),
            data={
                'risk_type': risk_data.get('risk_type'),
                'current_value': risk_data.get('current_value'),
                'limit_value': risk_data.get('limit_value'),
                'action_taken': risk_data.get('action_taken'),
                'drawdown_pct': risk_data.get('drawdown_pct'),
                'var_95': risk_data.get('var_95'),
                'correlation_risk': risk_data.get('correlation_risk'),
                'leverage': risk_data.get('leverage'),
                'message': risk_data.get('message')
            }
        )
        
        await self._log_event(event)
    
    async def log_error_event(self, error_data: Dict[str, Any]):
        """Log system errors"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=time.time(),
            event_type=EventType.ERROR,
            symbol=error_data.get('symbol'),
            data={
                'error_type': error_data.get('error_type', 'unknown'),
                'error_message': error_data.get('message', ''),
                'component': error_data.get('component', 'unknown'),
                'severity': error_data.get('severity', 'medium'),
                'stack_trace': error_data.get('stack_trace', ''),
                'recovery_action': error_data.get('recovery_action', ''),
                'context': error_data.get('context', {})
            }
        )
        
        await self._log_event(event)
    
    async def log_heartbeat(self):
        """Log system heartbeat"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=time.time(),
            event_type=EventType.HEARTBEAT,
            symbol=None,
            data={'status': 'alive', 'memory_usage': self._get_memory_usage()}
        )
        
        await self._log_event(event)
    
    async def _log_event(self, event: AuditEvent):
        """Internal method to log an event"""
        # Set previous checksum for chain integrity
        event.previous_checksum = self.last_checksum
        
        # Calculate event checksum
        event.checksum = self._calculate_checksum(event)
        
        # Update last checksum
        self.last_checksum = event.checksum
        
        # Add to buffer
        self.event_buffer.append(event)
        
        # Write to file
        await self._write_event_to_file(event)
        
        # Rotate log file if date changed
        await self._check_log_rotation()
    
    async def _write_event_to_file(self, event: AuditEvent):
        """Write event to audit log file"""
        try:
            log_line = json.dumps(event.to_dict(), separators=(',', ':')) + '\n'
            
            async with aiofiles.open(self.log_file, 'a') as f:
                await f.write(log_line)
            
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")
    
    async def _check_log_rotation(self):
        """Check if log file needs rotation (daily)"""
        current_date = datetime.now().strftime('%Y%m%d')
        
        if current_date != self.current_date:
            # Create new log file for new date
            self.current_date = current_date
            self.log_file = self.log_dir / f"audit_{self.current_date}.log"
            logger.info(f"Rotated audit log to: {self.log_file}")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'cpu_percent': process.cpu_percent()
            }
        except ImportError:
            return {'memory_monitoring': 'psutil not available'}
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent events from buffer"""
        recent_events = list(self.event_buffer)[-count:]
        return [event.to_dict() for event in recent_events]
    
    async def verify_audit_chain(self, log_file_path: Optional[str] = None) -> Dict[str, Any]:
        """Verify integrity of audit chain"""
        target_file = Path(log_file_path) if log_file_path else self.log_file
        
        if not target_file.exists():
            return {'valid': False, 'error': 'Log file not found'}
        
        verification_result = {
            'valid': True,
            'total_events': 0,
            'checksum_errors': [],
            'chain_breaks': []
        }
        
        try:
            previous_checksum = None
            
            async with aiofiles.open(target_file, 'r') as f:
                line_num = 0
                async for line in f:
                    line_num += 1
                    try:
                        event_dict = json.loads(line.strip())
                        verification_result['total_events'] += 1
                        
                        # Verify checksum chain
                        if event_dict.get('previous_checksum') != previous_checksum:
                            verification_result['chain_breaks'].append({
                                'line': line_num,
                                'expected': previous_checksum,
                                'found': event_dict.get('previous_checksum')
                            })
                        
                        previous_checksum = event_dict.get('checksum')
                        
                    except json.JSONDecodeError as e:
                        verification_result['checksum_errors'].append({
                            'line': line_num,
                            'error': str(e)
                        })
            
            verification_result['valid'] = (
                len(verification_result['checksum_errors']) == 0 and
                len(verification_result['chain_breaks']) == 0
            )
            
        except Exception as e:
            verification_result = {
                'valid': False,
                'error': str(e)
            }
        
        return verification_result

# Integration with MirrorCore-X
class MirrorCoreAuditLogger:
    """Specialized audit logger for MirrorCore-X integration"""
    
    def __init__(self, syncbus, audit_logger: AuditLogger):
        self.syncbus = syncbus
        self.audit_logger = audit_logger
        self.last_heartbeat = time.time()
        
        # Start heartbeat task
        asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await self.audit_logger.log_heartbeat()
                self.last_heartbeat = time.time()
                await asyncio.sleep(60)  # Heartbeat every minute
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(60)
    
    async def audit_trade_execution(self, order_data: Dict[str, Any], execution_result: Dict[str, Any]):
        """Audit complete trade execution flow"""
        # Log order submission
        await self.audit_logger.log_order_event(order_data, EventType.ORDER_SUBMITTED)
        
        # Log execution result
        if execution_result.get('status') == 'filled':
            await self.audit_logger.log_order_event(execution_result, EventType.ORDER_FILLED)
        elif execution_result.get('status') == 'cancelled':
            await self.audit_logger.log_order_event(execution_result, EventType.ORDER_CANCELLED)
    
    async def audit_scanner_signals(self, scanner_results: List[Dict[str, Any]]):
        """Audit scanner signal generation"""
        for signal in scanner_results:
            await self.audit_logger.log_signal_event(signal)
    
    async def audit_system_health(self, health_data: Dict[str, Any]):
        """Audit system health metrics"""
        if health_data.get('error'):
            await self.audit_logger.log_error_event(health_data)
        
        if health_data.get('balance_update'):
            await self.audit_logger.log_balance_update(health_data['balance_update'])

# Usage example
if __name__ == "__main__":
    async def test_audit_system():
        audit_logger = AuditLogger()
        
        # Test order event
        order_data = {
            'id': 'test_order_123',
            'symbol': 'BTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'amount': 0.1,
            'price': 45000.0,
            'status': 'open',
            'strategy': 'momentum_scanner'
        }
        
        await audit_logger.log_order_event(order_data, EventType.ORDER_SUBMITTED)
        
        # Test signal event
        signal_data = {
            'symbol': 'BTC/USDT',
            'signal': 'Strong Buy',
            'confidence_score': 0.85,
            'composite_score': 78.5,
            'price': 45000.0
        }
        
        await audit_logger.log_signal_event(signal_data)
        
        # Verify audit chain
        verification = await audit_logger.verify_audit_chain()
        print(f"Audit chain verification: {verification}")
        
        # Get recent events
        recent = audit_logger.get_recent_events(5)
        print(f"Recent events: {len(recent)}")
    
    asyncio.run(test_audit_system())
