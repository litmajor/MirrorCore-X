import asyncio
import json
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import eventlet
import threading
import time
from typing import Dict, Any

# Import your existing components
from mirrorcore_x import HighPerformanceSyncBus, create_mirrorcore_system
from scanner import MomentumScanner, get_dynamic_config
import ccxt
from trade_analyzer_agent import TradeAnalyzerAgent

eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mirrorcore-dashboard-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state
dashboard_state = {
    'running': False,
    'sync_bus': None,
    'scanner': None,
    'tick_count': 0,
    'performance_metrics': {},
    'strategy_grades': {},
    'recent_trades': [],
    'system_health': {}
}

logger = logging.getLogger(__name__)

class DashboardManager:
    def __init__(self):
        self.running = False
        self.sync_bus = None
        self.scanner = None
        self.trade_analyzer = None
        self.exchange = None
        self.active_connections = set() # To keep track of connected WebSocket clients

    async def initialize_system(self):
        """Initialize the trading system"""
        try:
            # Initialize exchange and scanner
            self.exchange = ccxt.kucoinfutures()
            self.exchange.enableRateLimit = True
            config = get_dynamic_config()
            self.scanner = MomentumScanner(self.exchange, config=config)
            # Create MirrorCore system
            system_components = await create_mirrorcore_system(dry_run=True)
            self.sync_bus = system_components[0]
            self.trade_analyzer = system_components[1]
            logger.info("Dashboard system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False

    async def run_simulation_loop(self):
        """Main simulation loop"""
        self.running = True
        tick_count = 0
        while self.running:
            try:
                if self.sync_bus is None:
                    logger.error("sync_bus is not initialized.")
                    await asyncio.sleep(1.0)
                    continue
                # Process one tick
                await self.sync_bus.tick()
                tick_count += 1
                # Get current state
                # Get real market data from scanner
                scanner_results = None
                if self.scanner:
                    try:
                        scanner_results = await self.scanner.scan_market(timeframe='1h', top_n=50)
                    except Exception as e:
                        logger.error(f"Scanner error: {e}")

                # Prepare dashboard data with real scanner results
                dashboard_data = {
                    'tick_count': tick_count,
                    'timestamp': datetime.now().isoformat(),
                    'market_data': scanner_results.to_dict('records') if scanner_results is not None and not scanner_results.empty else [],
                    'scanner_data': await self.sync_bus.get_state('scanner_data') if self.sync_bus else {},
                    'emotional_state': await self.sync_bus.get_state('emotional_state') if self.sync_bus else {},
                    'strategy_grades': await self.sync_bus.get_state('strategy_grades') if self.sync_bus else {},
                    'system_performance': await self.sync_bus.get_state('system_performance') if self.sync_bus else {},
                    'oracle_directives': await self.sync_bus.get_state('oracle_directives') if self.sync_bus else {},
                    'trades': await self.sync_bus.get_state('trades') if self.sync_bus else []
                }

                # Emit to connected clients
                socketio.emit('dashboard_update', dashboard_data)

                # Also emit market overview
                if scanner_results is not None and not scanner_results.empty:
                    market_overview = {
                        'total_volume': float(scanner_results['average_volume_usd'].sum()),
                        'market_sentiment': float(len(scanner_results[scanner_results['signal'].isin(['Strong Buy', 'Buy'])]) / len(scanner_results)),
                        'active_signals': int(len(scanner_results[scanner_results['composite_score'] > 70]))
                    }
                    socketio.emit('market_overview', market_overview)
                # Wait before next tick
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(1.0)

    async def broadcast_market_data(self):
        """Broadcast market data to all connected clients"""
        while self.running:
            try:
                # Get comprehensive data from sync_bus
                scanner_data = await self.sync_bus.get_state('scanner_data') or []
                parallel_data = await self.sync_bus.get_state('parallel_scanner_data') or []
                trades = await self.sync_bus.get_state('trades') or []
                oracle_directives = await self.sync_bus.get_state('oracle_directives') or []
                system_performance = await self.sync_bus.get_state('system_performance') or {}

                # Calculate real-time metrics
                active_trades = [t for t in trades if t.get('exit') is None]
                total_pnl = sum(t.get('pnl', 0) for t in trades)
                win_rate = len([t for t in trades if t.get('pnl', 0) > 0]) / len(trades) if trades else 0

                # Prepare comprehensive broadcast data
                data = {
                    'type': 'market_update',
                    'scanner_data': scanner_data[:20],
                    'parallel_exchanges': len(set(d.get('exchange') for d in parallel_data if d.get('exchange'))),
                    'active_signals': len([s for s in scanner_data if s.get('signal') in ['Strong Buy', 'Strong Sell']]),
                    'active_trades': len(active_trades),
                    'total_pnl': round(total_pnl, 2),
                    'win_rate': round(win_rate * 100, 1),
                    'oracle_directives': len(oracle_directives),
                    'system_performance': system_performance,
                    'timestamp': time.time()
                }

                # Broadcast to all clients
                if self.active_connections:
                    disconnected = []
                    for ws in self.active_connections:
                        try:
                            await ws.send_json(data)
                        except Exception:
                            disconnected.append(ws)

                    # Remove disconnected clients
                    for ws in disconnected:
                        self.active_connections.discard(ws)

                await asyncio.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(1)


    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False

# Global dashboard manager
dashboard_manager = DashboardManager()

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('enhanced_dashboard.html')

@app.route('/api/system/status')
def get_system_status():
    """Get current system status"""
    return jsonify({
        'running': dashboard_manager.running,
        'tick_count': dashboard_state.get('tick_count', 0),
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected to dashboard')
    emit('connection_status', {'status': 'connected'})
    # Add new connection to the set
    dashboard_manager.active_connections.add(request.sid) # Assuming request.sid is available

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected from dashboard')
    # Remove disconnected client from the set
    if hasattr(request, 'sid') and request.sid in dashboard_manager.active_connections:
        dashboard_manager.active_connections.remove(request.sid)


@socketio.on('start_system')
def handle_start_system():
    """Start the trading system"""
    try:
        if not dashboard_manager.running:
            def run_async_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(dashboard_manager.initialize_system())
                # Start the WebSocket broadcast loop
                broadcast_task = loop.create_task(dashboard_manager.broadcast_market_data())
                loop.run_until_complete(dashboard_manager.run_simulation_loop())
                # Ensure broadcast task is cancelled when simulation loop stops
                broadcast_task.cancel()
                try:
                    loop.run_until_complete(broadcast_task)
                except asyncio.CancelledError:
                    pass # Expected cancellation

            thread = threading.Thread(target=run_async_loop)
            thread.daemon = True
            thread.start()

            emit('system_status', {'status': 'starting', 'message': 'System initialization in progress'})
        else:
            emit('system_status', {'status': 'already_running', 'message': 'System is already running'})

    except Exception as e:
        logger.error(f"Error starting system: {e}")
        emit('system_status', {'status': 'error', 'message': str(e)})

@socketio.on('get_system_state')
def handle_get_system_state():
    """Get current system state"""
    try:
        state = {
            'running': dashboard_manager.running,
            'tick_count': dashboard_state.get('tick_count', 0),
            'performance': dashboard_state.get('performance_metrics', {}),
            'strategies': dashboard_state.get('strategy_grades', {})
        }
        emit('system_state', state)
    except Exception as e:
        logger.error(f"Error getting system state: {e}")
        emit('error', {'message': str(e)})

@socketio.on('stop_system')
def handle_stop_system():
    """Stop the trading system"""
    try:
        dashboard_manager.stop_simulation()
        emit('system_status', {'status': 'stopped', 'message': 'System stopped'})
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        emit('system_status', {'status': 'error', 'message': str(e)})

@socketio.on('get_strategy_performance')
def handle_get_strategy_performance():
    """Get detailed strategy performance metrics"""
    try:
        if dashboard_manager.trade_analyzer:
            performance_data = {
                'total_trades': len(dashboard_manager.trade_analyzer.trades),
                'total_pnl': dashboard_manager.trade_analyzer.get_total_pnl(),
                'strategy_breakdown': dashboard_manager.trade_analyzer.strategy_pnls,
                'recent_trades': dashboard_manager.trade_analyzer.recent_trades(10)
            }
            emit('strategy_performance', performance_data)
        else:
            emit('strategy_performance', {'error': 'Trade analyzer not available'})
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        emit('strategy_performance', {'error': str(e)})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)