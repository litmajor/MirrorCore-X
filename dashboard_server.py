
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
                # Prepare dashboard data
                dashboard_data = {
                    'tick_count': tick_count,
                    'timestamp': datetime.now().isoformat(),
                    'market_data': await self.sync_bus.get_state('market_data'),
                    'scanner_data': await self.sync_bus.get_state('scanner_data'),
                    'emotional_state': await self.sync_bus.get_state('emotional_state'),
                    'strategy_grades': await self.sync_bus.get_state('strategy_grades'),
                    'system_performance': await self.sync_bus.get_state('system_performance'),
                    'oracle_directives': await self.sync_bus.get_state('oracle_directives'),
                    'trades': await self.sync_bus.get_state('trades')
                }
                # Emit to connected clients
                socketio.emit('dashboard_update', dashboard_data)
                # Wait before next tick
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(1.0)
    
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

@socketio.on('start_system')
def handle_start_system():
    """Start the trading system"""
    try:
        if not dashboard_manager.running:
            def run_async_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(dashboard_manager.initialize_system())
                loop.run_until_complete(dashboard_manager.run_simulation_loop())
            
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
