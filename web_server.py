import time
import json
from enum import Enum
from dataclasses import is_dataclass, asdict
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import eventlet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# We must monkey patch for SocketIO to work with background threads
eventlet.monkey_patch()

# Import the core logic from your main script
try:
    from mirrorcore_x import create_mirrorcore_system, MarketData, ExecutionDaemon, ReflectionCore
    import ccxt
    from momentum_scanner import MomentumScanner
except Exception as e:
    print(f"Error importing MirrorCore-X modules: {e}")
    exit()

# --- Flask App Setup ---
app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Global state for the simulation ---
simulation_thread = None
simulation_running = False
tick_interval = 0.5 # seconds

# --- Custom JSON Encoder ---
# This is crucial for sending your custom Python objects to the frontend
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, '__dict__'):
            return o.__dict__
        return super().default(o)

# --- Simulation Background Task ---
def run_simulation():
    """
    This function runs in a background thread and continuously
    executes the simulation ticks, emitting the state to the frontend.
    """
    print("Initializing simulation components...")
    # Initialize your system just like in your original script's main block
    try:
        # This part might need adjustment based on your actual scanner setup
        exchange = ccxt.kucoinfutures({'enableRateLimit': True})
        scanner = None # The scanner is not used in the generator, so we can pass None
        if hasattr(ccxt, 'MomentumScanner'):
             scanner = scanner = MomentumScanner(exchange)
    except Exception as e:
        print(f"Could not initialize exchange/scanner. Using dummy values. Error: {e}")
        scanner = None

    mirrorcore_system, market_gen = create_mirrorcore_system(scanner)
    print("Simulation background task started.")

    # Setup DRY_RUN mode and dashboard data
    execution_daemon = ExecutionDaemon(exchange=scanner.exchange if scanner else None, capital=1000, risk_pct=0.01)
    execution_daemon.DRY_RUN = True
    reflection_core = ReflectionCore()
    dashboard_data = {
        'tick': [],
        'confidence': [],
        'fear': [],
        'stress': [],
        'drift': [],
        'trust': [],
        'grade': [],
        'virtual_balance': [],
    }

    tick_count = 0
    while True:
        if simulation_running:
            market_data = market_gen.generate_tick()
            global_state = mirrorcore_system.tick(market_data)
            tick_count += 1
            # Collect dashboard data
            dashboard_data['tick'].append(tick_count)
            psych_profile = global_state.get('psych_profile')
            dashboard_data['confidence'].append(psych_profile.confidence_level if psych_profile and hasattr(psych_profile, 'confidence_level') else 0.5)
            dashboard_data['fear'].append(global_state.get('fear_level', 0))
            psych_profile = global_state.get('psych_profile')
            dashboard_data['stress'].append(psych_profile.stress_level if psych_profile and hasattr(psych_profile, 'stress_level') else 0.5)
            dashboard_data['drift'].append(global_state.get('behavioral_drift_score', 0))
            dashboard_data['trust'].append(global_state.get('self_trust_multiplier', 0.8))
            dashboard_data['grade'].append(ord(global_state.get('session_grade', 'B')))
            dashboard_data['virtual_balance'].append(global_state.get('virtual_balance', 1000))
            # Emit state to frontend
            try:
                json_state = json.dumps(global_state, cls=CustomJSONEncoder)
                socketio.emit('update', json_state)
            except TypeError as e:
                print(f"Error serializing state to JSON: {e}")
                socketio.emit('update', json.dumps({'error': str(e)}))
        socketio.sleep(int(tick_interval))

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """
    This is called when a new client (the dashboard) connects.
    It starts the background simulation thread if it's not already running.
    """
    global simulation_thread
    print('Client connected')
    if simulation_thread is None:
        simulation_thread = socketio.start_background_task(target=run_simulation)
    emit('status', {'message': 'Connected to server.'})

@socketio.on('start_simulation')
def handle_start():
    """Starts the simulation when the 'Start' button is clicked."""
    global simulation_running
    simulation_running = True
    print("Simulation started.")
    emit('status', {'message': 'Simulation started.'})

@socketio.on('pause_simulation')
def handle_pause():
    """Pauses the simulation when the 'Pause' button is clicked."""
    global simulation_running
    simulation_running = False
    print("Simulation paused.")
    emit('status', {'message': 'Simulation paused.'})

@socketio.on('reset_simulation')
def handle_reset():
    """Resets the simulation. (For a true reset, re-initializes the system)"""
    global simulation_running, simulation_thread
    simulation_running = False
    # Stop the old thread and start a new one to reset the state
    if simulation_thread is not None:
        # A simple way to reset is to kill and restart the thread.
        # In a production app, you might want a more graceful reset method inside your agents.
        simulation_thread = None
        simulation_thread = socketio.start_background_task(target=run_simulation)
    print("Simulation reset.")
    emit('status', {'message': 'Simulation reset.'})


# --- Flask Route ---
@app.route('/')
def index():
    """Serves the dashboard HTML file."""
    # The template is rendered from the same directory
    return render_template('mirrorcore_dashboard.html')
