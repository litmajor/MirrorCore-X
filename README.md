# MirrorCore-X: Multi-Agent Cognitive Trading System

## Overview
MirrorCore-X is an advanced, modular trading system for cryptocurrency and forex markets. It integrates multiple cognitive agents for market analysis, decision-making, trade execution, performance tracking, psychological modeling, strategy management, and hyperparameter optimization. The system leverages technical indicators, emotional control, and Bayesian optimization for enhanced trading performance.

---

## Features
- **Multi-agent architecture** with asynchronous coordination via SyncBus
- **MomentumScanner** for real-time market signal generation
- **TradeAnalyzerAgent** for performance metrics and trade logging
- **ARCH_CTRL** for emotional state management and trade gating
- **StrategyTrainerAgent** for dynamic strategy weighting
- **MirrorOptimizerAgent** for hyperparameter tuning using Bayesian optimization
- **RiskSentinel** for robust risk management
- **Simulated market data generation** for testing
- **Support for dry-run and live trading modes**
- **Comprehensive logging and error handling**

---

## Architecture

### High-Level Diagram
```
+-------------------+
|   Market Data     |
+-------------------+
         |
         v
+-------------------+
| MomentumScanner   |
+-------------------+
         |
         v
+-------------------+
| PerceptionLayer   |
+-------------------+
         |
         v
+-------------------+
|  SyncBus (Core)   |
+-------------------+
   |   |   |   |   |
   v   v   v   v   v
+-----+-----+-----+-----+-----+
|Ego  |Fear |Self |Oracle|Exec|
|Proc |Anal |Aware|      |Daemon
+-----+-----+-----+-----+-----+
         |
         v
+-------------------+
| TradingBot        |
+-------------------+
         |
         v
+-------------------+
| TradeAnalyzer     |
+-------------------+
         |
         v
+-------------------+
| MirrorOptimizer   |
+-------------------+
```

### Component Overview
- **MomentumScanner**: Scans markets, generates signals
- **PerceptionLayer**: Processes scanner and market data
- **SyncBus**: Central async state manager, coordinates agents
- **EgoProcessor**: Manages psychological state, sentiment bias
- **FearAnalyzer**: Analyzes fear levels from market/trade outcomes
- **SelfAwarenessAgent**: Tracks performance, agent deviations
- **TradingOracleEngine**: Generates trading directives
- **ExecutionDaemon**: Executes trades (dry-run/live)
- **TradingBot**: Manages trade execution, stop-loss/take-profit
- **TradeAnalyzerAgent**: Logs trades, computes metrics
- **MirrorOptimizerAgent**: Optimizes agent hyperparameters
- **RiskSentinel**: Enforces risk limits
- **MarketDataGenerator**: Simulates market data for testing

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Dependencies
Install required packages:
```bash
pip install pandas numpy ccxt ta tqdm plotly pydantic bayes_opt aiohttp
```

---

## Usage

### 1. Quick Start (Demo Session)
Run a demo trading session (dry-run):
```bash
python mirrax.py
```
Or for the original version:
```bash
python mirrorcore_x.py
```

### 2. Main Entry Points
- `mirrax.py`: New, production-ready version
- `mirrorcore_x.py`: Original, cognitive modeling version

### 3. Customization
- Edit `TradingConfig` in `scanner.py` for timeframes, thresholds, and parameters
- Add/modify strategies in `strategy_trainer_agent.py` and `strategies/`
- Integrate new exchanges via CCXT

---

## Example: Running a Custom Session
```python
import asyncio
from mirrax import run_demo_session

asyncio.run(run_demo_session(ticks=100, dry_run=True, tick_interval=0.2))
```

---

## File Structure
```
MirrorCore-X/
├── mirrax.py                # Main system (new version)
├── mirrorcore_x.py          # Original cognitive system
├── scanner.py               # MomentumScanner agent
├── trade_analyzer_agent.py  # TradeAnalyzer agent
├── arch_ctrl.py             # Emotional state controller
├── strategy_trainer_agent.py# Strategy trainer agent
├── mirror_optimizer.py      # Optimizer agent
├── strategies/              # External strategy modules
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

---

## Agent API Example
```python
from scanner import MomentumScanner
scanner = MomentumScanner(exchange)
results = await scanner.scan_market()
```

---

## Advanced: Live Trading
- Set `dry_run=False` in `run_demo_session` or `ExecutionDaemon` to enable live trading (ensure proper API keys and risk controls)

---

## Troubleshooting
- Check `mirrorcore_x.log` and `momentum_scanner.log` for detailed logs
- Ensure all dependencies are installed
- For CCXT exchange errors, verify API credentials and network connectivity

---

## Contributing
Pull requests and issues are welcome! Please document new agents and strategies clearly.

---

## License
MIT License

---

## Contact
For questions or support, open an issue or contact the maintainer.

---

## Further Reading
- [CCXT Documentation](https://github.com/ccxt/ccxt)
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)
- [Pydantic](https://docs.pydantic.dev/)
- [TA-Lib](https://github.com/bukosabino/ta)

---

## Diagram Source
Diagrams generated with ASCII and [draw.io](https://draw.io) for architecture illustration.

---

## Changelog
- v1.0: Initial release
- v1.1: Added OptimizableAgent interface, improved error handling, documentation
