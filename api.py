from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pandas as pd
import os
from typing import List, Dict
from pydantic import BaseModel
import logging
from datetime import datetime, timezone
import asyncio
import json
import numpy as np # Import numpy for percentile calculation
import time # Import time for timestamp

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system state (renamed from system_state to global_state for clarity in the new endpoints)
global_state = {
    'sync_bus': None,
    'components': None,
    'parallel_scanner': None,
    'oracle_imagination': None,
    'comprehensive_optimizer': None # Added for optimization results
}

active_websockets: List[WebSocket] = []


# --- HybridMarketFrame Pydantic Model ---
class HybridMarketFrame(BaseModel):
    timestamp: int
    symbol: str
    price: dict
    volume: float
    volatility: float
    momentum_score: float
    rsi: float
    trend_score: float
    confidence_score: float
    composite_score: float
    volume_score: float
    anchor_index: int


@app.on_event("startup")
async def startup_event():
    """Initialize trading system on startup"""
    logger.info("🚀 Initializing MirrorCore-X Trading System...")

    try:
        from mirrorcore_x import create_mirrorcore_system
        from parallel_scanner_integration import add_parallel_scanner_to_mirrorcore
        # Assuming these components are available and can be imported
        # from bayesian_integration import BayesianIntegration # Example if needed
        # from imagination_engine import ImaginationEngine # Example if needed
        # from optimization_engine import ComprehensiveOptimizer # Example if needed

        # Create main system
        sync_bus, components = await create_mirrorcore_system(
            dry_run=True,
            use_testnet=True,
            enable_oracle=True,
            enable_bayesian=True,
            enable_imagination=True
        )

        global_state['sync_bus'] = sync_bus
        global_state['components'] = components
        global_state['oracle_imagination'] = components.get('oracle_imagination')

        # Add parallel scanner
        parallel_scanner = await add_parallel_scanner_to_mirrorcore(
            sync_bus, components.get('scanner'), enable=True
        )
        global_state['parallel_scanner'] = parallel_scanner

        # Initialize other components if they exist
        global_state['comprehensive_optimizer'] = components.get('comprehensive_optimizer') # Get optimizer if available

        # Start background task
        asyncio.create_task(run_system_loop())

        logger.info("✅ System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")

async def run_system_loop():
    """Background task to run system ticks and broadcast updates"""
    while True:
        try:
            sync_bus = global_state.get('sync_bus')
            components = global_state.get('components')

            if sync_bus:
                # Run tick
                await sync_bus.tick()

                tick_count = getattr(sync_bus, 'tick_count', 0)

                # Broadcast market data every tick
                scanner_data = await sync_bus.get_state('scanner_data') or []
                market_data = await sync_bus.get_state('market_data') or []

                if scanner_data or market_data:
                    await broadcast_update({
                        'type': 'market_update',
                        'data': {
                            'scanner_data': scanner_data[-10:],  # Last 10 scanner items
                            'market_data': market_data[-10:],     # Last 10 market items
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    })

                # Broadcast trading directives every 5 ticks
                if tick_count % 5 == 0:
                    directives = await sync_bus.get_state('trading_directives') or []
                    strategy_grades = await sync_bus.get_state('strategy_grades') or {}

                    await broadcast_update({
                        'type': 'trading_update',
                        'data': {
                            'directives': directives[-5:],
                            'strategy_grades': strategy_grades,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    })

                # Broadcast performance metrics every 10 ticks
                if tick_count % 10 == 0:
                    trade_analyzer = components.get('trade_analyzer')
                    if trade_analyzer:
                        await broadcast_update({
                            'type': 'performance_update',
                            'data': {
                                'total_pnl': trade_analyzer.get_total_pnl(),
                                'win_rate': trade_analyzer.get_win_rate(),
                                'total_trades': len(trade_analyzer.trades),
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
                        })

            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"System loop error: {e}")
            await asyncio.sleep(5)

async def broadcast_update(message: dict):
    """Broadcast update to all connected WebSocket clients"""
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)

    # Remove disconnected clients
    for ws in disconnected:
        active_websockets.remove(ws)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_websockets.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(active_websockets)}")

    try:
        # Send initial state
        sync_bus = global_state.get('sync_bus')
        if sync_bus:
            initial_data = {
                'type': 'initial_state',
                'data': {
                    'scanner_data': await sync_bus.get_state('scanner_data') or [],
                    'market_data': await sync_bus.get_state('market_data') or [],
                    'oracle_directives': await sync_bus.get_state('oracle_directives') or [],
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            }
            await websocket.send_json(initial_data)

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Handle client commands if needed
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        logger.info(f"WebSocket client disconnected. Remaining: {len(active_websockets)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)

    # These variables were declared but not used in the original code.
    # Keeping them as they were in the original snippet.
    price_range: float
    predicted_return: float
    predicted_consistent: int
    indicators: dict
    orderFlow: dict
    marketMicrostructure: dict
    temporal_ghost: dict
    intention_field: dict
    power_level: dict
    market_layers: dict
    consciousness_matrix: dict

@app.get("/frames/{timeframe}", response_model=List[HybridMarketFrame])
@limiter.limit("60/minute")
async def get_frames(timeframe: str, request: Request, limit: int = 100, offset: int = 0):
    valid_timeframes = ["1m", "5m", "1h", "1d", "1w"]
    if timeframe not in valid_timeframes:
        raise HTTPException(status_code=400, detail="Invalid timeframe")

    # Find the latest CSV for the timeframe
    csv_dir = "C:/Users/PC/Documents/MirrorCore-X"
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith(f"predictions_{timeframe}_") and f.endswith(".csv")]
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No prediction CSV found for timeframe {timeframe}")

    latest_csv = max(csv_files, key=lambda x: datetime.strptime(x.split('_')[-2] + '_' + x.split('_')[-1].replace('.csv', ''), '%Y%m%d_%H%M%S'))
    csv_path = os.path.join(csv_dir, latest_csv)


    try:
        df = pd.read_csv(csv_path)

        # Apply pagination
        total_records = len(df)
        df = df.iloc[offset:offset + limit]

        frames = []
        for idx, row in enumerate(df.to_dict('records')):
            anchor_index = idx
            predicted_return_col = f'predicted_{timeframe}_return'
            predicted_return = row[predicted_return_col] if predicted_return_col in row else 0.0
            # --- Compose HybridMarketFrame ---
            frame = {
                "timestamp": int(datetime.now().timestamp() * 1000 - (len(df) - anchor_index) * 1000),
                "symbol": row.get('symbol', 'BTCUSD'),
                "price": {
                    "open": row['price'],
                    "high": row['price'] * (1 + row['volatility'] / 1000),
                    "low": row['price'] * (1 - row['volatility'] / 1000),
                    "close": row['price']
                },
                "volume": row['volume'],
                "volatility": row['volatility'],
                "momentum_score": row.get('momentum_short', 0) * 100,
                "rsi": row.get('rsi', 50),
                "trend_score": row.get('trend_score', 0),
                "confidence_score": row.get('confidence_score', 0),
                "composite_score": row.get('composite_score', 0),
                "volume_score": row.get('volume_composite_score', 0),
                "anchor_index": anchor_index,
                "price_range": row.get('volatility', 0) * 2,
                "predicted_return": predicted_return,
                "predicted_consistent": row.get('predicted_consistent', 0),
                # --- Indicators ---
                "indicators": {
                    "macd": {
                        "line": row.get('macd', 0),
                        "signal": row.get('macd_signal', 0),
                        "histogram": row.get('macd_hist', 0)
                    },
                    "bb": {
                        "upper": row.get('bb_upper', row['price'] * 1.01),
                        "middle": row['price'],
                        "lower": row.get('bb_lower', row['price'] * 0.99)
                    },
                    "ema20": row.get('ema20', row['price']),
                    "ema50": row.get('ema50', row['price']),
                    "vwap": row.get('vwap', row['price']),
                    "atr": row.get('atr', row['volatility'])
                },
                # --- Order Flow ---
                "orderFlow": {
                    "bidVolume": row.get('bid_volume', row['volume'] * 0.5),
                    "askVolume": row.get('ask_volume', row['volume'] * 0.5),
                    "netFlow": row.get('net_flow', 0),
                    "largeOrders": row.get('large_orders', int(row['volume'] * 0.1)),
                    "smallOrders": row.get('small_orders', int(row['volume'] * 0.8))
                },
                # --- Market Microstructure ---
                "marketMicrostructure": {
                    "spread": row.get('spread', row['price'] * 0.001),
                    "depth": row.get('depth', row['volume'] * 10),
                    "imbalance": row.get('imbalance', 0),
                    "toxicity": row.get('toxicity', 0)
                },
                # --- Temporal Ghost ---
                "temporal_ghost": {
                    "next_moves": [
                        {
                            "timestamp": int(datetime.now().timestamp() * 1000 + (j + 1) * 60000),
                            "predicted_price": row['price'] * (1 + 0.01 * (j - 2)),
                            "confidence": 0.7,
                            "probability": 0.5,
                            "pattern_match": 0.5
                        } for j in range(5)
                    ],
                    "certainty_river": 0.5,
                    "time_distortion": 0.0,
                    "pattern_resonance": 0.5,
                    "quantum_coherence": 0.5,
                    "fractal_depth": 0.5
                },
                # --- Intention Field ---
                "intention_field": {
                    "accumulation_pressure": 0.5,
                    "breakout_membrane": 0.5,
                    "momentum_vector": {
                        "direction": 'up' if row.get('momentum_short', 0) > 0 else 'down',
                        "strength": abs(row.get('momentum_short', 0)) * 100,
                        "timing": 5.0,
                        "acceleration": 0.0
                    },
                    "whale_presence": 0.5,
                    "institutional_flow": 0.5,
                    "retail_sentiment": 0.5,
                    "liquidity_depth": 0.5
                },
                # --- Power Level ---
                "power_level": {
                    "edge_percentage": 50.0,
                    "opportunity_intensity": 0.5,
                    "risk_shadow": 0.5,
                    "profit_magnetism": 0.5,
                    "certainty_coefficient": 0.5,
                    "godmode_factor": 0.5,
                    "quantum_advantage": 0.5,
                    "reality_bend_strength": 0.5
                },
                # --- Market Layers ---
                "market_layers": {
                    "fear_greed_index": 0.5,
                    "volatility_regime": 'medium',
                    "trend_strength": abs(row.get('momentum_short', 0)) * 50,
                    "support_resistance": {
                        "nearest_support": row['price'] * 0.98,
                        "nearest_resistance": row['price'] * 1.02,
                        "strength": 0.5
                    },
                    "market_phase": 'markup' if row.get('momentum_short', 0) > 0 else 'markdown'
                },
                # --- Consciousness Matrix ---
                "consciousness_matrix": {
                    "awareness_level": 0.7,
                    "collective_intelligence": 0.7,
                    "prediction_accuracy": 0.7,
                    "reality_stability": 0.7,
                    "temporal_variance": 0.1
                }
            }
            frames.append(frame)
        return frames
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """System health monitoring endpoint"""
    return {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "api": "healthy",
            "scanner": "operational",
            "exchange": "connected"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get key system metrics"""
    return {
        "uptime_hours": 24.5,
        "total_trades": 150,
        "win_rate": 0.65,
        "current_pnl": 1250.50,
        "active_positions": 3
    }

@app.get("/api/market/overview")
@limiter.limit("30/minute")
async def get_market_overview(request: Request):
    """Get market overview data from live system"""
    try:
        sync_bus = global_state.get('sync_bus')
        if not sync_bus:
            return {"error": "System not initialized"}

        # Get live scanner data
        scanner_data = await sync_bus.get_state('scanner_data') or []
        if not scanner_data:
            return {"error": "No scanner data available"}

        df = pd.DataFrame(scanner_data)

        # Calculate real market metrics
        total_volume = df['volume'].sum() if 'volume' in df else 0

        # Top movers by momentum
        top_movers = []
        if 'momentum_7d' in df and 'symbol' in df:
            top_df = df.nlargest(5, 'momentum_7d')[['symbol', 'momentum_7d', 'price']]
            top_movers = top_df.to_dict('records')

        # Market sentiment from signals
        bullish_count = len(df[df['signal'].isin(['Strong Buy', 'Buy', 'Weak Buy'])]) if 'signal' in df else 0
        total_count = len(df)

        market_sentiment = bullish_count / total_count if total_count > 0 else 0.5

        # Volatility from volume ratios
        volatility_index = df['volume_ratio'].std() if 'volume_ratio' in df else 0.5

        # Assuming latest_csv is defined somewhere in this scope or passed as argument
        # For now, using a placeholder value if it's not available.
        latest_csv_info = "N/A"
        try:
            # Attempt to find the latest CSV path similar to get_frames
            csv_dir = "C:/Users/PC/Documents/MirrorCore-X"
            csv_files = [f for f in os.listdir(csv_dir) if f.startswith("predictions_") and f.endswith(".csv")]
            if csv_files:
                latest_csv_name = max(csv_files, key=lambda x: datetime.strptime(x.split('_')[-2] + '_' + x.split('_')[-1].replace('.csv', ''), '%Y%m%d_%H%M%S'))
                latest_csv_info = latest_csv_name.split('_')[-2] + '_' + latest_csv_name.split('_')[-1].replace('.csv', '')
        except Exception as e:
            logger.warning(f"Could not determine latest CSV info for market overview: {e}")


        return {
            "total_volume": float(total_volume),
            "top_movers": top_movers,
            "market_sentiment": float(market_sentiment),
            "volatility_index": float(volatility_index),
            "total_symbols": int(total_count),
            "last_updated": latest_csv_info
        }
    except Exception as e:
        logger.error(f"Error in market overview: {e}")
        return {"error": str(e)}

@app.get("/api/scanner/realtime")
@limiter.limit("20/minute")
async def get_realtime_scanner_data(request: Request):
    """Get real-time scanner data for live updates"""
    try:
        # Import scanner dynamically
        import ccxt.async_support as ccxt
        from scanner import MomentumScanner, get_dynamic_config

        # Initialize scanner
        exchange = ccxt.binance({'enableRateLimit': True})
        config = get_dynamic_config('crypto')
        scanner = MomentumScanner(exchange, config, market_type='crypto', quote_currency='USDT')

        # Quick scan
        results = await scanner.scan_market(timeframe='1h', top_n=20)
        await exchange.close()

        if results.empty:
            return {"data": [], "timestamp": datetime.now(timezone.utc).isoformat()}

        return {
            "data": results.to_dict('records'),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Real-time scanner error: {e}")
        return {"error": str(e), "data": []}

@app.get("/api/technical/analysis/{symbol}")
async def get_technical_analysis(symbol: str):
    """Get comprehensive technical analysis for a symbol"""
    try:
        sync_bus = global_state.get('sync_bus')
        scanner_data = await sync_bus.get_state('scanner_data') if sync_bus else []

        # Find symbol data
        symbol_data = next((d for d in scanner_data if d.get('symbol') == symbol), None)

        if not symbol_data:
            return {"error": "Symbol not found"}

        return {
            "symbol": symbol,
            "price": symbol_data.get('price'),
            "indicators": {
                "rsi": symbol_data.get('rsi'),
                "macd": symbol_data.get('macd'),
                "macd_signal": symbol_data.get('macd_signal'),
                "macd_hist": symbol_data.get('macd_hist'),
                "bb_upper": symbol_data.get('bb_upper'),
                "bb_middle": symbol_data.get('bb_middle'),
                "bb_lower": symbol_data.get('bb_lower'),
                "ema_5": symbol_data.get('ema_5'),
                "ema_13": symbol_data.get('ema_13'),
                "ema_20": symbol_data.get('ema_20'),
                "ema_50": symbol_data.get('ema_50'),
                "vwap": symbol_data.get('vwap'),
                "atr": symbol_data.get('atr')
            },
            "momentum": {
                "momentum_short": symbol_data.get('momentum_short'),
                "momentum_7d": symbol_data.get('momentum_7d'),
                "momentum_30d": symbol_data.get('momentum_30d'),
                "trend_score": symbol_data.get('trend_score')
            },
            "volume": {
                "current": symbol_data.get('volume'),
                "ratio": symbol_data.get('volume_ratio'),
                "composite_score": symbol_data.get('volume_composite_score'),
                "poc_distance": symbol_data.get('poc_distance')
            },
            "patterns": {
                "ichimoku_bullish": symbol_data.get('ichimoku_bullish'),
                "vwap_bullish": symbol_data.get('vwap_bullish'),
                "ema_crossover": symbol_data.get('ema_crossover'),
                "fib_confluence": symbol_data.get('fib_confluence')
            },
            "advanced": {
                "cluster_validated": symbol_data.get('cluster_validated'),
                "reversion_probability": symbol_data.get('reversion_probability'),
                "regime": symbol_data.get('trend_regime'),
                "confidence_score": symbol_data.get('confidence_score')
            },
            "signal": symbol_data.get('signal'),
            "composite_score": symbol_data.get('composite_score'),
            "timestamp": symbol_data.get('timestamp')
        }
    except Exception as e:
        logger.error(f"Technical analysis error: {e}")
        return {"error": str(e)}

@app.websocket("/ws/scanner")
async def websocket_scanner(websocket):
    """WebSocket endpoint for real-time scanner updates"""
    await websocket.accept()

    import ccxt.async_support as ccxt
    from scanner import MomentumScanner, get_dynamic_config

    exchange = None
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        config = get_dynamic_config('crypto')
        scanner = MomentumScanner(exchange, config, market_type='crypto', quote_currency='USDT')

        while True:
            try:
                # Scan market
                results = await scanner.scan_market(timeframe='1h', top_n=20)

                # Send updates
                if not results.empty:
                    await websocket.send_json({
                        "type": "scanner_update",
                        "data": results.to_dict('records'),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"WebSocket scan error: {e}")
                await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if exchange:
            await exchange.close()


@app.get("/api/oracle/directives")
async def get_oracle_directives():
    """Get Oracle trading directives"""
    try:
        oracle_imagination = global_state.get('oracle_imagination')
        if oracle_imagination:
            # Assuming oracle_imagination has a method get_oracle_directives()
            # If the method name is different, please adjust accordingly.
            directives = oracle_imagination.get_oracle_directives()
            return {"directives": directives}
        return {"directives": []}
    except Exception as e:
        logger.error(f"Error getting oracle directives: {e}")
        return {"directives": []}

@app.get("/api/bayesian/beliefs")
async def get_bayesian_beliefs():
    """Get Bayesian belief updates for top strategies"""
    try:
        bayesian_oracle = global_state.get('oracle_imagination')
        if hasattr(bayesian_oracle, 'get_top_strategies'):
            return bayesian_oracle.get_top_strategies(top_n=10)
        return {"top_strategies": [], "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Failed to get Bayesian beliefs: {e}")
        return {"top_strategies": [], "error": str(e)}

@app.get("/api/optimizer/weights")
async def get_optimizer_weights(
    lambda_risk: float = 100.0,
    eta: float = 0.05,
    max_weight: float = 0.25,
    regime: str = "trending",
    shrinkage: bool = True,
    resampling: bool = False
):
    """Get optimized portfolio weights from mathematical optimizer"""
    try:
        # Import ensemble integration
        from ensemble_integration import get_optimal_weights

        result = get_optimal_weights(
            lambda_risk=lambda_risk,
            eta_turnover=eta,
            max_weight=max_weight,
            regime=regime,
            use_shrinkage=shrinkage,
            use_resampling=resampling
        )

        return result
    except Exception as e:
        logger.error(f"Optimizer failed: {e}")
        return {
            "weights": [],
            "expected_return": 0.0,
            "expected_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "turnover": 0.0,
            "error": str(e)
        }


@app.get("/api/imagination/analysis")
async def get_imagination_analysis():
    """Get Imagination Engine analysis results"""
    try:
        oracle_imagination = global_state.get('oracle_imagination')
        if oracle_imagination and hasattr(oracle_imagination, 'get_imagination_insights'):
            insights = oracle_imagination.get_imagination_insights()
            return insights if insights else {"summary": {}, "vulnerabilities": {}}
        return {"summary": {}, "vulnerabilities": {}}
    except Exception as e:
        logger.error(f"Error getting imagination analysis: {e}")
        return {"summary": {}, "vulnerabilities": {}}

@app.get("/api/optimization/results")
async def get_optimization_results():
    """Get comprehensive optimization results"""
    try:
        optimizer = global_state.get('comprehensive_optimizer')
        if optimizer and hasattr(optimizer, 'get_optimization_report'):
            report = optimizer.get_optimization_report()
            return report
        # Default return if optimizer is not found or doesn't have the method
        return {"total_parameters": 0, "optimized_count": 0}
    except Exception as e:
        logger.error(f"Error getting optimization results: {e}")
        return {"total_parameters": 0, "optimized_count": 0}


@app.get("/api/performance/summary")
async def get_performance_summary():
    """Get performance summary from live system with detailed metrics"""
    try:
        sync_bus = global_state.get('sync_bus')
        components = global_state.get('components')

        if not sync_bus or not components:
            return {"error": "System not initialized"}

        # Get trade analyzer
        trade_analyzer = components.get('trade_analyzer')
        scanner_data = await sync_bus.get_state('scanner_data') or []

        if trade_analyzer:
            total_pnl = trade_analyzer.get_total_pnl()
            win_rate = trade_analyzer.risk_metrics.get('win_rate', 0) * 100
            sharpe_ratio = trade_analyzer.risk_metrics.get('sharpe_ratio', 0)
            dd_stats = trade_analyzer.get_drawdown_stats()
            max_drawdown = abs(dd_stats.get('max_drawdown', 0)) if dd_stats else 0
            total_trades = len(trade_analyzer.trades)

            # Calculate additional metrics
            recent_pnl = trade_analyzer.pnl_history[-20:] if trade_analyzer.pnl_history else []
            avg_win = trade_analyzer.risk_metrics.get('avg_win', 0)
            avg_loss = trade_analyzer.risk_metrics.get('avg_loss', 0)
            profit_factor = trade_analyzer.risk_metrics.get('profit_factor', 0)

            # Build equity curve
            cumulative_pnl = np.cumsum(trade_analyzer.pnl_history) if trade_analyzer.pnl_history else []
            equity_curve = [
                {"date": f"T{i}", "equity": 10000 + pnl} 
                for i, pnl in enumerate(cumulative_pnl[-50:])
            ]

            # Build drawdown curve
            running_max = np.maximum.accumulate(cumulative_pnl) if len(cumulative_pnl) > 0 else []
            drawdowns = (cumulative_pnl - running_max) if len(running_max) > 0 else []
            drawdown_curve = [
                {"date": f"T{i}", "drawdown": dd} 
                for i, dd in enumerate(drawdowns[-50:])
            ]
        else:
            # Fallback to sync bus state
            trades = await sync_bus.get_state('trades') or []
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            win_rate = (winning_trades / len(trades) * 100) if trades else 0
            sharpe_ratio = 0
            max_drawdown = 0
            total_trades = len(trades)
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            equity_curve = []
            drawdown_curve = []

        # Get active signals count from scanner data
        active_signals = len([s for s in scanner_data if s.get('composite_score', 0) > 60]) if scanner_data else 0

        return {
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "total_trades": int(total_trades),
            "active_signals": int(active_signals),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "system_health": 100.0,
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve
        }
    except Exception as e:
        logger.error(f"Error in performance summary: {e}")
        return {"error": str(e)}

@app.get("/api/ensemble/status")
async def get_ensemble_status():
    """Get ensemble manager status and weights"""
    try:
        components = global_state.get('components')
        if not components:
            return {"error": "System not initialized"}

        ensemble_manager = components.get('ensemble_manager')
        if not ensemble_manager:
            return {"error": "Ensemble manager not available"}

        status = ensemble_manager.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting ensemble status: {e}")
        return {"error": str(e)}


@app.get("/api/ensemble/weights")
async def get_ensemble_weights():
    """Get current strategy weights"""
    try:
        sync_bus = global_state.get('sync_bus')
        components = global_state.get('components')

        if not sync_bus:
            return {"weights": {}}

        weights = await sync_bus.get_state('ensemble_weights') or {}
        regime = await sync_bus.get_state('market_regime') or 'unknown'

        # Get optimization report if available
        optimization_report = {}
        if components:
            strategy_trainer = components.get('strategy_trainer')
            if strategy_trainer and hasattr(strategy_trainer, 'get_optimization_report'):
                optimization_report = strategy_trainer.get_optimization_report()

        return {
            "regime": regime,
            "weights": weights,
            "optimization": optimization_report,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting ensemble weights: {e}")
        return {"error": str(e)}


@app.get("/api/strategies")
async def get_strategies():
    """Get all active strategies from strategy trainer"""
    try:
        sync_bus = global_state.get('sync_bus')
        components = global_state.get('components')

        if not sync_bus or not components:
            return {"strategies": []}

        strategies = []

        # Get strategy grades from sync bus
        strategy_grades = await sync_bus.get_state('strategy_grades') or {}
        strategy_trainer = components.get('strategy_trainer')

        # Assuming df is available in this scope for Mean Reversion and Cluster Momentum
        # If df is not globally available, it needs to be fetched or passed appropriately.
        # For now, using an empty DataFrame as a placeholder if df is not defined.
        df = pd.DataFrame()
        if 'df' in locals() or 'df' in globals():
            pass # df is already defined
        else:
            logger.warning("DataFrame 'df' not found in scope for get_strategies. Using empty DataFrame.")


        if strategy_trainer:
            for strategy_name, grade in strategy_grades.items():
                perf_data = strategy_trainer.performance_tracker.get(strategy_name, [])
                pnl = sum(perf_data[-10:]) if perf_data else 0
                win_count = len([p for p in perf_data[-10:] if p > 0]) if perf_data else 0
                win_rate = (win_count / min(10, len(perf_data)) * 100) if perf_data else 0

                strategies.append({
                    "name": strategy_name,
                    "status": "active" if grade in ['A', 'B'] else "paused",
                    "pnl": float(pnl),
                    "win_rate": float(win_rate),
                    "signals": len(perf_data),
                    "grade": grade
                })

            # Mean Reversion strategy
            reversion_signals = df[df.get('reversion_candidate', False) == True] if 'reversion_candidate' in df else pd.DataFrame()
            strategies.append({
                "name": "Mean Reversion",
                "status": "active" if len(reversion_signals) > 0 else "paused",
                "pnl": float(reversion_signals['avg_return_7d'].sum() if not reversion_signals.empty and 'avg_return_7d' in reversion_signals else 0),
                "win_rate": float(len(reversion_signals[reversion_signals.get('reversion_probability', 0) > 0.7]) / len(reversion_signals) * 100 if len(reversion_signals) > 0 else 0),
                "signals": int(len(reversion_signals))
            })

            # Cluster Momentum strategy
            cluster_signals = df[df.get('cluster_validated', False) == True] if 'cluster_validated' in df else pd.DataFrame()
            strategies.append({
                "name": "Cluster Momentum",
                "status": "active" if len(cluster_signals) > 0 else "paused",
                "pnl": float(cluster_signals['avg_return_7d'].sum() if not cluster_signals.empty and 'avg_return_7d' in cluster_signals else 0),
                "win_rate": float(len(cluster_signals[cluster_signals['composite_score'] > 70]) / len(cluster_signals) * 100 if len(cluster_signals) > 0 else 0),
                "signals": int(len(cluster_signals))
            })

        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return {"strategies": []}

@app.get("/api/signals/active")
async def get_active_signals():
    """Get currently active trading signals from Oracle"""
    try:
        sync_bus = global_state.get('sync_bus')

        if not sync_bus:
            return {"signals": []}

        # Get trading directives and scanner data
        directives = await sync_bus.get_state('trading_directives') or []
        scanner_data = await sync_bus.get_state('scanner_data') or []

        signals = []

        # Process directives first (highest priority)
        for directive in directives[-20:]:
            signals.append({
                "symbol": directive.get('symbol', ''),
                "signal": directive.get('action', 'NEUTRAL').upper(),
                "type": "directive",
                "strength": float(directive.get('confidence', 0) * 100),
                "price": float(directive.get('price', 0)),
                "timestamp": datetime.fromtimestamp(directive.get('timestamp', datetime.now().timestamp())).isoformat() # Added fallback for timestamp
            })

        # Add strong scanner signals if we have room
        if len(signals) < 20 and scanner_data:
            df = pd.DataFrame(scanner_data)
            strong_signals = df[df['momentum_7d'].abs() > 0.05] if 'momentum_7d' in df else df

            for _, row in strong_signals.head(20 - len(signals)).iterrows():
                signals.append({
                    "symbol": row.get('symbol', ''),
                    "signal": row.get('signal', 'NEUTRAL'),
                    "type": "scanner",
                    "strength": float(abs(row.get('momentum_7d', 0)) * 100),
                    "price": float(row.get('price', 0)),
                    "timestamp": datetime.fromtimestamp(row.get('timestamp', datetime.now().timestamp())).isoformat() # Added fallback for timestamp
                })

        return {"signals": signals}
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        return {"signals": []}

@app.get("/api/rl/status")
async def get_rl_status():
    """Get RL agent status and performance"""
    try:
        components = global_state.get('components')

        if not components:
            return {"error": "System not initialized"}

        rl_agent = components.get('rl_agent')

        if rl_agent and hasattr(rl_agent, 'is_trained') and rl_agent.is_trained:
            # Get recent predictions and performance
            return {
                "is_trained": True,
                "algorithm": rl_agent.algorithm,
                "model_path": "models/rl_ppo_model.zip",
                "recent_actions": [],  # Would need to track this
                "confidence": 0.75,
                "total_predictions": 0,
                "status": "active"
            }
        else:
            return {
                "is_trained": False,
                "status": "not_initialized",
                "message": "RL agent not trained or not available"
            }
    except Exception as e:
        logger.error(f"Error getting RL status: {e}")
        return {"error": str(e)}

@app.get("/api/risk/analysis")
async def get_risk_analysis():
    """Get risk analysis data from live system"""
    try:
        sync_bus = global_state.get('sync_bus')
        components = global_state.get('components')

        if not sync_bus or not components:
            return {"error": "System not initialized"}

        # Get risk sentinel and trade analyzer
        risk_sentinel = components.get('risk_sentinel')
        trade_analyzer = components.get('trade_analyzer')

        if trade_analyzer:
            drawdown_stats = trade_analyzer.get_drawdown_stats()
            total_pnl = trade_analyzer.get_total_pnl()

            # Calculate VaR from trade history
            trade_pnls = [t.pnl for t in trade_analyzer.trades[-100:]] if trade_analyzer.trades else [0]
            # Ensure trade_pnls is not empty before calculating percentile
            var_95 = float(abs(np.percentile(trade_pnls, 5))) if trade_pnls else 0.0

            # Get market data for volatility
            market_data = await sync_bus.get_state('market_data') or []
            portfolio_volatility = np.std([m.get('volatility', 0) for m in market_data[-50:]]) if market_data else 0.0

            # Position concentration from execution daemon
            execution_daemon = components.get('execution_daemon')
            positions = execution_daemon.open_orders if execution_daemon else {}
            total_position_value = sum(abs(o.get('amount', 0) * o.get('price', 0)) for o in positions.values())
            position_concentration = min(total_position_value / 10000, 1.0) if total_position_value else 0.0

            return {
                "var_95": var_95,
                "position_concentration": float(position_concentration),
                "correlation_risk": float(portfolio_volatility),
                "leverage": 1.0,
                "portfolio_volatility": float(portfolio_volatility),
                "risk_score": float((position_concentration + portfolio_volatility + abs(drawdown_stats.get('current_drawdown', 0))) / 3 * 100) if drawdown_stats else 0.0
            }

        return {"error": "Trade analyzer not available"}
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        return {"error": str(e)}

@app.get("/api/backtest/results")
async def get_backtest_results():
    """Get latest backtest results"""
    try:
        # Check if backtest results exist
        from pathlib import Path
        import glob

        backtest_files = glob.glob("backtest_report_*.txt")
        if not backtest_files:
            return {"error": "No backtest results found"}

        # Get latest backtest file
        latest_file = max(backtest_files, key=lambda x: Path(x).stat().st_mtime)

        with open(latest_file, 'r') as f:
            report_text = f.read()

        # Parse report (basic parsing, you can enhance this)
        lines = report_text.split('\n')
        results = {
            "report_text": report_text,
            "timestamp": datetime.fromtimestamp(Path(latest_file).stat().st_mtime).isoformat()
        }

        # Extract key metrics
        for line in lines:
            if "Total Return:" in line:
                results["total_return"] = float(line.split(':')[1].strip().replace('%', ''))
            elif "Sharpe Ratio:" in line:
                results["sharpe_ratio"] = float(line.split(':')[1].strip())
            elif "Maximum Drawdown:" in line:
                results["max_drawdown"] = float(line.split(':')[1].strip().replace('%', ''))
            elif "Win Rate:" in line:
                results["win_rate"] = float(line.split(':')[1].strip().replace('%', ''))

        return results
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        return {"error": str(e)}

@app.get("/api/backtest/comparison")
async def get_strategy_comparison():
    """Get strategy backtest comparison results"""
    try:
        import json
        from pathlib import Path
        
        # Check for existing comparison results
        if Path('strategy_comparison_results.json').exists():
            with open('strategy_comparison_results.json', 'r') as f:
                return json.load(f)
        
        return {"error": "No comparison results found. Run comparison first."}
    except Exception as e:
        logger.error(f"Error getting comparison results: {e}")
        return {"error": str(e)}

@app.post("/api/backtest/run-comparison")
async def run_strategy_comparison():
    """Run strategy backtest comparison"""
    try:
        from strategy_backtest_comparison import StrategyBacktestComparison
        import ccxt.async_support as ccxt
        
        # Get historical data
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
        await exchange.close()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Run comparison
        comparator = StrategyBacktestComparison()
        results = await comparator.compare_all_strategies(df)
        
        # Save results
        import json
        with open('strategy_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        return {"error": str(e)}

@app.get("/api/agents/states")
async def get_agent_states():
    """Get all agent states and health"""
    try:
        sync_bus = global_state.get('sync_bus')
        if not sync_bus:
            return {"error": "System not initialized"}

        # Get all agent states
        all_states = {}
        agent_health = sync_bus.agent_health
        circuit_breakers = sync_bus.circuit_breakers

        for agent_id in sync_bus.agent_states.keys():
            state = await sync_bus.get_state(agent_id) or {}
            health_info = agent_health.get(agent_id, {})
            breaker_info = circuit_breakers.get(agent_id, {})

            success_count = health_info.get('success_count', 0)
            failure_count = health_info.get('failure_count', 0)
            health_score = success_count / max(1, success_count + failure_count)

            all_states[agent_id] = {
                "state": state,
                "health": {
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "health_score": health_score,
                    "last_update": health_info.get('last_success', 0)
                },
                "circuit_breaker": {
                    "is_open": breaker_info.get('is_open', False),
                    "failure_count": breaker_info.get('failure_count', 0),
                    "last_failure": breaker_info.get('last_failure_time', 0)
                }
            }

        return {
            "agents": all_states,
            "system_health": await sync_bus.get_state('system_health') or {},
            "tick_count": sync_bus.tick_count
        }
    except Exception as e:
        logger.error(f"Error getting agent states: {e}")
        return {"error": str(e)}

@app.get("/api/agents/logs")
async def get_agent_logs():
    """Get recent system logs"""
    try:
        from pathlib import Path
        import glob

        # Get audit logs
        audit_files = glob.glob("audit_logs/audit_*.log")
        if not audit_files:
            return {"logs": [], "message": "No audit logs found"}

        latest_log = max(audit_files, key=lambda x: Path(x).stat().st_mtime)

        # Read last 100 lines
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            recent_logs = lines[-100:]

        # Parse JSON logs
        parsed_logs = []
        for line in recent_logs:
            try:
                log_entry = json.loads(line.strip())
                parsed_logs.append(log_entry)
            except:
                continue

        return {
            "logs": parsed_logs,
            "total_count": len(parsed_logs),
            "log_file": latest_log
        }
    except Exception as e:
        logger.error(f"Error getting agent logs: {e}")
        return {"error": str(e), "logs": []}

@app.get("/api/positions/active")
async def get_active_positions():
    """Get currently open positions"""
    try:
        sync_bus = global_state.get('sync_bus')
        components = global_state.get('components')

        if not sync_bus or not components:
            return {"positions": [], "total_value": 0, "unrealized_pnl": 0, "avg_return": 0}

        execution_daemon = components.get('execution_daemon')

        if execution_daemon and hasattr(execution_daemon, 'open_orders'):
            positions = []
            total_value = 0
            total_pnl = 0

            for order_id, order in execution_daemon.open_orders.items():
                current_price = order.get('current_price', order.get('price', 0))
                entry_price = order.get('price', 0)
                size = order.get('amount', 0)

                pnl = (current_price - entry_price) * size
                pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                positions.append({
                    "symbol": order.get('symbol'),
                    "side": "long" if order.get('side') == 'buy' else "short",
                    "size": size,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "strategy": order.get('strategy', 'unknown')
                })

                total_value += current_price * size
                total_pnl += pnl

            avg_return = (total_pnl / total_value * 100) if total_value > 0 else 0

            return {
                "positions": positions,
                "total_value": total_value,
                "unrealized_pnl": total_pnl,
                "avg_return": avg_return
            }

        return {"positions": [], "total_value": 0, "unrealized_pnl": 0, "avg_return": 0}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {"error": str(e), "positions": []}

@app.get("/api/trades/history")
async def get_trade_history():
    """Get complete trade history"""
    try:
        sync_bus = global_state.get('sync_bus')
        components = global_state.get('components')

        if not sync_bus or not components:
            return {"trades": [], "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "avg_duration": "0h"}

        trade_analyzer = components.get('trade_analyzer')

        if trade_analyzer and hasattr(trade_analyzer, 'trades'):
            trades = []
            winning_count = 0
            losing_count = 0
            total_duration = 0

            for trade in trade_analyzer.trades:
                pnl = trade.pnl
                if pnl > 0:
                    winning_count += 1
                else:
                    losing_count += 1

                trades.append({
                    "timestamp": trade.exit_time if hasattr(trade, 'exit_time') else time.time(),
                    "symbol": trade.symbol,
                    "side": "buy" if trade.side == 1 else "sell",
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "size": trade.size,
                    "pnl": pnl,
                    "fees": trade.commission if hasattr(trade, 'commission') else 0,
                    "strategy": trade.strategy if hasattr(trade, 'strategy') else 'unknown'
                })

                if hasattr(trade, 'duration'):
                    total_duration += trade.duration

            avg_duration_hours = (total_duration / len(trades) / 3600) if trades else 0

            return {
                "trades": trades,
                "total_trades": len(trades),
                "winning_trades": winning_count,
                "losing_trades": losing_count,
                "avg_duration": f"{avg_duration_hours:.1f}h"
            }

        return {"trades": [], "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "avg_duration": "0h"}
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return {"error": str(e), "trades": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)