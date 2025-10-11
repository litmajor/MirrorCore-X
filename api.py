from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import List, Dict
from pydantic import BaseModel
import logging
from datetime import datetime, timezone
import asyncio
import json

app = FastAPI()
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

# Global system state
system_state = {
    'sync_bus': None,
    'components': None,
    'parallel_scanner': None,
    'oracle_imagination': None
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
        
        # Create main system
        sync_bus, components = await create_mirrorcore_system(
            dry_run=True,
            use_testnet=True,
            enable_oracle=True,
            enable_bayesian=True,
            enable_imagination=True
        )
        
        system_state['sync_bus'] = sync_bus
        system_state['components'] = components
        system_state['oracle_imagination'] = components.get('oracle_imagination')
        
        # Add parallel scanner
        parallel_scanner = await add_parallel_scanner_to_mirrorcore(
            sync_bus, components.get('scanner'), enable=True
        )
        system_state['parallel_scanner'] = parallel_scanner
        
        # Start background task
        asyncio.create_task(run_system_loop())
        
        logger.info("✅ System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")

async def run_system_loop():
    """Background task to run system ticks and broadcast updates"""
    while True:
        try:
            sync_bus = system_state.get('sync_bus')
            oracle_imagination = system_state.get('oracle_imagination')
            parallel_scanner = system_state.get('parallel_scanner')
            
            if sync_bus:
                # Run tick
                await sync_bus.tick()
                
                # Run enhanced cycle every 5 ticks
                tick_count = getattr(sync_bus, 'tick_count', 0)
                if tick_count % 5 == 0 and oracle_imagination:
                    results = await oracle_imagination.run_enhanced_cycle()
                    
                    # Broadcast to WebSocket clients
                    await broadcast_update({
                        'type': 'oracle_update',
                        'data': {
                            'directives': len(results.get('oracle_directives', [])),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    })
                
                # Parallel scan every 10 ticks
                if tick_count % 10 == 0 and parallel_scanner:
                    scan_results = await parallel_scanner.scan_and_update()
                    
                    await broadcast_update({
                        'type': 'scanner_update',
                        'data': {
                            'symbols': len(scan_results),
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
        sync_bus = system_state.get('sync_bus')
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
async def get_frames(timeframe: str):
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
async def get_market_overview():
    """Get market overview data from live system"""
    try:
        sync_bus = system_state.get('sync_bus')
        if not sync_bus:
            return {"error": "System not initialized"}
        
        # Get live scanner data
        scanner_data = await sync_bus.get_state('scanner_data') or []
        if not scanner_data:
            return {"error": "No scanner data available"}
        
        df = pd.DataFrame(scanner_data)
        
        # Calculate real market metrics
        total_volume = df['average_volume_usd'].sum() if 'average_volume_usd' in df else 0
        
        # Top movers by momentum
        top_movers = []
        if 'momentum_short' in df and 'symbol' in df:
            top_df = df.nlargest(5, 'momentum_short')[['symbol', 'momentum_short', 'price', 'composite_score']]
            top_movers = top_df.to_dict('records')
        
        # Market sentiment from signals
        bullish_count = len(df[df['signal'].isin(['Strong Buy', 'Buy', 'Weak Buy'])]) if 'signal' in df else 0
        total_count = len(df)


@app.get("/api/scanner/realtime")
async def get_realtime_scanner_data():
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


        market_sentiment = bullish_count / total_count if total_count > 0 else 0.5
        
        # Volatility from volume ratios
        volatility_index = df['volume_ratio'].std() if 'volume_ratio' in df else 0.5
        
        return {
            "total_volume": float(total_volume),
            "top_movers": top_movers,
            "market_sentiment": float(market_sentiment),
            "volatility_index": float(volatility_index),
            "total_symbols": int(total_count),
            "last_updated": latest_csv.split('_')[-2] + '_' + latest_csv.split('_')[-1].replace('.csv', '')
        }
    except Exception as e:
        logger.error(f"Error in market overview: {e}")
        return {"error": str(e)}

@app.get("/api/performance/summary")
async def get_performance_summary():
    """Get performance summary from backtest results"""
    try:
        # Try to load backtest results
        backtest_files = [f for f in os.listdir("C:/Users/PC/Documents/MirrorCore-X") if f.startswith("backtest_") and f.endswith(".csv")]
        
        if backtest_files:
            latest_backtest = max(backtest_files)
            df = pd.read_csv(os.path.join("C:/Users/PC/Documents/MirrorCore-X", latest_backtest))
            
            total_pnl = df['total_return'].sum() if 'total_return' in df else 0
            win_rate = df['win_rate'].mean() if 'win_rate' in df else 0
            sharpe_ratio = df['sharpe_ratio'].mean() if 'sharpe_ratio' in df else 0
            max_drawdown = df['max_drawdown'].max() if 'max_drawdown' in df else 0
            total_trades = len(df)
        else:
            # Fallback to scan results for estimates
            csv_files = [f for f in os.listdir("C:/Users/PC/Documents/MirrorCore-X") if f.startswith("scan_results_") and f.endswith(".csv")]
            if csv_files:
                df = pd.read_csv(os.path.join("C:/Users/PC/Documents/MirrorCore-X", max(csv_files)))
                total_pnl = df['avg_return_7d'].sum() if 'avg_return_7d' in df else 0
                win_rate = len(df[df['composite_score'] > 70]) / len(df) * 100 if len(df) > 0 else 0
                sharpe_ratio = 0
                max_drawdown = 0
                total_trades = len(df)
            else:
                raise Exception("No performance data available")
        
        return {
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "total_trades": int(total_trades)
        }
    except Exception as e:
        logger.error(f"Error in performance summary: {e}")
        return {"error": str(e)}

@app.get("/api/strategies")
async def get_strategies():
    """Get all active strategies from strategy trainer"""
    try:
        strategies = []
        
        # Load scan results to check which strategies are active
        csv_files = [f for f in os.listdir("C:/Users/PC/Documents/MirrorCore-X") if f.startswith("scan_results_") and f.endswith(".csv")]
        if csv_files:
            df = pd.read_csv(os.path.join("C:/Users/PC/Documents/MirrorCore-X", max(csv_files)))
            
            # Momentum Scanner strategy
            momentum_signals = df[df['momentum_short'] > 0.01] if 'momentum_short' in df else pd.DataFrame()
            strategies.append({
                "name": "Momentum Scanner",
                "status": "active",
                "pnl": float(momentum_signals['avg_return_7d'].sum() if not momentum_signals.empty and 'avg_return_7d' in momentum_signals else 0),
                "win_rate": float(len(momentum_signals[momentum_signals['composite_score'] > 70]) / len(momentum_signals) * 100 if len(momentum_signals) > 0 else 0),
                "signals": int(len(momentum_signals))
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
        sync_bus = system_state.get('sync_bus')
        oracle_imagination = system_state.get('oracle_imagination')
        
        if not sync_bus or not oracle_imagination:
            return {"signals": []}
        
        # Get Oracle directives
        directives = await sync_bus.get_state('oracle_directives') or []
        scanner_data = await sync_bus.get_state('scanner_data') or []
        
        df = pd.DataFrame(scanner_data) if scanner_data else pd.DataFrame()
        
        # Filter for strong signals
        strong_signals = df[
            (df['composite_score'] > 70) | 
            (df.get('cluster_validated', False) == True) |
            (df.get('reversion_probability', 0) > 0.7)
        ] if len(df) > 0 else pd.DataFrame()
        
        signals = []
        for _, row in strong_signals.head(20).iterrows():
            signals.append({
                "symbol": row.get('symbol', ''),
                "signal": row.get('signal', 'NEUTRAL'),
                "type": "momentum" if row.get('enhanced_momentum_score', 0) > 0.02 else "reversion" if row.get('reversion_candidate', False) else "neutral",
                "strength": float(row.get('composite_score', 0)),
                "price": float(row.get('price', 0)),
                "timestamp": datetime.now().isoformat()
            })
        
        return {"signals": signals}
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        return {"signals": []}

@app.get("/api/risk/analysis")
async def get_risk_analysis():
    """Get risk analysis data"""
    try:
        csv_files = [f for f in os.listdir("C:/Users/PC/Documents/MirrorCore-X") if f.startswith("scan_results_") and f.endswith(".csv")]
        if not csv_files:
            return {"error": "No data available"}
        
        df = pd.read_csv(os.path.join("C:/Users/PC/Documents/MirrorCore-X", max(csv_files)))
        
        # Calculate risk metrics
        portfolio_volatility = df['volume_ratio'].std() if 'volume_ratio' in df else 0
        
        # VaR calculation (simplified)
        returns = df['avg_return_7d'] / 100 if 'avg_return_7d' in df else pd.Series([0])
        var_95 = float(abs(returns.quantile(0.05)) * 10000) if len(returns) > 0 else 0
        
        # Position concentration
        top_position = df['average_volume_usd'].max() if 'average_volume_usd' in df else 0
        total_volume = df['average_volume_usd'].sum() if 'average_volume_usd' in df else 1
        position_concentration = float(top_position / total_volume) if total_volume > 0 else 0
        
        # Correlation risk (simplified based on volatility regime)
        high_vol_count = len(df[df.get('volatility_regime', 'normal') == 'high']) if 'volatility_regime' in df else 0
        correlation_risk = float(high_vol_count / len(df)) if len(df) > 0 else 0
        
        return {
            "var_95": var_95,
            "position_concentration": position_concentration,
            "correlation_risk": correlation_risk,
            "leverage": 1.0,
            "portfolio_volatility": float(portfolio_volatility),
            "risk_score": float((position_concentration + correlation_risk + portfolio_volatility) / 3 * 100)
        }
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)