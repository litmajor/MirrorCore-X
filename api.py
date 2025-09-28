
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import List, Dict
from pydantic import BaseModel
import logging
from datetime import datetime

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
