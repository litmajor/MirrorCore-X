import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SignalEncoder:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """
        Accepts a dataframe with raw columns and returns a normalized feature matrix
        """
        # Ensure required columns exist
        required_cols = [
            "price", "volume", "fear", "btc_dominance",
            "alt_season", "psych", "return", "volatility"
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0  # Fallback if missing

        # Normalize all inputs between 0 and 1
        features = df[required_cols].fillna(0.0).values
        return self.scaler.fit_transform(features)

    def transform_live(self, latest_row: dict) -> np.ndarray:
        """
        Encodes a single observation (for live trading)
        """
        row = np.array([[latest_row.get(k, 0.0) for k in [
            "price", "volume", "fear", "btc_dominance",
            "alt_season", "psych", "return", "volatility"
        ]]])
        return self.scaler.transform(row)




def build_feature_df(price_df, fear_df, dominance_df, altcoin_df, psych_df):
    df = pd.DataFrame()
    df["price"] = price_df["close"]
    df["volume"] = price_df["volume"]
    df["fear"] = fear_df["value"]
    df["btc_dominance"] = dominance_df["value"]
    df["alt_season"] = altcoin_df["score"]
    df["psych"] = psych_df["level"]
    df["return"] = df["price"].pct_change().fillna(0)
    df["volatility"] = df["price"].rolling(window=5).std().fillna(0)
    return df.dropna()



# This module provides SignalEncoder and build_feature_df for use by other scripts.
# Import and call build_feature_df and SignalEncoder from your scanner or main pipeline, passing real DataFrames.

