
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

def migrate_csv_to_postgres():
    """Migrate CSV data to PostgreSQL database"""
    
    # Get database connection from environment
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("DATABASE_URL not found. Please create a PostgreSQL database in Replit.")
        return
    
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    try:
        # Create tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_predictions (
                id SERIAL PRIMARY KEY,
                timestamp BIGINT NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                price DECIMAL(20, 8),
                volume DECIMAL(20, 8),
                volatility DECIMAL(10, 4),
                momentum_short DECIMAL(10, 6),
                rsi DECIMAL(10, 4),
                macd DECIMAL(10, 6),
                predicted_return DECIMAL(10, 6),
                confidence_score DECIMAL(10, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON market_predictions(symbol, timeframe);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON market_predictions(timestamp);
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id SERIAL PRIMARY KEY,
                timestamp BIGINT NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10),
                entry_price DECIMAL(20, 8),
                exit_price DECIMAL(20, 8),
                size DECIMAL(20, 8),
                pnl DECIMAL(20, 8),
                strategy VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_history(symbol);
        """)
        
        # Migrate CSV files
        csv_dir = "C:/Users/PC/Documents/MirrorCore-X"
        csv_files = [f for f in os.listdir(csv_dir) if f.startswith("predictions_") and f.endswith(".csv")]
        
        for csv_file in csv_files:
            print(f"Migrating {csv_file}...")
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            
            # Extract timeframe from filename
            timeframe = csv_file.split('_')[1]
            
            # Prepare data for insertion
            values = []
            for _, row in df.iterrows():
                values.append((
                    int(datetime.now().timestamp() * 1000),
                    row.get('symbol', 'BTCUSD'),
                    timeframe,
                    row.get('price'),
                    row.get('volume'),
                    row.get('volatility'),
                    row.get('momentum_short'),
                    row.get('rsi'),
                    row.get('macd'),
                    row.get(f'predicted_{timeframe}_return'),
                    row.get('confidence_score')
                ))
            
            # Batch insert
            execute_values(cur, """
                INSERT INTO market_predictions 
                (timestamp, symbol, timeframe, price, volume, volatility, 
                 momentum_short, rsi, macd, predicted_return, confidence_score)
                VALUES %s
            """, values)
            
            print(f"Migrated {len(values)} records from {csv_file}")
        
        conn.commit()
        print("Migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"Migration error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    migrate_csv_to_postgres()
