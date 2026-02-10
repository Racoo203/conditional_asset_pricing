import pandas as pd
import sqlite3
import os
import shutil
from joblib import Parallel, delayed
from src.utils.logger import setup_logger

logger = setup_logger("SilverProcessor")

def _process_month(date_val, db_path, output_dir):
    """
    Worker function:
    1. Connects to DB (Must create new connection per thread/process)
    2. Reads ONE month
    3. Ranks & Scales to [-1, 1]
    4. Writes to Parquet partition
    """
    try:
        # 1. Connect (Read-only mode is safer)
        # URI mode requires Python 3.4+ and SQLite 3.7.7+
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        
        # 2. Read Data
        query = f"SELECT * FROM bronze_characteristics WHERE date = {date_val}"
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return f"Skipped {date_val} (Empty)"

        # 3. Identify Features (Exclude Metadata)
        # We assume metadata cols are known. Adapt list as needed.
        meta_cols = ['permno', 'date', 'siccd', 'ticker', 'ret', 'shrout']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        # 4. GKX Transformation: Rank -> [-1, 1]
        # Formula: (Rank / (N + 1)) * 2 - 1
        ranks = df[feature_cols].rank(method='average', na_option='keep')
        counts = df[feature_cols].count()
        
        # Apply formula
        df[feature_cols] = (ranks.div(counts + 1)) * 2 - 1
        
        # Impute missing with 0.0 (Median)
        df[feature_cols] = df[feature_cols].fillna(0.0)
        
        # 5. Write to Parquet Partition
        # Structure: data/processed/silver/date=19800131/part.parquet
        partition_dir = os.path.join(output_dir, f"date={date_val}")
        os.makedirs(partition_dir, exist_ok=True)
        
        output_file = os.path.join(partition_dir, "part.parquet")
        df.to_parquet(output_file, index=False, compression='snappy')
        
        return None # Success
        
    except Exception as e:
        return f"Error on {date_val}: {str(e)}"

class SilverProcessor:
    def __init__(self, db_path: str, output_path: str = "data/processed/silver"):
        self.db_path = db_path
        self.output_path = output_path
        
    def run_parallel(self, n_jobs=-1):
        """
        Main entry point.
        n_jobs=-1 uses all available CPU cores.
        """
        logger.info(f"Starting Parallel Processing (Jobs: {n_jobs})...")
        
        # 1. Get List of Dates
        conn = sqlite3.connect(self.db_path)
        dates = pd.read_sql("SELECT DISTINCT date FROM bronze_characteristics ORDER BY date", conn)['date'].tolist()
        conn.close()
        
        logger.info(f"Found {len(dates)} months to process.")
        
        # 2. Prepare Output Directory
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)
        
        # 3. Execute Parallel Loop
        # We use joblib's Parallel/delayed syntax
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_process_month)(d, self.db_path, self.output_path) 
            for d in dates
        )
        
        # 4. Check for Errors
        errors = [r for r in results if r is not None]
        if errors:
            logger.error(f"Encountered {len(errors)} errors:")
            for e in errors[:5]: logger.error(e)
        else:
            logger.info("Silver Processing Complete (Parquet).")