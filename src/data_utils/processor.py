import pandas as pd
import sqlite3
import numpy as np

class SilverProcessor:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def _rank_standardize(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Applies GKX transformation:
        1. Rank values cross-sectionally.
        2. Normalize to [-0.5, 0.5].
        3. Fill missing with 0 (the median).
        """
        # 1. Rank (handle ties by averaging)
        ranks = df[feature_cols].rank(method='average', na_option='keep')
        
        # 2. Normalize: (rank / (count + 1)) - 0.5
        # We use count() per column to handle varying missingness
        counts = df[feature_cols].count()
        standardized = (ranks.div(counts + 1)) - 0.5
        
        # 3. Impute missing with 0.0
        return standardized.fillna(0.0)

    def process_characteristics(self):
        """
        Iterates through dates in Bronze, processes, saves to Silver.
        """
        print("Starting Silver processing (Rank Standardization)...")
        
        # Get unique dates to iterate over
        dates = pd.read_sql("SELECT DISTINCT date FROM bronze_characteristics ORDER BY date", self.conn)['date'].tolist()
        
        for i, d in enumerate(dates):
            # Load only one month into memory
            query = f"SELECT * FROM bronze_characteristics WHERE date = {d}"
            df_month = pd.read_sql(query, self.conn)
            
            # Identify feature columns (exclude IDs)
            id_cols = ['permno', 'date', 'siccd', 'ticker'] 
            feature_cols = [c for c in df_month.columns if c not in id_cols]
            
            # Transform features
            df_month[feature_cols] = self._rank_standardize(df_month, feature_cols)
            
            # Create a standard SQL date string (YYYY-MM-DD) for easier joining later
            # Assuming 'date' is YYYYMMDD int
            df_month['date_fmt'] = pd.to_datetime(df_month['date'], format='%Y%m%d')
            
            # Save to Silver
            if_exists = 'replace' if i == 0 else 'append'
            df_month.to_sql("silver_characteristics", self.conn, if_exists=if_exists, index=False)
            
            if i % 50 == 0: 
                print(f"Processed date {d} ({i}/{len(dates)})")

        # Create indices for fast joining in Gold step
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_silver_date ON silver_characteristics(date_fmt)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_silver_permno ON silver_characteristics(permno)")
        print("Silver processing complete.")

    def sanity_check(self):
        """
        Validates the data integrity.
        """
        print("Running diagnostics...")
        
        # Check 1: Range [-0.5, 0.5]
        # We check just one characteristic (e.g., mvel1) to save time
        stats = pd.read_sql("SELECT MIN(mvel1) as min_val, MAX(mvel1) as max_val FROM silver_characteristics", self.conn)
        min_val, max_val = stats.iloc[0]['min_val'], stats.iloc[0]['max_val']
        
        if min_val < -0.51 or max_val > 0.51:
            print(f"FAILURE: Standardization out of bounds. Range is [{min_val}, {max_val}]")
        else:
            print(f"Range check passed: [{min_val:.4f}, {max_val:.4f}]")

        # Check 2: History Length
        # Warn if too many stocks have short histories (IPO noise)
        counts = pd.read_sql("SELECT permno, COUNT(*) as cnt FROM silver_characteristics GROUP BY permno", self.conn)
        short_history = counts[counts['cnt'] < 12]
        print(f"Info: {len(short_history)} stocks have < 12 months of data (out of {len(counts)} total).")