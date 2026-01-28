import pandas as pd
import sqlite3
import os
import tidyfinance as tf
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Strict types prevent pandas from misinterpreting IDs as floats
GKX_DTYPES = {
    'permno': 'int64',
    'DATE': 'int64', 
    # Add 'siccd': 'float64' if it contains NaNs, otherwise 'int64'
}

class BronzeIngestor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        load_dotenv()

    def load_crsp_api(self, start_date: str, end_date: str):
        """
        Downloads CRSP monthly data via TidyFinance/WRDS and stores in 'bronze_crsp'.
        """
        print(f"Starting CRSP download: {start_date} to {end_date}...")
        
        connection_string = (
            "postgresql+psycopg2://"
            f"{os.getenv('WRDS_USER')}:{os.getenv('WRDS_PASSWORD')}"
            "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
        )
        
        wrds = create_engine(connection_string, pool_pre_ping=True)

        try:
            crsp_monthly = tf.download_data(
                domain="wrds",
                dataset="crsp_monthly",
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            raise RuntimeError(f"CRSP Download failed. Check WRDS credentials. Error: {e}")

        # Basic cleanup: Drop rows where we have no return target
        crsp_monthly = crsp_monthly.dropna(subset=["ret_excess", "mktcap"])
        
        crsp_monthly.to_sql("bronze_crsp", self.conn, if_exists="replace", index=False)
        print("CRSP data loaded into 'bronze_crsp'.")

    def ingest_gkx_csv_chunked(self, csv_path: str, chunksize: int = 100_000):
        """
        Reads the massive GKX CSV in chunks and dumps to 'bronze_characteristics'.
        """
        print(f"Ingesting {csv_path} in chunks...")
        
        first_chunk = True
        
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, dtype=GKX_DTYPES)):
            # Standardize column names to lowercase
            chunk.columns = chunk.columns.str.lower()
            
            if_exists = 'replace' if first_chunk else 'append'
            chunk.to_sql("bronze_characteristics", self.conn, if_exists=if_exists, index=False)
            
            first_chunk = False
            if i % 5 == 0:
                print(f"Processed chunk {i+1}...")
        
        print("GKX characteristics loaded into 'bronze_characteristics'.")

from pathlib import Path

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    
    db_path = current_dir.parent.parent / "data" / "raw" / "data.sqlite"
    csv_path = current_dir.parent.parent / "data" / "raw" / "characteristics_raw.csv"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    ingestor = BronzeIngestor(str(db_path))
    ingestor.load_crsp_api('2000-01-01', '2024-12-31')    
    ingestor.ingest_gkx_csv_chunked(str(csv_path))