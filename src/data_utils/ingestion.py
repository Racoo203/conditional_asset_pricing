import pandas as pd
import sqlite3
import os
import tidyfinance as tf
from sqlalchemy import create_engine
from dotenv import load_dotenv

GKX_DTYPES = {
    'permno': 'int64',
    'DATE': 'int64', 
}

class BronzeIngestor:
    def __init__(self, db_path: str, csv_path: str):
        self.db_path = db_path
        self.csv_path = csv_path

        self.conn = sqlite3.connect(db_path)
        load_dotenv()

    def load_crsp_api(self):
        """
        Downloads CRSP monthly data via TidyFinance/WRDS and stores in 'bronze_crsp'.
        """
        print(f"Starting CRSP download: {self.start_date} to {self.end_date}...")
    
        try:
            crsp_monthly = tf.download_data(
                domain="wrds",
                dataset="crsp_monthly",
                start_date=self.start_date,
                end_date=self.end_date
            )
        except Exception as e:
            raise RuntimeError(f"CRSP Download failed. Check WRDS credentials. Error: {e}")

        # Basic cleanup: Drop rows where we have no return target
        crsp_monthly = crsp_monthly.dropna(subset=["ret_excess", "mktcap"])
        
        crsp_monthly.to_sql("bronze_crsp", self.conn, if_exists="replace", index=False)
        print("CRSP data loaded into 'bronze_crsp'.")

    def ingest_gkx_csv_chunked(self, chunksize: int = 100_000):
        """
        Reads the massive GKX CSV in chunks and dumps to 'bronze_characteristics'.
        """
        print(f"Ingesting {self.csv_path} in chunks...")
        
        first_chunk = True
        min_date = None
        max_date = None
        
        for i, chunk in enumerate(pd.read_csv(self.csv_path, chunksize=chunksize, dtype=GKX_DTYPES)):
            # Standardize column names to lowercase
            chunk.columns = chunk.columns.str.lower()

            chunk_dates = pd.to_datetime(chunk['date'], format='%Y%m%d')
            current_min = chunk_dates.min()
            current_max = chunk_dates.max()

            if min_date is None or current_min < min_date: min_date = current_min
            if max_date is None or current_max > max_date: max_date = current_max  
            
            if_exists = 'replace' if first_chunk else 'append'
            chunk.to_sql("bronze_characteristics", self.conn, if_exists=if_exists, index=False)
            
            first_chunk = False
            if i % 5 == 0:
                print(f"Processed chunk {i+1}...")
        
        self.start_date = min_date.strftime('%Y-%m-%d') if min_date else None
        self.end_date = max_date.strftime('%Y-%m-%d') if max_date else None
        
        print("GKX characteristics loaded into 'bronze_characteristics'.")    

    def run(self):
        self.ingest_gkx_csv_chunked()
        self.load_crsp_api()