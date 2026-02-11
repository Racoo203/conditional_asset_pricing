import sqlite3
import pandas as pd
import os
import shutil
import glob
from tqdm import tqdm
from collections import defaultdict
from src.utils.logger import setup_logger

logger = setup_logger("GoldBuilder", "reports/logs")

class GoldBuilder:
    def __init__(self, db_path: str, silver_path: str = "data/processed/silver", output_path: str = "data/processed/gold_panel"):
        self.db_path = db_path
        self.silver_path = silver_path
        self.output_path = output_path
        self.conn = sqlite3.connect(db_path)

    def _load_returns_lookup(self):
        """Loads the returns into a memory lookup table."""
        logger.info("Loading Returns for Lookup...")
        query = "SELECT permno, date, ret_excess, mktcap FROM bronze_crsp"
        df_ret = pd.read_sql(query, self.conn)
        
        df_ret['date'] = pd.to_datetime(df_ret['date'])
        # Create Join Key: Char(Jan) joins with Ret(Feb)
        df_ret['join_key'] = df_ret['date'].dt.to_period('M')
        
        return df_ret

    def _group_files_by_year(self):
        """Scans the Silver directory and groups parquet files by Year."""
        pattern = os.path.join(self.silver_path, "date=*", "*.parquet")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No Silver Parquet files found in {self.silver_path}")
            
        files_by_year = defaultdict(list)
        
        for f in files:
            try:
                # Robustly extract year from "date=YYYYMMDD"
                # Split path by 'date=' and take the first 4 chars of the next segment
                part_segment = f.split("date=")[1]
                year = part_segment[:4]
                files_by_year[year].append(f)
            except IndexError:
                continue
                
        return files_by_year

    def build_parquet(self):
        logger.info(f"BUILDING GOLD PANEL AT {self.output_path}...")
        
        # 1. Load Returns
        df_ret = self._load_returns_lookup()
        
        # 2. Map Files
        files_by_year = self._group_files_by_year()
        sorted_years = sorted(files_by_year.keys())
        
        # 3. Clean output dir
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        # 4. Stream & Join
        for year in tqdm(sorted_years, desc="Processing Years"):
            year_files = files_by_year[year]
            
            daily_dfs = []
            for f in year_files:
                daily_dfs.append(pd.read_parquet(f))
            
            if not daily_dfs:
                continue
            
            df_char = pd.concat(daily_dfs, ignore_index=True)
            
            # Ensure Date Format
            if not pd.api.types.is_datetime64_any_dtype(df_char['date']):
                df_char['date'] = pd.to_datetime(df_char['date'], format='%Y%m%d')
            
            # Create Join Key: The NEXT month
            df_char['join_key'] = df_char['date'].dt.to_period('M') + 1
            
            # Merge (Inner Join = Strict Alignment)
            df_merged = pd.merge(
                df_char,
                df_ret[['permno', 'join_key', 'ret_excess', 'mktcap']],
                on=['permno', 'join_key'],
                how='inner',
                suffixes=('', '_next')
            )
            
            if df_merged.empty:
                continue

            # Rename & Clean
            df_merged.rename(columns={'ret_excess': 'target_ret_excess', 'mktcap': 'mktcap_next'}, inplace=True)
            df_merged.drop(columns=['join_key'], inplace=True)
            
            # Add Partition Column for Gold
            df_merged['year'] = int(year)
            
            # Save Partition (Hive Style: year=YYYY/part.parquet)
            df_merged.to_parquet(
                self.output_path, 
                partition_cols=['year'], 
                index=False, 
                engine='pyarrow', 
                compression='snappy'
            )
            
        logger.info("GOLD PANEL PARQUET BUILD COMPLETE")

    def run(self):
        self.build_parquet()