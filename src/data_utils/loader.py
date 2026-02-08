import pandas as pd
import numpy as np
import os
from typing import Optional, List
from src.utils.logger import setup_logger

class DataLoader:
    def __init__(self, data_path: str = "data/processed/gold_panel", output_dir: str = "reports/loading"):
        """
        Args:
            data_path: Path to the Parquet dataset (root folder).
            output_dir: Directory for logs.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.logger = setup_logger("ParquetLoader", log_dir=output_dir)

    def load_panel_data(
        self,
        feature_cols: Optional[List[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        target_col: str = 'target_ret_excess',
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Memory-efficient Parquet Loader v2.
        
        Capabilities:
        1. Column Pruning: Loads only requested features + identifiers.
        2. Predicate Pushdown: Filters years at the metadata level before reading files.
        3. Float32 Downcasting: Halves memory usage for floats.
        """
        
        # 1. Define Columns to Load
        # We ALWAYS need identifiers and the target
        base_cols = ['permno', date_col, target_col, 'year'] 
        
        columns = None
        if feature_cols:
            # Deduplicate just in case
            columns = list(set(base_cols + feature_cols))
            self.logger.info(f"REQ: Loading {len(columns)} columns...")
        else:
            self.logger.info("REQ: Loading ALL columns...")

        # 2. Define Year Filters (Predicate Pushdown)
        # This tells PyArrow to skip folders that don't match the year criteria
        filters = []
        if start_year:
            filters.append(('year', '>=', start_year))
        if end_year:
            filters.append(('year', '<=', end_year))
            
        # If no filters provided, list is empty (loads all years)
        if not filters:
            filters = None
            
        self.logger.info(f"SOURCE: {self.data_path}")
        if filters:
            self.logger.info(f"FILTERS: {filters}")

        try:
            # 3. Read Parquet
            # This is where the magic happens. PyArrow handles the optimization.
            df = pd.read_parquet(
                self.data_path,
                columns=columns,
                filters=filters,
                engine='pyarrow'
            )
            
            # 4. Post-Processing
            # Ensure date is datetime (Parquet usually preserves this, but good to be safe)
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])

            # 5. Float32 Downcasting
            # Crucial for avoiding OOM on 100+ features
            float_cols = df.select_dtypes(include=['float64']).columns
            if len(float_cols) > 0:
                df[float_cols] = df[float_cols].astype('float32')

            # 6. Sorting
            # Parquet partitions don't guarantee time order when concatenated.
            # Expanding Window CV requires sorted data.
            df = df.sort_values(by=[date_col, 'permno']).reset_index(drop=True)

            mem_usage = df.memory_usage(deep=True).sum() / 1e9
            self.logger.info(f"LOAD COMPLETE. Shape: {df.shape} | RAM: {mem_usage:.2f} GB")

            return df
            
        except Exception as e:
            self.logger.error(f"PARQUET LOAD FAILED: {e}")
            raise