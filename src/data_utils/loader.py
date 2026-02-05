import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, List

from src.utils.logger import setup_logger

class DataLoader:
    def __init__(self, db_path, output_dir="reports/loading"):
        self.db_path = db_path
        self.output_dir = output_dir
        self.logger = setup_logger("DB_Loader", log_dir=output_dir)

    def load_panel_data(
        self,
        feature_cols : Optional[List[str]] = None,
        target_col: str = 'target_ret_excess',
        date_col: str = 'date_fmt'
    ):
        """
        Memory-efficient dataloader v1
        """

        conn = sqlite3.connect(self.db_path)
        base_cols = ['perm_no', date_col, target_col]

        if feature_cols:
            cols_to_load = base_cols + feature_cols
            query_cols = ", ".join(cols_to_load)
        else:
            query_cols = "*"        

        query = f"SELECT {query_cols} FROM gold_panel ORDER BY {date_col}"
        self.logger.info(f"STREAMING DATA, NUMBER OF COLUMNS: {len(query_cols)}...")

        try:
            df = pd.read_sql(query, conn)
            df['date'] = pd.to_datetime(df[date_col])
            df.drop(columns=[date_col], inplace=True)

            mem_usage = df.memory_usage(deep=True).sum() / 1e9
            self.logger.info(f"DATA LOADED SUCCESSFULLY. USING: {mem_usage:.2f} GB")

            return df
        except Exception as e:
            self.logger.error(f"DATA LOADED FAILED: {e}")
            raise
        finally:
            conn.close()
            self.logger.info("CONNECTION CLOSED")