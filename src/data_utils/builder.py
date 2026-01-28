import sqlite3
import pandas as pd

class GoldBuilder:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def build_modeling_panel(self):
        print("Building Gold Panel (Joining Returns)...")
        
        # We use SQLite's strftime to handle the 1-month offset.
        # Logic: A characteristic dated '2000-01-31' should predict the return dated '2000-02-29'.
        # We match on Year-Month strings to be robust against day-of-month differences (30th vs 31st).
        
        query = """
        CREATE TABLE gold_panel AS
        SELECT 
            s.*,
            c.ret_excess as target_ret_excess,
            c.mktcap as mktcap_next
        FROM silver_characteristics s
        INNER JOIN bronze_crsp c 
            ON s.permno = c.permno 
            -- Match char date + 1 month to return date
            AND strftime('%Y-%m', s.date_fmt, '+1 month') = strftime('%Y-%m', c.date)
        """
        
        try:
            self.conn.execute("DROP TABLE IF EXISTS gold_panel")
            self.conn.execute(query)
            
            # Create Index
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_gold_date ON gold_panel(date_fmt)")
            print("Gold panel built successfully.")
            
        except Exception as e:
            print(f"Gold Build Failed: {e}")

    def validate_alignment(self):
        """
        Simple check to ensure we didn't accidentally map t to t.
        """
        df = pd.read_sql("SELECT date_fmt, target_ret_excess FROM gold_panel LIMIT 5", self.conn)
        print("Sample of Gold Panel (Check dates manually):")
        print(df)