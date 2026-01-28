import sqlite3
import pandas as pd

class GoldBuilder:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def build_modeling_panel(self):
        print("Building Gold Panel (Joining Returns)...")
        
        try:
            # 1. Prepare the Silver table with a pre-calculated join key
            # This avoids running strftime() millions of times during the JOIN
            self.conn.execute("DROP TABLE IF EXISTS silver_temp")
            self.conn.execute("""
                CREATE TABLE silver_temp AS 
                SELECT *, 
                       strftime('%Y-%m', date_fmt, '+1 month') as join_month
                FROM silver_characteristics
            """)
            
            # 2. Create high-performance indexes
            # We index (permno, join_month) to make the JOIN an O(log n) operation
            print("Creating temporary indexes...")
            self.conn.execute("CREATE INDEX idx_s_temp ON silver_temp(permno, join_month)")
            
            # Ensure the bronze table has a similar index for the lookup
            # We use an expression-based index here to match the join logic
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_c_lookup 
                ON bronze_crsp(permno, strftime('%Y-%m', date))
            """)

            # 3. Execute the Join
            self.conn.execute("DROP TABLE IF EXISTS gold_panel")
            query = """
            CREATE TABLE gold_panel AS
            SELECT 
                s.*,
                c.ret_excess as target_ret_excess,
                c.mktcap as mktcap_next
            FROM silver_temp s
            INNER JOIN bronze_crsp c 
                ON s.permno = c.permno 
                AND s.join_month = strftime('%Y-%m', c.date)
            """
            
            self.conn.execute(query)
            
            # Clean up and final index
            self.conn.execute("DROP TABLE silver_temp")
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

    def run(self):
        self.build_modeling_panel()
        self.validate_alignment()