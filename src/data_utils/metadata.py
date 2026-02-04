import sqlite3
import pandas as pd
import os
import json
from datetime import datetime
from src.utils.logger import setup_logger # <--- Import shared logger

class DatabaseInspector:
    def __init__(self, db_path, output_dir="reports/metadata"):
        self.db_path = db_path
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup Logging via Shared Utility
        self.logger = setup_logger("DB_Inspector", log_dir=output_dir)
        
        # Connect
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self.logger.info(f"CONNECTED TO DB at {db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"FAILED TO CONNECT: {e}")
            raise

    def get_tables(self):
        """Returns a list of all tables in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        self.cursor.execute(query)
        tables = [row[0] for row in self.cursor.fetchall()]
        self.logger.info(f"FOUND TABLES: {tables}")
        return tables

    def get_table_schema(self, table_name):
        """Returns column names and types for a specific table."""
        query = f"PRAGMA table_info({table_name})"
        try:
            # PRAGMA returns: (cid, name, type, notnull, dflt_value, pk)
            df = pd.read_sql(query, self.conn)
            # Simplify for report
            schema = df[['name', 'type', 'notnull']].to_dict(orient='records')
            return schema
        except Exception as e:
            self.logger.error(f"COULD NOT GET SCHEMA FOR {table_name}: {e}")
            return []

    def get_table_stats(self, table_name):
        """Get row count and date range (if applicable)."""
        stats = {}
        try:
            # Row Count
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            stats['row_count'] = self.cursor.execute(count_query).fetchone()[0]
            
            # Date Range (Heuristic check for common date columns)
            schema = self.get_table_schema(table_name)
            date_col = next((c['name'] for c in schema if 'date' in c['name'].lower()), None)
            
            if date_col:
                min_max_query = f"SELECT MIN({date_col}), MAX({date_col}) FROM {table_name}"
                min_val, max_val = self.cursor.execute(min_max_query).fetchone()
                stats['date_col'] = date_col
                stats['min_date'] = min_val
                stats['max_date'] = max_val
            else:
                stats['date_info'] = "No obvious date column found"

            self.logger.info(f"ANALYZED {table_name}: {stats['row_count']} rows.")
            return stats
        except Exception as e:
            self.logger.error(f"STATS FAILED FOR {table_name}: {e}")
            return {}

    def generate_full_report(self):
        """Runs full inspection and saves to JSON and Markdown."""
        self.logger.info("STARTING FULL METADATA GENERATION...")
        
        tables = self.get_tables()
        full_report = {}
        
        for table in tables:
            full_report[table] = {
                "schema": self.get_table_schema(table),
                "stats": self.get_table_stats(table)
            }
            
        # Save as JSON (Machine Readable)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"db_metadata_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(full_report, f, indent=4)
        self.logger.info(f"SAVED JSON REPORT: {json_path}")
        
        # Save as Markdown (Human Readable)
        md_path = os.path.join(self.output_dir, f"db_metadata_{timestamp}.md")
        self._save_markdown(md_path, full_report)
        self.logger.info(f"SAVED MARKDOWN REPORT: {md_path}")
        
    def _save_markdown(self, path, report):
        """Helper to format the report as a readable Markdown file."""
        with open(path, 'w') as f:
            f.write(f"# Database Metadata Report\nGenerated: {datetime.now()}\n\n")
            for table, data in report.items():
                f.write(f"## Table: `{table}`\n")
                f.write(f"- **Rows**: {data['stats'].get('row_count', 'N/A')}\n")
                if 'min_date' in data['stats']:
                    f.write(f"- **Range**: {data['stats']['min_date']} to {data['stats']['max_date']}\n")
                
                f.write("\n### Columns\n")
                f.write("| Name | Type | Not Null |\n|---|---|---|\n")
                for col in data['schema']:
                    f.write(f"| {col['name']} | {col['type']} | {col['notnull']} |\n")
                f.write("\n---\n")

    def close(self):
        self.conn.close()
        self.logger.info("CONNECTION CLOSED")