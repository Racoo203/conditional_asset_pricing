import sqlite3
import os
import logging
from src.utils.logger import setup_logger

logger = setup_logger("DB_Cleanup", "reports/logs")

def vacuum_database(db_path: str, drop_bronze_char: bool = False):
    """
    Optimizes the SQLite database.
    
    Args:
        db_path: Path to sqlite file.
        drop_bronze_char: If True, drops the massive bronze_characteristics table.
                          Only do this if Silver/Gold pipelines are confirmed working!
    """
    if not os.path.exists(db_path):
        logger.error("Database not found.")
        return

    conn = sqlite3.connect(db_path)
    
    # List of tables to strictly REMOVE from SQLite (since they are now Parquet or temp)
    # We DO NOT drop 'bronze_crsp' as that is our source of truth for Returns.
    tables_to_drop = ['silver_characteristics', 'gold_panel', 'silver_temp']
    
    if drop_bronze_char:
        tables_to_drop.append('bronze_characteristics')
    
    logger.info(f"Cleaning database: {db_path}")
    
    for table in tables_to_drop:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"Dropped table: {table}")
        except sqlite3.Error as e:
            logger.warning(f"Could not drop {table}: {e}")
            
    # VACUUM is essential to actually reclaim the disk space
    logger.info("Running VACUUM (This re-writes the DB file)...")
    try:
        conn.execute("VACUUM")
        logger.info("Database optimized. Size should be significantly reduced.")
    except sqlite3.Error as e:
        logger.error(f"Vacuum failed: {e}")
        
    conn.close()