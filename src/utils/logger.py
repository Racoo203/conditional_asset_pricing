import logging
import os
import sys

def setup_logger(name: str, log_dir: str = "reports/logs", console: bool = True):
    """
    Configures a logger that writes to a file and optionally to the console.
    
    Args:
        name (str): The name of the logger (e.g., 'DB_Inspector').
        log_dir (str): Directory where log files will be saved.
        console (bool): Whether to print logs to stdout.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 1. File Handler (Always on)
    log_file = os.path.join(log_dir, f"{name.lower()}.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 2. Console Handler (Optional)
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger