import pandas as pd

def temporal_split(df, train_end, val_end):
    """
    Splits panel into train/val/test based on DATE.
    """
