import pandas as pd

def load_crsp(path) -> pd.DataFrame:
    """
    Load CRSP monthly data.
    Returns standardized columns: permno, date, ret, mktcap, prc
    """

    

def load_characteristics(path) -> pd.DataFrame:
    """
    Load GKX characteristics.
    Ensures DATE is month-end and permno is int.
    """