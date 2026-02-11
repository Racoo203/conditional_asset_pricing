import pandas as pd
import numpy as np
import sqlite3
import os
# import tidyfinance as tf
from sqlalchemy import create_engine
from dotenv import load_dotenv

import io
import re
import zipfile
from curl_cffi import requests

GKX_DTYPES = {
    'permno': 'int64',
    'DATE': 'int64', 
}

class BronzeIngestor:
    def __init__(self, db_path: str, csv_path: str):
        self.db_path = db_path
        self.csv_path = csv_path

        self.conn = sqlite3.connect(db_path)
        load_dotenv()

    def load_ff3_factors(self):
        # As done with the tidyfinance module

        dataset = "F-F_Research_Data_Factors"
        base_url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        url = f"{base_url}{dataset}_CSV.zip"

        resp = requests.get(url)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            file_name = zf.namelist()[0]  # Ken French ZIPs contain one file
            raw_text = zf.read(file_name).decode("latin1")

        chunks = raw_text.split("\r\n\r\n")
        table_text = max(chunks, key=len)

        match = re.search(r"^\s*,", table_text, flags=re.M)
        start = match.start()
        csv_text = "Date" + table_text[start:]

        factors_ff_raw = pd.read_csv(io.StringIO(csv_text), index_col=0)

        s = factors_ff_raw.index.astype(str)

        if (s.str.len() == 8).all():  # daily: YYYYMMDD
            dt = pd.to_datetime(s, format="%Y%m%d")
        elif (s.str.len() == 6).all():  # monthly: YYYYMM
            dt = pd.to_datetime(s + "01", format="%Y%m%d")
        elif (s.str.len() == 4).all():  # annual: YYYY
            dt = pd.to_datetime(s + "0101", format="%Y%m%d")
            dt = dt.dt.to_period("A-DEC").dt.to_timestamp("end")
        else:
            raise ValueError("Unknown date format in Fama–French index.")

        factors_ff_raw = factors_ff_raw.set_index(dt)
        factors_ff_raw.index.name = "date"

        factors_ff_raw.to_parquet(
            path = 'data/raw/factors_ff3_monthly.parquet',
            index=False, 
            engine='pyarrow', 
            compression='snappy'
        )

    def load_crsp_api(self):
        """
        Downloads CRSP monthly data via TidyFinance/WRDS and stores in 'bronze_crsp'.
        """
        print(f"Starting CRSP download: {self.start_date} to {self.end_date}...")

        try:
            connection_string = (
                "postgresql+psycopg2://"
                f"{os.getenv('WRDS_USER')}:{os.getenv('WRDS_PASSWORD')}"
                "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
            )

            wrds = create_engine(connection_string, pool_pre_ping=True)
        except Exception as e:
            raise RuntimeError(f"CRSP Download failed. Check WRDS credentials. Error: {e}")
        try:
            crsp_monthly_query = f"""
                SELECT 
                    msf.permno,
                    date_trunc('month', msf.mthcaldt)::date AS date,
                    msf.mthret AS ret,
                    msf.shrout,
                    msf.mthprc AS altprc,
                    ssih.primaryexch,
                    ssih.siccd
                FROM crsp.msf_v2 AS msf
                INNER JOIN crsp.stksecurityinfohist AS ssih
                    ON msf.permno = ssih.permno
                    AND ssih.secinfostartdt <= msf.mthcaldt
                    AND msf.mthcaldt <= ssih.secinfoenddt
                WHERE msf.mthcaldt BETWEEN '{self.start_date}' AND '{self.end_date}'
            """

            crsp_monthly = pd.read_sql_query(
                sql=crsp_monthly_query,
                con=wrds,
                dtype={"permno": int, "siccd": int},
                parse_dates={"date"}).assign(shrout=lambda x: x["shrout"]*1000
            )

            print("Connection successfully made.")

            crsp_monthly = (crsp_monthly
                .assign(mktcap=lambda x: x["shrout"]*x["altprc"]/1000000)
                .assign(mktcap=lambda x: x["mktcap"].replace(0, np.nan))
            )

            mktcap_lag = (crsp_monthly
                .assign(
                    date=lambda x: x["date"]+pd.DateOffset(months=1),
                    mktcap_lag=lambda x: x["mktcap"]
                )
                .get(["permno", "date", "mktcap_lag"])
            )

            crsp_monthly = (crsp_monthly
                .merge(mktcap_lag, how="left", on=["permno", "date"])
            )

            def assign_exchange(primaryexch):
                if primaryexch == "N":
                    return "NYSE"
                elif primaryexch == "A":
                    return "AMEX"
                elif primaryexch == "Q":
                    return "NASDAQ"
                else:
                    return "Other"

            crsp_monthly["exchange"] = (crsp_monthly["primaryexch"]
                .apply(assign_exchange)
            )

            def assign_industry(siccd):
                if 1 <= siccd <= 999:
                    return "Agriculture"
                elif 1000 <= siccd <= 1499:
                    return "Mining"
                elif 1500 <= siccd <= 1799:
                    return "Construction"
                elif 2000 <= siccd <= 3999:
                    return "Manufacturing"
                elif 4000 <= siccd <= 4899:
                    return "Transportation"
                elif 4900 <= siccd <= 4999:
                    return "Utilities"
                elif 5000 <= siccd <= 5199:
                    return "Wholesale"
                elif 5200 <= siccd <= 5999:
                    return "Retail"
                elif 6000 <= siccd <= 6799:
                    return "Finance"
                elif 7000 <= siccd <= 8999:
                    return "Services"
                elif 9000 <= siccd <= 9999:
                    return "Public"
                else:
                    return "Missing"

            crsp_monthly["industry"] = (crsp_monthly["siccd"]
                .apply(assign_industry)
            )

            factors_ff3_monthly = (pd.read_parquet("data/raw/factors_ff3_monthly.parquet")
                .get(["date", "risk_free"])
            )
            
            crsp_monthly = (crsp_monthly
                .merge(factors_ff3_monthly, how="left", on="date")
                .assign(ret_excess=lambda x: x["ret"]-x["risk_free"])
                .assign(ret_excess=lambda x: x["ret_excess"].clip(lower=-1))
                .drop(columns=["risk_free"])
            )

            crsp_monthly = crsp_monthly.dropna(subset=["ret_excess", "mktcap"])
            
            crsp_monthly.to_sql("bronze_crsp", self.conn, if_exists="replace", index=False)
            print("CRSP data loaded into 'bronze_crsp'.")
            
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed. Error: {e}")

    def ingest_gkx_csv_chunked(self, chunksize: int = 200_000):
        """
        Reads the massive GKX CSV in chunks and dumps to 'bronze_characteristics'.
        """
        print(f"Ingesting {self.csv_path} in chunks...")
        
        first_chunk = True
        min_date = None
        max_date = None
        
        for i, chunk in enumerate(pd.read_csv(self.csv_path, chunksize=chunksize, dtype=GKX_DTYPES)):
            # Standardize column names to lowercase
            chunk.columns = chunk.columns.str.lower()

            chunk_dates = pd.to_datetime(chunk['date'], format='%Y%m%d')
            current_min = chunk_dates.min()
            current_max = chunk_dates.max()

            if min_date is None or current_min < min_date: min_date = current_min
            if max_date is None or current_max > max_date: max_date = current_max  
            
            if_exists = 'replace' if first_chunk else 'append'
            chunk.to_sql("bronze_characteristics", self.conn, if_exists=if_exists, index=False)
            
            first_chunk = False
            if i % 5 == 0:
                print(f"Processed chunk {i+1}...")
        
        self.start_date = min_date.strftime('%Y-%m-%d') if min_date else None
        self.end_date = max_date.strftime('%Y-%m-%d') if max_date else None
        
        print("GKX characteristics loaded into 'bronze_characteristics'.")    

    def run(self):
        self.load_ff3_factors()
        # self.ingest_gkx_csv_chunked()
        # self.load_crsp_api()