from typing import List, Tuple
import numpy as np
import pandas as pd
from quantmod.timeseries import Gap, HiLo, OpCl, dailyReturn, lead
import yfinance as yf
from IPython.display import display

class DataHandler:

    def __init__(
        self,
        yahoo_symbol: str,
        start_date: str,
        test_size: float,
        required_columns: List,
        windows: List,
        vol_scale: float
    ) -> None:
        self.yahoo_symbol = yahoo_symbol
        self.start_date = start_date
        self.test_size = test_size
        self.required_columns = required_columns
        self.windows = windows
        self.vol_scale = vol_scale 

    def load_data(self) -> pd.DataFrame:

        print(f"loading historical data for: {self.yahoo_symbol}")
        downloaded = yf.download(
            tickers = self.yahoo_symbol,
            start=self.start_date,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )

        downloaded.columns = [str(column).lower() for column in downloaded.columns]

        self.data = downloaded.loc[:, self.required_columns].copy()
        self.data.index = pd.to_datetime(self.data.index)
        self.data.index.name = "date"

        self.clean_data()
        return self.data

    def clean_data(self):
        self.data = self.data[self.data["volume"] != 0]

    def create_features(self) -> Tuple[pd.DataFrame, pd.Series]:

        self.data["OC"] = OpCl(self.data)
        self.data["HC"] = HiLo(self.data)
        self.data["GAP"] = Gap(self.data)
        self.data["RET"] = dailyReturn(self.data["close"])

        for window in self.windows:
            self.data[f"PCHG{window}"] = self.data["close"].pct_change(window)
            self.data[f"VCHG{window}"] = self.data["volume"].pct_change(window)
            self.data[f"RET{window}"] = self.data[f"RET"].rolling(window).sum()
            self.data[f"OC{window}"] = self.data[f"OC"].rolling(window).mean()
            self.data[f"HC{window}"] = self.data[f"HC"].rolling(window).mean()
            self.data[f"GAP{window}"] = self.data[f"GAP"].rolling(window).mean()
            self.data[f"STD{window}"] = self.data[f"RET"].rolling(window).std()
        
        self.create_label()

        self.data.drop(self.required_columns, axis=1, inplace=True)
        self.data = self.data.dropna()

        self.data["Label"] = self.data["Label"].astype(int)
        
        self.X = self.data.drop(columns=["Label"], errors="ignore")
        self.y = self.data["Label"]

        return self.X, self.y

    def create_label(self) -> None:
        # self.data["Label"] = np.where(lead(self.data["close"]) > 0.9950 * self.data["close"], 1, 0)
        vol = self.data["RET"].rolling(window=20).std()
        threshold = self.vol_scale * vol
        forward_log_return = np.log(lead(self.data["close"]) / self.data["close"])
        self.data["Label"] = (forward_log_return > threshold).where(forward_log_return.notna() & threshold.notna())


    def data_summary(self):
        summary = self.data.describe(include='all').T
        summary['dtype'] = self.data.dtypes
        summary['missing'] = self.data.isna().sum()
        summary['missing_pct'] = (self.data.isna().mean() * 100).round(2)
        summary['unique'] = self.data.nunique()

        display(summary)