from typing import List, Tuple
import numpy as np
import pandas as pd
from quantmod.datasets import fetch_historical_data
from quantmod.timeseries import Gap, HiLo, OpCl, dailyReturn, lead

class DataHandler:

    def __init__(
        self,
        symbol: str,
        start_date: str,
        test_size: float,
        excluded_columns: List,
        windows: List
    ) -> None:
        self.symbol = symbol
        self.start_date = start_date
        self.test_size = test_size
        self.excluded_columns = excluded_columns
        self.windows = windows

    def load_data(self) -> pd.DataFrame:

        print(f"loading historical data for: {self.symbol}")
        (
            fetch_historical_data(self.symbol)
            .assign(date=lambda frame: pd.to_datetime(frame["date"]))
            .set_index("date")
            .loc[self.start_date:]
            .copy()
        )

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

        self.data.drop(self.excluded_columns, axis=1, inplace=True)
        self.data = self.data.dropna()

        self.X = self.data.drop(columns=["Label"], errors="ignore")
        self.y = self.data["Label"]

        return self.X, self.y

    def create_label(self) -> None:
        self.data["Label"] = np.where(lead(self.data["close"]) > 0.9950 * self.data["close"], 1, 0)