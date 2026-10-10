from typing import List, Tuple
import numpy as np
import pandas as pd
from quantmod.timeseries import Gap, HiLo, OpCl, dailyReturn, lead
from quantmod.indicators import BBands
import yfinance as yf
from IPython.display import display
from feature_engineering import FeatureEngineering
from utils import split_data
from ta.momentum import RSIIndicator, StochasticOscillator

class DataEngineering:

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

    def create_features(self) -> Tuple:

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
            lower_band, _, upper_band = BBands(self.data["close"], window, 2)
            self.data[f"BB{window}"] = ((self.data['close'] - lower_band) / (upper_band - lower_band))
            self.data[f"RSI{window}"] = RSIIndicator(close=self.data["close"],window=window).rsi()
            stoch = StochasticOscillator(
                                        high=self.data["high"],
                                        low=self.data["low"],
                                        close=self.data["close"],
                                        window=window,
                                        smooth_window=1
                                    )

            self.data[f"STOCH{window}"] = stoch.stoch()
        
        self.create_label()

        self.data.drop(self.required_columns, axis=1, inplace=True)
        self.data = self.data.dropna()

        self.data["Label"] = self.data["Label"].astype(int)

        self.X = self.data.drop(columns=["Label"], errors="ignore")
        self.y = self.data["Label"]
        self.y_train, _ = split_data(self.y, self.test_size)
        self.X_train, _ = split_data(self.X, self.test_size)

        return self.X, self.y

    def create_returns(self):
        return self.data["RET"]

    def create_label(self) -> None:
        # self.data["Label"] = np.where(lead(self.data["close"]) > 0.9950 * self.data["close"], 1, 0)
        vol = self.data["RET"].rolling(window=20).std()
        threshold = self.vol_scale * vol
        forward_log_return = np.log(lead(self.data["close"]) / self.data["close"])
        self.data["Label"] = (forward_log_return > threshold).where(forward_log_return.notna() & threshold.notna())

    def feature_engineering(self,
                            feature_selection_methods, 
                            vif_threshold, 
                            n_features):
        fe = FeatureEngineering(self.X_train,
                                self.y_train,
                                feature_selection_methods, 
                                vif_threshold, 
                                n_features)

        self.features = fe.main()
        return self.features

    def data_summary(self, data):
        summary = data.describe(include='all').T
        summary['dtype'] = data.dtypes
        summary['missing'] = data.isna().sum()
        summary['missing_pct'] = (data.isna().mean() * 100).round(2)
        summary['unique'] = data.nunique()

        display(summary)

    def summarize_engineered_data(self):
        data = pd.concat([self.X[self.features], self.y], axis=1)
        self.data_summary(data)

    def show_label_imbalance(self) -> None:
        """Print and return label counts and percentages for both data splits."""
        labels = pd.Index(pd.concat([self.y_train, self.y_test]).unique())
        rows = []
        for split_name, target in (
            ("in-sample", self.y_train),
            ("out-of-sample", self.y_test),
        ):
            counts = target.value_counts().reindex(labels, fill_value=0)
            for label, count in counts.items():
                rows.append(
                    {
                        "split": split_name,
                        "label": label,
                        "count": int(count),
                        "percentage": count / len(target) * 100,
                    }
                )

        report = pd.DataFrame(rows)
        print("Label distribution:")
        print(
            report.to_string(
                index=False,
                formatters={"percentage": "{:.2f}%".format},
            )
        )