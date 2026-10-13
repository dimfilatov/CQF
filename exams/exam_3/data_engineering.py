from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from quantmod.timeseries import Gap, HiLo, OpCl, dailyReturn, lead
from quantmod.indicators import BBands
import yfinance as yf
from IPython.display import display
from feature_engineering import FeatureEngineering
from utils import split_data
from ta.momentum import RSIIndicator, StochasticOscillator
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

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
        self.stationarity_alpha = 0.05

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

        self.downloaded_row_count = len(downloaded)
        self.data = downloaded.loc[:, self.required_columns].copy()
        self.data.index = pd.to_datetime(self.data.index)
        self.data.index.name = "date"
        self.original_feature_columns = list(self.data.columns)

        self.clean_data()
        return self.data

    def clean_data(self):
        self.rows_removed_for_zero_volume = int((self.data["volume"] == 0).sum())
        self.data = self.data[self.data["volume"] != 0]

    def create_features(self) -> Tuple:

        self.data["GAP"] = Gap(self.data)
        self.data["RET"] = dailyReturn(self.data["close"])

        for window in self.windows:
            self.data[f"PCHG{window}"] = self.data["close"].pct_change(window)
            self.data[f"VCHG{window}"] = self.data["volume"].pct_change(window)
            self.data[f"RET{window}"] = self.data[f"RET"].rolling(window).sum()
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
        self.engineered_feature_columns = [
            column for column in self.data.columns if column != "Label"
        ]

        self.missing_values_before_drop = self.data.isna().sum()
        self.rows_before_missing_value_drop = len(self.data)
        self.data = self.data.dropna()
        self.rows_removed_for_missing_values = (
            self.rows_before_missing_value_drop - len(self.data)
        )

        self.data["Label"] = self.data["Label"].astype(int)

        self.X = self.data.drop(columns=["Label"], errors="ignore")
        self.y = self.data["Label"]
        self.y_train, self.y_test = split_data(self.y, self.test_size)
        self.X_train, _ = split_data(self.X, self.test_size)

    def create_returns(self):
        return self.forward_log_return

    def create_label(self) -> None:
        vol = self.data["RET"].rolling(window=20).std()
        threshold = self.vol_scale * vol
        self.forward_log_return = np.log(lead(self.data["close"]) / self.data["close"])
        self.data["Label"] = (self.forward_log_return > threshold).where(self.forward_log_return.notna() & threshold.notna())

    def feature_engineering(self,
                            feature_selection_methods, 
                            corr_threshold, 
                            n_features):
        self.feature_selection_methods = list(feature_selection_methods)
        fe = FeatureEngineering(self.X_train,
                                self.y_train,
                                self.feature_selection_methods,
                                corr_threshold, 
                                n_features)

        self.features, self.corr, self.feature_df = fe.main()

        display(self.feature_df)
        display(self.corr)
        display(self.features)

    def data_summary(self, data):
        """Summarize columns and flag numeric series passing both ADF and KPSS at the 5% level."""
        summary = data.describe(include='all').T
        summary['dtype'] = data.dtypes
        summary['missing'] = data.isna().sum()
        summary['missing_pct'] = (data.isna().mean() * 100).round(2)
        summary['unique'] = data.nunique()

        adf_statistics = {}
        adf_pvalues = {}
        kpss_statistics = {}
        kpss_pvalues = {}
        for column in data.columns:
            if column == "Label":
                continue
            values = data[column]
            series = values.dropna().astype(float)
            result = adfuller(series, maxlag = 1, autolag=None, result_object=True)
            adf_statistics[column] = result.statistic
            adf_pvalues[column] = result.pvalue
            # KPSS p-values are interpolated and clipped to [0.01, 0.1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                kpss_result = kpss(series, regression="c", nlags=1, result_object=True)
            kpss_statistics[column] = kpss_result.statistic
            kpss_pvalues[column] = kpss_result.pvalue

        summary["ADF Statistic"] = pd.Series(adf_statistics)
        summary["ADF p-value"] = pd.Series(adf_pvalues)
        summary["KPSS Statistic"] = pd.Series(kpss_statistics)
        summary["KPSS p-value"] = pd.Series(kpss_pvalues)
        # ADF: H0 = unit root, pass when rejected. KPSS: H0 = stationary, pass when not rejected.
        # Columns without tests (Label) stay NaN/<NA> instead of False
        adf_pass = summary["ADF p-value"].lt(self.stationarity_alpha).astype("boolean")
        kpss_pass = summary["KPSS p-value"].gt(self.stationarity_alpha).astype("boolean")
        summary["ADF Pass"] = adf_pass.where(summary["ADF p-value"].notna()).astype("Int64")
        summary["KPSS Pass"] = kpss_pass.where(summary["KPSS p-value"].notna()).astype("Int64")
        summary["Include"] = (summary["ADF Pass"] * summary["KPSS Pass"]).astype("boolean")

        display(summary)
        return summary

    def drop_non_stationary_features(self):
        summary = self.data_summary(self.X_train)
        features_to_include = list(summary.index[summary["Include"]])
        dropped = [f for f in self.X_train.columns if f not in features_to_include]
        print(
            f"Stationarity filter (ADF and KPSS): kept {len(features_to_include)}, "
            f"dropped {len(dropped)}: {dropped}"
        )
        self.X = self.X[features_to_include]
        self.X_train = self.X_train[features_to_include]

    def summarize_engineered_data(self):
        data = pd.concat([self.X_train, self.y_train], axis=1)
        self.engineered_data_summary = self.data_summary(data)

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

        self.label_imbalance = pd.DataFrame(rows)
        print("Label distribution:")
        print(
            self.label_imbalance.to_string(
                index=False,
                formatters={"percentage": "{:.2f}%".format},
            )
        )