import os
import warnings
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from quantmod.datasets import fetch_historical_data
    from quantmod.indicators import BBands
    from quantmod.timeseries import Gap, HiLo, OpCl, dailyReturn, lead
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "The quantmod package is required for this module. "
        "Install it with: pip install quantmod"
    ) from exc

class TrendPredictionLogisticRegression:
    """Logistic-regression workflow for trend prediction using NIFTY-style market data."""

    def __init__(
        self,
        symbol: str = "NIFTY",
        start_date: str = "2010",
        test_size: float = 0.2,
        corr_threshold: float = 0.9
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.test_size = test_size
        self.corr_threshold = corr_threshold
        self.df = pd.DataFrame()
        self.feature_frame = pd.DataFrame()
        self.X = pd.DataFrame()
        self.y = pd.Series(dtype="int64")
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.Series(dtype="int64")
        self.y_test = pd.Series(dtype="int64")
        self.model = None
        self.feature_names = []
        self.outlier_features = ["LB7", "OC", "HC", "HC7"]
        self.other_features: List[str] = []
        self.metric_summary: Dict[str, float] = {}
        self.run = None

    def load_data(self) -> pd.DataFrame:
        data = (
            fetch_historical_data(self.symbol)
            .assign(date=lambda frame: pd.to_datetime(frame["date"]))
            .set_index("date")
            .loc[self.start_date:]
            .copy()
        )
        self.df = data
        return self.df

    @staticmethod
    def create_features(frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        multiplier = 2

        df["OC"] = OpCl(df)
        df["HC"] = HiLo(df)
        df["GAP"] = Gap(df)
        df["RET"] = dailyReturn(df["close"])

        for window in [7, 14, 28]:
            df[f"PCHG{window}"] = df["close"].pct_change(window)
            df[f"VCHG{window}"] = df["volume"].pct_change(window)
            df[f"RET{window}"] = df["RET"].rolling(window).sum()
            df[f"OC{window}"] = df["OC"].rolling(window).mean()
            df[f"HC{window}"] = df["HC"].rolling(window).mean()
            df[f"GAP{window}"] = df["GAP"].rolling(window).mean()
            df[f"STD{window}"] = df["RET"].rolling(window).std()
            lower_band, _, upper_band = BBands(df["close"], window, multiplier)
            df[f"LB{window}"] = lower_band
            df[f"UB{window}"] = upper_band

        df["Label"] = np.where(lead(df["close"]) > df["close"], 1, 0)
        df["Label"] = np.where(lead(df["close"]) > 0.9950 * df["close"], 1, 0)
        df["Label"] = np.where(lead(df["close"]) > df["UB7"], 1, 0)

        df.drop(["open", "high", "low", "close", "volume"], axis=1, inplace=True)
        df = df.dropna()
        return df

    def prepare_dataset(self) -> pd.DataFrame:
        self.feature_frame = self.create_features(self.df)
        self.X = self.feature_frame.drop(columns=["Label"], errors="ignore")
        self.y = self.feature_frame["Label"]

        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column for column in upper.columns if any(upper[column] > self.corr_threshold)
        ]

        self.X = self.X.drop(columns=to_drop, errors="ignore")
        self.feature_names = list(self.X.columns)
        return self.X

    def split_data(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            shuffle=False,
        )

    def build_model(self) -> Pipeline:
        self.other_features = [
            col for col in self.feature_names if col not in self.outlier_features
        ]

        scaler = ColumnTransformer(
            transformers=[
                ("robust", RobustScaler(), self.outlier_features),
                ("standard", StandardScaler(), self.other_features),
            ]
        )

        model = Pipeline(
            steps=[
                ("scaling", scaler),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=1000,
                    ),
                ),
            ]
        )

        self.model = model
        return self.model

    def fit(self) -> Pipeline:
        self.model = self.build_model()
        self.model.fit(self.X_train, self.y_train)

        return self.model

    def evaluate(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("The model must be fit before calling evaluate().")

        preds = self.model.predict(self.X_test)
        proba = self.model.predict_proba(self.X_test)
        positive_index = list(self.model.named_steps["classifier"].classes_).index(1)
        positive_probability = proba[:, positive_index]

        metrics = {
            "train_accuracy": accuracy_score(self.y_train, self.model.predict(self.X_train)),
            "test_accuracy": accuracy_score(self.y_test, preds),
            "precision": precision_score(self.y_test, preds),
            "recall": recall_score(self.y_test, preds),
            "f1": f1_score(self.y_test, preds),
            "roc_auc": roc_auc_score(self.y_test, positive_probability),
        }
        self.metric_summary = metrics

        return metrics

    def plot_confusion_matrix(self) -> None:
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")

        disp = ConfusionMatrixDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            cmap=plt.cm.Blues,
        )
        disp.ax_.set_title("Confusion matrix")
        plt.show()

    def plot_roc_curve(self) -> None:
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")

        positive_index = list(self.model.named_steps["classifier"].classes_).index(1)
        probabilities = self.model.predict_proba(self.X_test)[:, positive_index]
        disp = RocCurveDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            name="Baseline Model",
        )
        plt.title("AUC-ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random 50:50")
        plt.legend()
        plt.show()
        _ = probabilities

    def create_trading_signal(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("The model must be fit before creating the trading signal.")

        signal_frame = self.X_test.copy()
        signal_frame["Signal"] = self.model.predict(self.X_test)
        signal_frame["Returns"] = self.feature_frame.loc[self.X_test.index, "RET"]
        signal_frame["Strategy"] = signal_frame["Returns"] * signal_frame["Signal"].shift(1).fillna(0)
        signal_frame.index = signal_frame.index.tz_localize("utc") if hasattr(signal_frame.index, "tz") else signal_frame.index

        cumulative_strategy = (1 + signal_frame["Strategy"]).cumprod() - 1
        cumulative_benchmark = (1 + signal_frame["Returns"]).cumprod() - 1

        sharpe = (signal_frame["Strategy"].mean() / signal_frame["Strategy"].std()) * np.sqrt(252)
        total_return = cumulative_strategy.iloc[-1]
        benchmark_return = cumulative_benchmark.iloc[-1]
        max_drawdown = (cumulative_strategy - cumulative_strategy.cummax()).min()

        signal_frame["Strategy_Cum"] = cumulative_strategy
        signal_frame["Benchmark_Cum"] = cumulative_benchmark
        signal_frame["Sharpe"] = sharpe
        signal_frame["Total_Return"] = total_return
        signal_frame["Benchmark_Return"] = benchmark_return
        signal_frame["Max_Drawdown"] = max_drawdown
        return signal_frame

    def run_full_pipeline(self) -> Dict[str, object]:
        self.load_data()
        self.prepare_dataset()
        self.split_data()
        self.fit()
        self.evaluate()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        strategy = self.create_trading_signal()

        return {
            "data": self.df,
            "features": self.X,
            "labels": self.y,
            "model": self.model,
            "metrics": self.metric_summary,
            "strategy": strategy,
        }

    def __repr__(self) -> str:
        return (
            f"TrendPredictionLogisticRegression(symbol={self.symbol!r}, "
            f"start_date={self.start_date!r}, test_size={self.test_size})"
        )


def main() -> None:
    warnings.filterwarnings("ignore")
    model = TrendPredictionLogisticRegression(
        symbol="NIFTY",
        start_date="2010",
        test_size=0.2,
        corr_threshold=0.9
    )
    result = model.run_full_pipeline()
    print("Training metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}")
    print("\nCompleted logistic regression trend prediction workflow.")


if __name__ == "__main__":
    main()
