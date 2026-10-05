import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from quantmod.datasets import fetch_historical_data
    from quantmod.indicators import BBands, SMA
    from quantmod.timeseries import Gap, HiLo, OpCl, dailyReturn, lead
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The quantmod package is required for this module. "
        "Install it with: pip install quantmod"
    ) from exc


class LinearRegressionPriceModel:
    """Linear regression workflow for index price prediction without W&B tracking."""

    def __init__(self, symbol: str = "NIFTY", test_size: float = 0.2, corr_threshold: float = 0.9):
        self.symbol = symbol
        self.test_size = test_size
        self.corr_threshold = corr_threshold

        self.df = pd.DataFrame()
        self.feature_frame = pd.DataFrame()
        self.X = pd.DataFrame()
        self.y = pd.Series(dtype=float)
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.Series(dtype=float)
        self.y_test = pd.Series(dtype=float)
        self.models = {}
        self.feature_names = []

    def load_data(self) -> pd.DataFrame:
        self.df = (
            fetch_historical_data(self.symbol)
            .assign(date=lambda frame: pd.to_datetime(frame["date"]))
            .set_index("date")
            .copy()
        )
        return self.df

    @staticmethod
    def create_features(frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        multiplier = 2

        df["OC"] = OpCl(df)
        df["HC"] = HiLo(df)
        df["GAP"] = Gap(df)
        df["RET"] = dailyReturn(df["close"])

        for period in [7, 14, 28]:
            df[f"PCHG{period}"] = df["close"].pct_change(period)
            df[f"VCHG{period}"] = df["volume"].pct_change(period)
            df[f"RET{period}"] = df["RET"].rolling(period).sum()
            df[f"MA{period}"] = SMA(df["close"], period)
            df[f"OC{period}"] = df["OC"].rolling(period).mean()
            df[f"HC{period}"] = df["HC"].rolling(period).mean()
            df[f"GAP{period}"] = df["GAP"].rolling(period).mean()
            df[f"STD{period}"] = df["RET"].rolling(period).std()
            lower_band, _, upper_band = BBands(df["close"], period, multiplier)
            df[f"LB{period}"] = lower_band
            df[f"UB{period}"] = upper_band

        df["Label"] = lead(df["close"])

        df.drop(["open", "high", "low", "close", "volume"], axis=1, inplace=True)
        df = df.dropna()
        return df

    def prepare_dataset(self) -> pd.DataFrame:
        self.feature_frame = self.create_features(self.df)

        self.X = self.feature_frame.drop(columns=["Label"], errors="ignore")
        self.y = self.feature_frame["Label"]

        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]

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

    def build_models(self) -> dict:
        models = {
            "LinearRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]),
            "Lasso": Pipeline([
                ("scaler", MinMaxScaler()),
                ("regressor", Lasso(alpha=0.1)),
            ]),
            "Ridge": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1)),
            ]),
            "ElasticNet": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", ElasticNet(alpha=0.1, l1_ratio=0.3)),
            ]),
        }
        self.models = models
        return self.models

    @staticmethod
    def regression_metrics(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            "train_r2": model.score(X_train, y_train),
            "test_r2": model.score(X_test, y_test),
            "mse": mse,
            "rmse": rmse,
        }
        return metrics

    def fit_models(self) -> dict:
        fitted_models = {}
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            fitted_models[name] = {
                "model": model,
                "metrics": self.regression_metrics(model, self.X_train, self.X_test, self.y_train, self.y_test),
            }
        return fitted_models

    def compare_models(self) -> pd.DataFrame:
        fitted = self.fit_models()
        summary = []
        for name, payload in fitted.items():
            model = payload["model"]
            metrics = payload["metrics"]
            summary.append({
                "model": name,
                "train_r2": metrics["train_r2"],
                "test_r2": metrics["test_r2"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "coef_count": int(np.sum(np.abs(model.named_steps["regressor"].coef_) > 0)) if hasattr(model.named_steps["regressor"], "coef_") else None,
            })

        comparison = pd.DataFrame(summary).sort_values("test_r2", ascending=False)
        return comparison.reset_index(drop=True)

    def run_full_pipeline(self) -> dict:
        self.load_data()
        self.prepare_dataset()
        self.split_data()
        self.build_models()
        comparison = self.compare_models()

        return {
            "data": self.df,
            "features": self.X,
            "labels": self.y,
            "comparison": comparison,
            "models": self.models,
        }

    def __repr__(self) -> str:
        return f"LinearRegressionPriceModel(symbol={self.symbol!r}, test_size={self.test_size})"


def main() -> None:
    model = LinearRegressionPriceModel(symbol="NIFTY", test_size=0.2, corr_threshold=0.9)
    result = model.run_full_pipeline()
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
