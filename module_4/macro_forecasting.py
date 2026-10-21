"""Rolling macroeconomic forecasting models translated from the R workflow."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


DEFAULT_REGRESSORS = (
    "US_10Y_1Y",
    "vol_10Y",
    "SPY",
    "vol_SPY",
    "INDPRO",
    "PAYEMS",
    "BAA10YM",
    "BAA",
    "CPI_Core",
)
MODEL_NAMES = ("AR", "LR", "DT", "RF", "XB")


class MacroForecasting:
    """Compare rolling one-period-ahead forecasts of macroeconomic variables.

    The input CSV uses its first five rows for metadata and column names; actual
    observations start on row six. The companion periods CSV specifies inclusive
    evaluation date ranges. By default, the class forecasts the transformed SPY
    variable from the R script using its configured macroeconomic regressors.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        periods_path: Optional[Path] = None,
        regressors: Sequence[str] = DEFAULT_REGRESSORS,
        target: str = "SPY",
        window_size: int = 60,
        h_gap: int = 1,
        xgb_max_depth: int = 2,
        xgb_learning_rate: float = 0.01,
        xgb_max_rounds: int = 100,
        xgb_early_stopping_rounds: int = 10,
        xgb_n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        module_dir = Path(__file__).resolve().parent
        self.data_path = Path(data_path) if data_path else (
            module_dir / "data" / "Macro_Data_US_1963.csv"
        )
        self.periods_path = Path(periods_path) if periods_path else (
            module_dir / "data" / "Macro_OOS Periods.csv"
        )
        self.regressors = tuple(regressors)
        self.target = target
        self.window_size = window_size
        self.h_gap = h_gap
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_rounds = xgb_max_rounds
        self.xgb_early_stopping_rounds = xgb_early_stopping_rounds
        self.xgb_n_splits = xgb_n_splits
        self.random_state = random_state

        if self.window_size <= self.h_gap + 2:
            raise ValueError("window_size must leave at least two training rows after the gap.")
        if self.h_gap < 0:
            raise ValueError("h_gap must be non-negative.")
        train_size = self.window_size - self.h_gap - 1
        if self.xgb_n_splits < 2 or self.xgb_n_splits > train_size:
            raise ValueError(
                "xgb_n_splits must be between 2 and the rolling training-set size."
            )
        if self.xgb_max_rounds < 1:
            raise ValueError("xgb_max_rounds must be positive.")

    def load_data(self) -> pd.DataFrame:
        """Read metadata and observations, applying scaling and differencing."""
        raw = pd.read_csv(self.data_path, header=None, dtype=str)
        if raw.shape[0] <= 5 or raw.shape[1] < 2:
            raise ValueError("Macro data CSV must contain five metadata rows and observations.")

        columns = raw.iloc[4, 1:].astype(str).tolist()
        if len(columns) != len(set(columns)):
            raise ValueError("Macro data CSV contains duplicate variable names.")

        divide_by_100 = pd.to_numeric(raw.iloc[1, 1:], errors="raise").to_numpy()
        differences = raw.iloc[2, 1:].fillna("0").astype(str).to_numpy()
        if len(columns) != len(divide_by_100) or len(columns) != len(differences):
            raise ValueError("Macro data metadata does not match the variable columns.")

        dates = pd.to_datetime(
            raw.iloc[5:, 0],
            format="%d/%m/%Y",
            errors="raise",
        )
        levels = raw.iloc[5:, 1:].apply(pd.to_numeric, errors="coerce")
        levels.columns = columns
        levels.index = pd.DatetimeIndex(dates, name="Date")

        scaled_levels = levels.copy()
        for column, divide in zip(columns, divide_by_100):
            if divide > 0:
                scaled_levels[column] = scaled_levels[column] / 100.0

        transformed = scaled_levels.copy()
        if "US_1Y" not in transformed.columns and "ER" in differences:
            raise ValueError("US_1Y is required to calculate excess-return variables.")
        for column, operation in zip(columns, differences):
            if operation == "D":
                transformed[column] = scaled_levels[column].diff()
            elif operation == "R":
                transformed[column] = scaled_levels[column].pct_change(fill_method=None)
            elif operation == "ER":
                log_return = np.log(
                    scaled_levels[column] / scaled_levels[column].shift(1)
                )
                transformed[column] = (
                    log_return - scaled_levels["US_1Y"].shift(1) / 12.0
                )
            elif operation not in ("0", "NA", ""):
                raise ValueError(
                    f"Unsupported transformation {operation!r} for {column!r}."
                )

        self.raw_levels = levels
        self.scaled_levels = scaled_levels
        self.transform_codes = dict(zip(columns, differences))
        self.transformed_data = transformed.dropna()
        self.periods = pd.read_csv(
            self.periods_path,
            parse_dates=["Start", "End"],
            dayfirst=True,
        )
        required_period_columns = {"Start", "End", "Comment"}
        missing_period_columns = required_period_columns.difference(self.periods.columns)
        if missing_period_columns:
            raise ValueError(
                "Periods CSV is missing columns: "
                + ", ".join(sorted(missing_period_columns))
            )
        if (self.periods["Start"] > self.periods["End"]).any():
            raise ValueError("Each evaluation period must start on or before it ends.")

        missing_variables = set(self.regressors + (self.target,)).difference(
            self.transformed_data.columns
        )
        if missing_variables:
            raise ValueError(
                "Requested variables are absent from macro data: "
                + ", ".join(sorted(missing_variables))
            )
        return self.transformed_data

    def prepare_dataset(self) -> pd.DataFrame:
        """Align predictors at time t with the target observed at time t+1."""
        if not hasattr(self, "transformed_data"):
            self.load_data()

        future_target = self.transformed_data[self.target].shift(-1).rename(
            "target_next"
        )
        future_date = pd.Series(
            self.transformed_data.index,
            index=self.transformed_data.index,
            name="target_date",
        ).shift(-1)
        self.supervised_data = self.transformed_data.loc[
            :, list(dict.fromkeys(self.regressors + (self.target,)))
        ].copy()
        self.supervised_data["target_next"] = future_target
        self.supervised_data["target_date"] = future_date
        self.supervised_data = self.supervised_data.dropna(
            subset=["target_next", "target_date"]
        )
        if len(self.supervised_data) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} usable observations for forecasting; "
                f"found {len(self.supervised_data)}."
            )
        return self.supervised_data

    @staticmethod
    def _standardize(
        train_features: pd.DataFrame,
        forecast_features: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_means = train_features.mean()
        feature_scales = train_features.std(ddof=1).replace(0.0, 1.0).fillna(1.0)

        standardized_train = (train_features - feature_means) / feature_scales
        standardized_forecast = (forecast_features - feature_means) / feature_scales
        return standardized_train, standardized_forecast

    @staticmethod
    def _durbin_watson(residuals: np.ndarray) -> float:
        residuals = np.asarray(residuals, dtype=float)
        denominator = float(np.dot(residuals, residuals))
        if denominator == 0.0:
            return float("nan")
        return float(np.diff(residuals).dot(np.diff(residuals)) / denominator)

    def _fit_xgboost(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        forecast_features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Choose boosting rounds by CV RMSE, then refit on the full window."""
        dtrain = xgb.DMatrix(
            train_features.to_numpy(dtype=float),
            label=train_target.to_numpy(dtype=float),
        )
        parameters = {
            "max_depth": self.xgb_max_depth,
            "eta": self.xgb_learning_rate,
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": self.random_state,
            "nthread": 1,
        }
        cv_results = xgb.cv(
            params=parameters,
            dtrain=dtrain,
            num_boost_round=self.xgb_max_rounds,
            nfold=self.xgb_n_splits,
            metrics=("rmse",),
            early_stopping_rounds=self.xgb_early_stopping_rounds,
            seed=self.random_state,
            shuffle=True,
            verbose_eval=False,
        )
        best_rounds = int(cv_results["test-rmse-mean"].idxmin()) + 1
        model = xgb.train(
            params=parameters,
            dtrain=dtrain,
            num_boost_round=best_rounds,
            verbose_eval=False,
        )
        in_sample_predictions = model.predict(dtrain)
        forecast_predictions = model.predict(
            xgb.DMatrix(forecast_features.to_numpy(dtype=float))
        )
        return in_sample_predictions, forecast_predictions, best_rounds

    def _fit_window(
        self,
        window: pd.DataFrame,
        forecast_number: int,
    ) -> Dict[str, Any]:
        train_size = self.window_size - self.h_gap - 1
        train = window.iloc[:train_size]
        forecast = window.iloc[[-1]]
        y_train_original = train["target_next"].astype(float)
        target_mean = float(y_train_original.mean())
        target_scale = float(y_train_original.std(ddof=1))
        if not np.isfinite(target_scale) or target_scale == 0.0:
            target_scale = 1.0
        y_train = (y_train_original - target_mean) / target_scale

        row: Dict[str, Any] = {
            "feature_date": forecast.index[0],
            "target_date": forecast["target_date"].iloc[0],
            "actual": float(forecast["target_next"].iloc[0]),
            "forecast_number": forecast_number,
        }
        model_features = list(self.regressors)
        for model_name in MODEL_NAMES:
            columns = [self.target] if model_name == "AR" else model_features
            X_train_original = train.loc[:, columns].astype(float)
            X_forecast_original = forecast.loc[:, columns].astype(float)
            X_train, X_forecast = self._standardize(
                X_train_original,
                X_forecast_original,
            )

            if model_name == "XB":
                in_sample_scaled, forecast_scaled, best_rounds = self._fit_xgboost(
                    X_train,
                    y_train,
                    X_forecast,
                )
                row["XB_best_rounds"] = best_rounds
            else:
                if model_name in ("AR", "LR"):
                    estimator = LinearRegression()
                elif model_name == "DT":
                    estimator = DecisionTreeRegressor(
                        min_samples_split=20,
                        min_samples_leaf=7,
                        random_state=self.random_state,
                    )
                else:
                    estimator = BaggingRegressor(
                        estimator=DecisionTreeRegressor(
                            max_features=min(3, X_train.shape[1])
                        ),
                        n_estimators=100,
                        bootstrap=False,
                        max_samples=0.632,
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
                estimator.fit(X_train, y_train)
                in_sample_scaled = estimator.predict(X_train)
                forecast_scaled = estimator.predict(X_forecast)

            in_sample = np.asarray(in_sample_scaled) * target_scale + target_mean
            prediction = (
                float(np.asarray(forecast_scaled).reshape(-1)[0]) * target_scale
                + target_mean
            )
            residuals = y_train_original.to_numpy() - in_sample
            row[model_name] = prediction
            row[f"{model_name}_train_mse"] = float(
                mean_squared_error(y_train_original, in_sample)
            )
            row[f"{model_name}_durbin_watson"] = self._durbin_watson(residuals)

        return row

    def rolling_forecast(self) -> pd.DataFrame:
        """Fit each model on a rolling window and predict its final row's target."""
        if not hasattr(self, "supervised_data"):
            self.prepare_dataset()

        forecast_count = len(self.supervised_data) - self.window_size + 1
        forecasts = []
        for start in range(forecast_count):
            window = self.supervised_data.iloc[
                start : start + self.window_size
            ]
            forecasts.append(self._fit_window(window, start + 1))

        self.forecasts = pd.DataFrame(forecasts).set_index("target_date").sort_index()
        self.forecasts.index.name = "Date"
        return self.forecasts

    def evaluate_periods(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate IC, hit ratio, and RMSFE within each configured date range."""
        if not hasattr(self, "forecasts"):
            self.rolling_forecast()

        result_rows = []
        relative_rows = []
        model_columns = list(MODEL_NAMES)
        for period in self.periods.itertuples(index=False):
            start = pd.Timestamp(period.Start)
            end = pd.Timestamp(period.End)
            sample = self.forecasts.loc[
                (self.forecasts.index >= start) & (self.forecasts.index <= end)
            ]
            actual = sample["actual"]
            metrics: Dict[str, Any] = {
                "Start": start,
                "End": end,
                "Comment": period.Comment,
                "n_forecasts": len(sample),
            }
            relative: Dict[str, Any] = {
                "Start": start,
                "End": end,
                "Comment": period.Comment,
                "n_forecasts": len(sample),
            }
            for model_name in model_columns:
                prediction = sample[model_name]
                metrics[f"{model_name}_IC"] = (
                    float(actual.corr(prediction)) if len(sample) > 1 else float("nan")
                )
                metrics[f"{model_name}_HR"] = (
                    float((actual.gt(0) == prediction.gt(0)).mean())
                    if len(sample)
                    else float("nan")
                )
                metrics[f"{model_name}_RMSFE"] = (
                    float(np.sqrt(mean_squared_error(actual, prediction)))
                    if len(sample)
                    else float("nan")
                )
                if model_name == "AR":
                    relative[f"{model_name}_IC"] = 0.0
                    relative[f"{model_name}_HR"] = 0.0
                    relative[f"{model_name}_RMSFE"] = 1.0
                else:
                    for metric in ("IC", "HR"):
                        relative[f"{model_name}_{metric}"] = (
                            metrics[f"{model_name}_{metric}"] - metrics[f"AR_{metric}"]
                        )
                    ar_rmsfe = metrics["AR_RMSFE"]
                    relative[f"{model_name}_RMSFE"] = (
                        metrics[f"{model_name}_RMSFE"] / ar_rmsfe
                        if ar_rmsfe and np.isfinite(ar_rmsfe)
                        else float("nan")
                    )
            result_rows.append(metrics)
            relative_rows.append(relative)

        self.period_results = pd.DataFrame(result_rows)
        self.relative_results = pd.DataFrame(relative_rows)
        return self.period_results, self.relative_results

    def plot_forecasts(self) -> None:
        """Plot rolling forecasts and in-sample Durbin-Watson statistics."""
        if not hasattr(self, "forecasts"):
            self.rolling_forecast()

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(self.forecasts.index, self.forecasts["actual"], label="Actual")
        for model_name in MODEL_NAMES:
            axes[0].plot(
                self.forecasts.index,
                self.forecasts[model_name],
                label=model_name,
                alpha=0.8,
            )
        axes[0].set_title(f"One-period-ahead forecast: {self.target}")
        axes[0].legend()

        for model_name in MODEL_NAMES:
            axes[1].plot(
                self.forecasts.index,
                self.forecasts[f"{model_name}_durbin_watson"],
                label=model_name,
            )
        axes[1].set_title("In-sample Durbin-Watson statistic")
        axes[1].legend()
        fig.tight_layout()
        plt.show()

    def run_full_pipeline(
        self,
        save_dir: Optional[Path] = None,
        show_plots: bool = False,
    ) -> Dict[str, Any]:
        self.load_data()
        self.prepare_dataset()
        forecasts = self.rolling_forecast()
        period_results, relative_results = self.evaluate_periods()

        if save_dir is not None:
            output_dir = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            forecasts.to_csv(output_dir / "macro_forecasts.csv")
            period_results.to_csv(output_dir / "macro_period_results.csv", index=False)
            relative_results.to_csv(
                output_dir / "macro_relative_results.csv",
                index=False,
            )

        if show_plots:
            self.plot_forecasts()

        return {
            "transformed_data": self.transformed_data,
            "supervised_data": self.supervised_data,
            "forecasts": forecasts,
            "period_results": period_results,
            "relative_results": relative_results,
        }

    def __repr__(self) -> str:
        return (
            f"MacroForecasting(target={self.target!r}, "
            f"window_size={self.window_size}, h_gap={self.h_gap}, "
            f"regressors={self.regressors!r})"
        )


def main() -> None:
    model = MacroForecasting()
    output_dir = Path(__file__).resolve().parent / "macro_forecast_output"
    result = model.run_full_pipeline(save_dir=output_dir)
    print("Forecast performance by configured period:")
    print(result["period_results"].to_string(index=False))
    print("\nPerformance relative to the AR benchmark:")
    print(result["relative_results"].to_string(index=False))


if __name__ == "__main__":
    main()
