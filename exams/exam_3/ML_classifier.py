from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, plot_importance
from ModelConfiguration import ModelConfiguration
from utils import split_data

class MLClassifier:
    """XGBoost workflow for predicting next-period SPY price direction."""

    def __init__(self, X: pd.DataFrame, 
                y: pd.Series,
                features: List,
                test_size: float,
                search_params: Dict,
                n_iter: int,
                cv_splits: int,
                cv_gap: int,
                verbosity: int,
                eval_metric: str,
                random_state: int,
                n_jobs: int,
                boosting_params: Dict
                ) -> None:
        self.X_train, self.X_test  = split_data(X[features], test_size)
        self.y_train, self.y_test  = split_data(y, test_size)
        self.search_params = search_params
        self.n_iter = n_iter
        self.cv_splits = cv_splits
        self.cv_gap = cv_gap
        self.verbosity = verbosity
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.boosting_params = boosting_params

    def tune_model(
        self
    ) -> RandomizedSearchCV:
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=self.y_train,
        )
        model = XGBClassifier(**self.boosting_params)
        self.search = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.search_params,
            n_iter=self.n_iter,
            scoring="roc_auc",
            cv=TimeSeriesSplit(n_splits=self.cv_splits, gap=self.cv_gap),
            n_jobs=self.n_jobs,
            refit=True,
            random_state=self.random_state,
        )
        self.search.fit(
            self.X_train,
            self.y_train,
            sample_weight=sample_weights,
        )
        self.model = self.search.best_estimator_
        return self.search

    def fit(self) -> Any:
        self.tune_model()
        return self.model

    def evaluate(self) -> Dict[str, float]:
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        test_probabilities = self.model.predict_proba(self.X_test)
        positive_index = list(self.model.classes_).index(1)
        positive_probabilities = test_probabilities[:, positive_index]

        metrics = {
            "train_accuracy": accuracy_score(self.y_train, train_predictions),
            "test_accuracy": accuracy_score(self.y_test, test_predictions),
            "train_balanced_accuracy": balanced_accuracy_score(
                self.y_train, train_predictions
            ),
            "test_balanced_accuracy": balanced_accuracy_score(
                self.y_test, test_predictions
            ),
            "roc_auc": roc_auc_score(self.y_test, positive_probabilities),
            "f1": f1_score(self.y_test, test_predictions, zero_division=0),
            "precision": precision_score(
                self.y_test, test_predictions, zero_division=0
            ),
            "recall": recall_score(self.y_test, test_predictions, zero_division=0),
        }
        self.metric_summary = metrics
        self.classification_summary = classification_report(
            self.y_test,
            test_predictions,
            output_dict=True,
            zero_division=0,
        )
        return metrics

    def plot_confusion_matrix(self) -> None:
        disp = ConfusionMatrixDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            display_labels=self.model.classes_,
            cmap=plt.cm.Blues,
        )
        disp.ax_.set_title("Confusion matrix")
        plt.show()

    def plot_roc_curve(self) -> None:
        disp = RocCurveDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            name="XGBoost",
        )
        disp.ax_.set_title("ROC curve")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.show()

    def plot_precision_recall_curve(self) -> None:
        disp = PrecisionRecallDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            name="XGBoost",
        )
        disp.ax_.set_title("Precision-recall curve")
        plt.show()

    def plot_feature_importance(self, importance_type: str = "gain") -> None:
        plot_importance(
            self.model,
            importance_type=importance_type,
            show_values=False,
        )
        plt.tight_layout()
        plt.show()

    def create_trading_signal(self, returns) -> pd.DataFrame:

        signal_frame = self.X_test.copy()
        signal_frame["Signal"] = self.model.predict(self.X_test)
        signal_frame["RET"] = returns
        signal_frame["Strategy"] = signal_frame["RET"] * signal_frame["Signal"].shift(1).fillna(0)
        signal_frame.index = signal_frame.index.tz_localize("utc") if hasattr(signal_frame.index, "tz") else signal_frame.index

        cumulative_strategy = (1 + signal_frame["Strategy"]).cumprod() - 1
        cumulative_benchmark = (1 + signal_frame["RET"]).cumprod() - 1

        sharpe = (signal_frame["Strategy"].mean() / signal_frame["Strategy"].std()) * np.sqrt(252)
        benchmark_sharpe = (
            signal_frame["RET"].mean() / signal_frame["RET"].std()
        ) * np.sqrt(252)
        total_return = cumulative_strategy.iloc[-1]
        benchmark_return = cumulative_benchmark.iloc[-1]
        max_drawdown = (cumulative_strategy - cumulative_strategy.cummax()).min()

        signal_frame["Strategy_Cum"] = cumulative_strategy
        signal_frame["Benchmark_Cum"] = cumulative_benchmark
        signal_frame["Sharpe"] = sharpe
        signal_frame["Benchmark_Sharpe"] = benchmark_sharpe
        signal_frame["Total_Return"] = total_return
        signal_frame["Benchmark_Return"] = benchmark_return
        signal_frame["Max_Drawdown"] = max_drawdown
        return signal_frame

    def run_full_pipeline(
        self,
        show_plots: bool = True
    ) -> Dict[str, Any]:
        self.fit()
        metrics = self.evaluate()

        if show_plots:
            self.plot_confusion_matrix()
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
            self.plot_feature_importance()

        return {
            "model": self.model,
            "metrics": metrics,
            "classification_report": self.classification_summary,
            "best_params": self.search.best_params_,
            "best_cv_roc_auc": self.search.best_score_,
            "best_cv_roc_auc_std": self.search.cv_results_["std_test_score"][self.search.best_index_]
        }