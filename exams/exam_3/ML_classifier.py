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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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
                n_jobs: int,
                random_state: int,
                boosting_params: Dict,
                boosting_algo: str
                ) -> None:
        self.X_train, self.X_test  = split_data(X[features], test_size)
        self.y_train, self.y_test  = split_data(y, test_size)
        self.test_size = test_size
        self.search_params = search_params[boosting_algo]
        self.n_iter = n_iter
        self.cv_splits = cv_splits
        self.cv_gap = cv_gap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.boosting_params = boosting_params[boosting_algo]
        self.boosting_algo = boosting_algo
        self.strategy = dict()

    def tune_model(
        self
    ) -> RandomizedSearchCV:
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=self.y_train,
        )
        self.positive_class_ratio = self.y_train.value_counts(normalize=True)[1]

        if self.boosting_algo == "xgboost":
            model = XGBClassifier(**self.boosting_params)
        elif self.boosting_algo == "lightgbm":
            model = LGBMClassifier(**self.boosting_params)
        elif self.boosting_algo == "catboost":
            model = CatBoostClassifier(**self.boosting_params)
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

    def evaluate(self) -> Dict[str, float]:
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        test_probabilities = self.model.predict_proba(self.X_test)
        positive_index = list(self.model.classes_).index(1)
        positive_probabilities = test_probabilities[:, positive_index]

        self.metrics = {
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
        self.classification_summary = classification_report(
            self.y_test,
            test_predictions,
            output_dict=True,
            zero_division=0,
        )

    @staticmethod
    def plot_evaluation_grid(classifiers: Dict[str, "MLClassifier"]) -> None:
        """Plot confusion matrices, ROC curves and precision-recall curves.

        Rows are the three plot types and columns are the boosting algorithms.
        """
        n_algos = len(classifiers)
        fig, axes = plt.subplots(3, n_algos, figsize=(5 * n_algos, 16), squeeze=False)
        for col, (algo, clf) in enumerate(classifiers.items()):
            ConfusionMatrixDisplay.from_estimator(
                clf.model,
                clf.X_test,
                clf.y_test,
                display_labels=clf.model.classes_,
                cmap=plt.cm.Blues,
                ax=axes[0, col],
                colorbar=False,
            )
            axes[0, col].set_title(f"{algo}: confusion matrix")

            RocCurveDisplay.from_estimator(
                clf.model, clf.X_test, clf.y_test, name=algo, ax=axes[1, col]
            )
            axes[1, col].plot([0, 1], [0, 1], linestyle="--", color="grey")
            axes[1, col].set_title(f"{algo}: ROC curve")

            PrecisionRecallDisplay.from_estimator(
                clf.model, clf.X_test, clf.y_test, name=algo, ax=axes[2, col]
            )
            axes[2, col].set_title(f"{algo}: precision-recall curve")
        fig.tight_layout(h_pad=4.0)
        plt.show()
        plt.show()

    def plot_feature_importance(self, model, importance_type: str = "gain") -> None:
        plot_importance(
            model,
            importance_type=importance_type,
            show_values=False,
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Annualised Sortino ratio with a zero minimum acceptable return.

        Downside deviation is sqrt(mean(min(r, 0)^2)) over all observations,
        so days out of the market (zero return) stay in the denominator.
        """
        downside_deviation = np.sqrt((np.minimum(returns, 0.0) ** 2).mean())
        return (returns.mean() / downside_deviation) * np.sqrt(periods_per_year)

    @staticmethod
    def max_drawdown(cumulative_returns: pd.Series) -> float:
        """Worst peak-to-trough fall of a cumulative log-return curve."""
        return (cumulative_returns - cumulative_returns.cummax()).min()

    def create_trading_signal(self, returns) -> pd.DataFrame:
        
        self.signal = self.X_test.copy()
        self.signal["Signal"] = self.model.predict(self.X_test)
        self.signal["tomorrow_returns"] = returns
        self.signal["Strategy"] = self.signal["tomorrow_returns"] * self.signal["Signal"]

        strategy_cum = self.signal["Strategy"].cumsum()
        benchmark_cum = self.signal["tomorrow_returns"].cumsum()

        self.signal["coin_flip"] = self.coin_flip_strategy() * self.signal["tomorrow_returns"]
        flip_coin_cum = self.signal["coin_flip"].cumsum()

        self.strategy["Sharpe"] = (self.signal["Strategy"].mean() / self.signal["Strategy"].std()) * np.sqrt(252)
        self.strategy["Benchmark_Sharpe"] = (self.signal["tomorrow_returns"].mean() / self.signal["tomorrow_returns"].std()) * np.sqrt(252)
        self.strategy["Coin_Sharpe"] = (self.signal["coin_flip"].mean() / self.signal["coin_flip"].std()) * np.sqrt(252)
        
        self.strategy["Total_Return"] = strategy_cum.iloc[-1]
        self.strategy["Benchmark_Return"] = benchmark_cum.iloc[-1]
        self.strategy["Coin_Return"] = flip_coin_cum.iloc[-1]

        self.strategy["Max_Drawdown"] = self.max_drawdown(strategy_cum)
        self.strategy["Benchmark_Max_Drawdown"] = self.max_drawdown(benchmark_cum)
        self.strategy["Coin_Max_Drawdown"] = self.max_drawdown(flip_coin_cum)

        self.strategy["Sortino"] = self.sortino_ratio(self.signal["Strategy"])
        self.strategy["Benchmark_Sortino"] = self.sortino_ratio(self.signal["tomorrow_returns"])
        self.strategy["Coin_Sortino"] = self.sortino_ratio(self.signal["coin_flip"])

    def coin_flip_strategy(self) -> List:
        np.random.seed(self.random_state)
        return np.random.rand(len(self.X_test)) < self.positive_class_ratio
        
    def run_full_pipeline(self) -> None:
        self.tune_model()
        self.evaluate()