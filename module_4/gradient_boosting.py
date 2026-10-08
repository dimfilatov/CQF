from pathlib import Path
from typing import Any, Dict, Optional

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


SEARCH_PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
    "n_estimators": [100, 200, 300],
}


class GradientBoostingTrendClassifier:
    """XGBoost workflow for predicting next-period SPY price direction."""

    def __init__(
        self,
        data_path,
        start_date,
        test_size,
    ) -> None:
        self.data_path = data_path
        self.start_date = start_date
        self.test_size = test_size

    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        self.df = data.sort_index().loc[self.start_date:].copy()
        return self.df

    @staticmethod
    def create_features(frame: pd.DataFrame) -> pd.DataFrame:
        """Create rolling return/volatility features and the next-period label."""

        df = frame.copy()
        df["Returns"] = np.log(df["Adj Close"]).diff()
        for window in range(10, 65, 5):
            df[f"Ret_{window}"] = df["Returns"].rolling(window).sum()
            df[f"Std_{window}"] = df["Returns"].rolling(window).std()

        next_adjusted_close = df["Adj Close"].shift(-1)
        df["Label"] = (
            next_adjusted_close > 0.995 * df["Adj Close"]
        ).astype("float64")
        df.loc[next_adjusted_close.isna(), "Label"] = np.nan
        df = df.dropna()
        df["Label"] = df["Label"].astype("int64")
        return df

    def prepare_dataset(self) -> pd.DataFrame:
        self.feature_frame = self.create_features(self.df)
        excluded_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Returns",
            "Label",
        ]
        self.X = self.feature_frame.drop(columns=excluded_columns, errors="ignore")
        self.y = self.feature_frame["Label"]
        self.feature_names = list(self.X.columns)
        return self.X

    def split_data(self) -> None:
        split_at = int(len(self.X) * (1 - self.test_size))
        self.X_train, self.X_test = self.X.iloc[:split_at], self.X.iloc[split_at:]
        self.y_train, self.y_test = self.y.iloc[:split_at], self.y.iloc[split_at:]

    def show_label_imbalance(self) -> pd.DataFrame:
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
        return report

    @staticmethod
    def _make_model(parameters: Optional[Dict[str, Any]] = None) -> Any:

        model_parameters = {
            "verbosity": 0,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
        }
        if parameters:
            model_parameters.update(parameters)
        return XGBClassifier(**model_parameters)

    def build_model(self) -> Any:
        self.model = self._make_model()
        return self.model

    def tune_model(
        self,
        n_iter: int = 5,
        cv_splits: int = 5,
        cv_gap: int = 1,
    ) -> RandomizedSearchCV:
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=self.y_train,
        )
        self.search = RandomizedSearchCV(
            estimator=self._make_model(),
            param_distributions=SEARCH_PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=TimeSeriesSplit(n_splits=cv_splits, gap=cv_gap),
            n_jobs=-1,
            refit=True,
            random_state=42,
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

    def run_full_pipeline(
        self,
        show_plots: bool = True,
    ) -> Dict[str, Any]:
        self.load_data()
        self.prepare_dataset()
        self.split_data()
        label_imbalance = self.show_label_imbalance()
        self.fit()
        metrics = self.evaluate()

        if show_plots:
            self.plot_confusion_matrix()
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
            self.plot_feature_importance()

        return {
            "data": self.df,
            "features": self.X,
            "labels": self.y,
            "label_imbalance": label_imbalance,
            "model": self.model,
            "metrics": metrics,
            "classification_report": self.classification_summary,
            "best_params": self.search.best_params_,
            "best_cv_roc_auc": self.search.best_score_,
            "best_cv_roc_auc_std": self.search.cv_results_["std_test_score"][
                self.search.best_index_
            ],
        }

    def __repr__(self) -> str:
        return (
            f"GradientBoostingTrendClassifier(data_path={str(self.data_path)!r}, "
            f"start_date={self.start_date!r}, test_size={self.test_size})"
        )


def main() -> None:
    data_path = Path("./data/SPY.csv")
    start_date="2010"
    test_size=0.2
    model = GradientBoostingTrendClassifier(data_path=data_path, start_date=start_date, test_size=test_size)
    result = model.run_full_pipeline()
    print("Model metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}")
    print(f"Best CV ROC-AUC: {result['best_cv_roc_auc']:.4f}")
    print(f"CV ROC-AUC standard deviation: {result['best_cv_roc_auc_std']:.4f}")
    print("Best parameters:")
    for key, value in result["best_params"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
