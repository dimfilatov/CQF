from data_engineering import DataEngineering
from ModelConfiguration import ModelConfiguration
from ML_classifier import MLClassifier
from word_summary import (
    Data_engineering_template,
    data_engineering_word_summary,
    feature_engineering_word_summary,
    produce_label_imbalance_summary,
)

config_path = "config.cfg"
config = ModelConfiguration(config_path)
config.load_params()
data_engineering= DataEngineering(
    yahoo_symbol=config.yahoo_symbol,
    start_date=config.start_date,
    test_size=config.test_size,
    required_columns=config.required_columns,
    windows=config.windows,
    vol_scale=config.vol_scale
)
data_engineering.load_data()
Data_engineering_template(data_engineering)
data_engineering.create_features()
data_engineering.summarize_engineered_data()
data_engineering.show_label_imbalance()
data_engineering.drop_non_stationary_features()
tomorrow_returns = data_engineering.create_returns()
data_engineering_word_summary(data_engineering)
produce_label_imbalance_summary(data_engineering)
data_engineering.feature_engineering(config.feature_selection_methods, 
                                                config.corr_threshold, 
                                                config.n_features)

feature_engineering_word_summary(data_engineering)
model = MLClassifier(X=data_engineering.X, 
                    y=data_engineering.y,
                    features=data_engineering.features,
                    test_size=config.test_size,
                    search_params = config.search_params,
                    n_iter = config.n_iter,
                    cv_splits = config.cv_splits,
                    cv_gap = config.cv_gap,
                    verbosity = config.verbosity,
                    eval_metric=config.eval_metric,
                    random_state=config.random_state,
                    n_jobs=config.n_jobs,
                    boosting_params = config.boosting_params)
result = model.run_full_pipeline()
strategy = model.create_trading_signal(tomorrow_returns)
print("Model metrics:")
for key, value in result["metrics"].items():
    print(f"  {key}: {value:.4f}")
print(f"Best CV ROC-AUC: {result['best_cv_roc_auc']:.4f}")
print(f"CV ROC-AUC standard deviation: {result['best_cv_roc_auc_std']:.4f}")
print("Best parameters:")
for key, value in result["best_params"].items():
    print(f"  {key}: {value}")

for key in [
    "Sharpe",
    "Benchmark_Sharpe",
    "Sortino",
    "Benchmark_Sortino",
    "Total_Return",
    "Benchmark_Return",
    "Max_Drawdown",
    "Benchmark_Max_Drawdown",
]:
    print(f"{key}: {strategy[key].iloc[-1]:.4f}")
print("\nCompleted trend prediction workflow.")