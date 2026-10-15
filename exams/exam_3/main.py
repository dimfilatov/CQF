from data_engineering import DataEngineering
from ModelConfiguration import ModelConfiguration
from ML_classifier import MLClassifier
import pandas as pd
config_path = "config.cfg"
config = ModelConfiguration(config_path)
config.load_params()
data_engineering= DataEngineering(
    yahoo_symbol=config.yahoo_symbol,
    start_date=config.start_date,
    end_date=config.end_date,
    test_size=config.test_size,
    required_columns=config.required_columns,
    windows=config.windows,
    vol_scale=config.vol_scale
)
data_engineering.load_data()
data_engineering.create_features()
data_engineering.summarize_engineered_data()
data_engineering.show_label_imbalance()
data_engineering.drop_non_stationary_features()
tomorrow_returns = data_engineering.create_returns()

data_engineering.feature_engineering(config.feature_selection_methods, 
                                                config.corr_threshold, 
                                                config.n_features)

best_parameters = {}
metrics = {}
strategies = {}
classifiers = {}
confusion_matrices = {}
roc_curves = {}
precision_recall_curves = {}
for boosting_algo in config.boosting_algorithms:
    model = MLClassifier(X=data_engineering.X, 
                        y=data_engineering.y,
                        features=data_engineering.features,
                        test_size=config.test_size,
                        search_params = config.search_params,
                        n_iter = config.n_iter,
                        cv_splits = config.cv_splits,
                        cv_gap = config.cv_gap,
                        n_jobs=config.n_jobs,
                        random_state=config.random_state,
                        boosting_params = config.boosting_params,
                        boosting_algo = boosting_algo)
    model.run_full_pipeline()
    best_parameters[boosting_algo] = model.search.best_params_
    metrics[boosting_algo] = model.metrics
    classifiers[boosting_algo] = model

    model.create_trading_signal(tomorrow_returns)
    strategies[boosting_algo] = model.strategy

for boossting_algo in config.boosting_algorithms:

    print(f"Best parameters for {boossting_algo}:")
    print(pd.DataFrame(best_parameters[boossting_algo], index=[0]))


print("Model metrics:")
print(pd.DataFrame(metrics))

MLClassifier.plot_evaluation_grid(classifiers)

print("Trading strategies:")
print(pd.DataFrame(strategies))


