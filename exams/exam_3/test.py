from data_handler import DataHandler
from ModelConfiguration import ModelConfiguration
from spy_signal_prediction import GradientBoostingTrendClassifier

config_path = r"exams\exam_3\config.cfg"
config = ModelConfiguration(config_path)
config.load_params()
data_handler = DataHandler(
    yahoo_symbol=config.yahoo_symbol,
    start_date=config.start_date,
    test_size=config.test_size,
    required_columns=config.required_columns,
    windows=config.windows,
    vol_scale=config.vol_scale
)
data_handler.load_data()
data_handler.data_summary()
X, y = data_handler.create_features()
data_handler.data_summary()
model = GradientBoostingTrendClassifier(X=X, 
                                        y=y,
                                        test_size = config.test_size, 
                                        search_params = config.search_params,
                                        n_iter = config.n_iter,
                                        cv_splits = config.cv_splits,
                                        cv_gap = config.cv_gap,
                                        verbosity = config.verbosity,
                                        eval_metric=config.eval_metric,
                                        random_state=config.random_state,
                                        n_jobs=config.n_jobs,
                                        boosting_params = config.boosting_params
                                        )
result = model.run_full_pipeline()
print("Model metrics:")
for key, value in result["metrics"].items():
    print(f"  {key}: {value:.4f}")
print(f"Best CV ROC-AUC: {result['best_cv_roc_auc']:.4f}")
print(f"CV ROC-AUC standard deviation: {result['best_cv_roc_auc_std']:.4f}")
print("Best parameters:")
for key, value in result["best_params"].items():
    print(f"  {key}: {value}")