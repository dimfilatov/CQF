import ast
from configparser import ConfigParser, ExtendedInterpolation


class ModelConfiguration:
    def __init__(self, config_path):
        self.config_path = config_path

    def load_params(self) -> None:
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.optionxform = lambda key: key
        config.read(self.config_path)

        # data params
        self.yahoo_symbol = config.get("data", "yahoo_symbol")
        self.start_date = config.get("data", "start_date")
        self.test_size = config.getfloat("data", "test_size")
        self.required_columns = ast.literal_eval(
            config.get("data", "required_columns")
        )
        self.windows = ast.literal_eval(config.get("data", "windows"))
        self.vol_scale = config.getfloat("data", "vol_scale")

        # search params
        self.learning_rate = ast.literal_eval(config.get("model", "learning_rate"))
        self.max_depth = ast.literal_eval(config.get("model", "max_depth"))
        self.min_child_weight = ast.literal_eval(
            config.get("model", "min_child_weight")
        )
        self.gamma = ast.literal_eval(config.get("model", "gamma"))
        self.colsample_bytree = ast.literal_eval(
            config.get("model", "colsample_bytree")
        )
        self.n_estimators = ast.literal_eval(config.get("model", "n_estimators"))
        self.reg_alpha = ast.literal_eval(config.get("model", "reg_alpha"))
        self.reg_lambda = ast.literal_eval(config.get("model", "reg_lambda"))
        self.boosting_algorithms = ast.literal_eval(config.get("model", "boosting_algorithms"))
        self.random_strength = ast.literal_eval(config.get("model", "random_strength"))
        self.subsample = ast.literal_eval(config.get("model", "subsample"))
        self.subsample_freq = config.getint("model", "subsample_freq")
        self.search_params = {
            "xgboost":{
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                    "min_child_weight": self.min_child_weight,
                    "gamma": self.gamma,
                    "reg_alpha": self.reg_alpha,
                    "reg_lambda": self.reg_lambda,
                    "colsample_bytree": self.colsample_bytree,
                    "n_estimators": self.n_estimators,
                    "subsample": self.subsample,
                    },
            "lightgbm": {
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_child_weight": self.min_child_weight,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "min_split_gain": self.gamma,
                "colsample_bytree": self.colsample_bytree,
                "n_estimators": self.n_estimators,
                "subsample": self.subsample
            },
            "catboost": {
                "learning_rate": self.learning_rate,
                "depth": self.max_depth,
                "l2_leaf_reg": self.reg_lambda,
                "random_strength": self.random_strength,
                "rsm": self.colsample_bytree,
                "iterations": self.n_estimators
            },
        }

        # validation params
        self.n_iter = config.getint("model", "n_iter")
        self.cv_splits = config.getint("model", "cv_splits")
        self.cv_gap = config.getint("model", "cv_gap")

        # boosting params
        self.random_state = config.getint("model", "random_state")
        self.n_jobs = config.getint("model", "n_jobs")

        self.boosting_params = {
            "xgboost": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbosity": 0,
            },
            "lightgbm": {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbosity": -1,
                "subsample_freq": self.subsample_freq
            },
            "catboost": {
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "grow_policy": "SymmetricTree",
                "has_time": True,
                "boosting_type": "Ordered",
                "random_seed": self.random_state,
                "thread_count": self.n_jobs,
                "verbose": False,
                "allow_writing_files": False,
            },
        }

        # feature engineering
        self.feature_selection_methods = ast.literal_eval(config.get("feature_engineering", "feature_selection_methods"))
        self.corr_threshold=config.getfloat("feature_engineering","corr_threshold")
        self.n_features=config.getint("feature_engineering","n_features")
