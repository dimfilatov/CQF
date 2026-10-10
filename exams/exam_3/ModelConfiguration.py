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

        self.search_params = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "colsample_bytree": self.colsample_bytree,
            "n_estimators": self.n_estimators,
        }

        # validation params
        self.n_iter = config.getint("model", "n_iter")
        self.cv_splits = config.getint("model", "cv_splits")
        self.cv_gap = config.getint("model", "cv_gap")

        # boosting params
        self.verbosity = config.getint("model", "verbosity")
        self.eval_metric = config.get("model", "eval_metric")
        self.random_state = config.getint("model", "random_state")
        self.n_jobs = config.getint("model", "n_jobs")

        self.boosting_params = {
            "verbosity": self.verbosity,
            "eval_metric": self.eval_metric,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

        # feature engineering
        self.feature_selection_methods = ast.literal_eval(config.get("feature_engineering", "feature_selection_methods"))
        self.vif_threshold=config.getint("feature_engineering","vif_threshold")
        self.n_features=config.getint("feature_engineering","n_features")
