import configparser


class ModelConfiguration:
    def __init__(self, config_path: str = "config.cfg"):
        self.config_path = config_path

    def load_params(self) -> None:
        config = configparser.ConfigParser()
        config.read(self.config_path)

        # data params
        self.symbol = config.get("data", "symbol")
        self.start_date = config.get("data", "start_date")
        self.test_size = config.get("data", "test_size")
        self.excluded_columns = config.get("data", "excluded_columns")
        self.windows = config.get("data", "windows")

        # search params
        self.learning_rate = config.get("model", "learning_rate")
        self.max_depth = config.get("model", "max_depth")
        self.min_child_weight = config.get("model", "min_child_weight")
        self.gamma = config.get("model", "gamma")
        self.colsample_bytree = config.get("model", "colsample_bytree")
        self.n_estimators = config.get("model", "n_estimators")

        self.search_params = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "colsample_bytree": self.colsample_bytree,
            "n_estimators": self.n_estimators,
        }

        # validation params
        self.n_iter= config.get("model", "n_iter")
        self.cv_splits = config.get("model", "cv_splits")
        self.cv_gap= config.get("model", "cv_gap")

        # boosting params
        self.verbosity= config.get("model", "verbosity")
        self.eval_metric=config.get("model", "eval_metric")
        self.random_state=config.get("model", "random_state")
        self.n_jobs=config.get("model", "n_jobs")

        self.boosting_params = {
            "verbosity": self.verbosity,
            "eval_metric": self.eval_metric,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def print_params(self):
        print()
