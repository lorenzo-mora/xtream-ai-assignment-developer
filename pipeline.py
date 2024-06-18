from datetime import datetime
import json
from logging import Logger
from pathlib import Path
import tempfile
import time
from typing import Any, List, Tuple
import uuid
import pandas as pd
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge)
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import joblib
import requests
from sklearn.tree import DecisionTreeRegressor

from utils import (get_config_value, DEFAULT_TRAIN_CONF, DEFAULT_DATA_CONF,
                   LoggerManager, transform_data, FileUtils, snake_to_camel)

BASE_PATH = Path(__file__).resolve().parent

class DataManager:
    def __init__(self, config_file: Path, logger: Logger) -> None:
        self.configuration = FileUtils.retrieve_config(config_file,
                                                       default_conf=DEFAULT_DATA_CONF)
        self.logger = logger

    # 1. Data Acquisition
    def fetch_data(self) -> Tuple[pd.DataFrame, str]:
        fetch_from_url = get_config_value(self.configuration,
                                          ['data', 'onlineSource'],
                                          DEFAULT_DATA_CONF)
        if fetch_from_url:
            # The data are to be downloaded online
            source_path = get_config_value(self.configuration,
                                           ['data', 'url'],
                                           DEFAULT_DATA_CONF)
        else:
            # Data are to be retrieved from local
            source_path = get_config_value(self.configuration,
                                           ['data', 'localPath'],
                                           DEFAULT_DATA_CONF)
        self.logger.info(f"Acquiring data from '{source_path}'")

        data = (self._fetch_from_online(url=source_path)
                if fetch_from_url else self._fetch_from_local(path_to_data=source_path))

        return data, source_path.rsplit('/', 1)[-1]

    def _fetch_from_online(self, url: str) -> pd.DataFrame:        
        try:
            # Fetch data from url
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch data from {url}: {e}")
            raise

        tmp_file_path = None

        try:
            # Reading data from url content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = Path(tmp_file.name)

            data = pd.read_csv(tmp_file_path)
        except Exception as e:
            self.logger.error(f"Failed to read data into DataFrame: {e}")
            raise
        finally:
            # Cleaning unused files
            if tmp_file_path and tmp_file_path.exists():
                try:
                    tmp_file_path.unlink()
                except Exception as e:
                    self.logger.warning(
                        f"Failed to delete temporary file {tmp_file_path}: {e}")
        return data

    def _fetch_from_local(self, path_to_data: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(path_to_data)
        except Exception as e:
            self.logger.error(f"Failed to read data from {path_to_data}: {e}")
            raise
        return data

    # 2. Data Preprocessing
    def preprocess_data(self, original_data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting data processing...")

        processed_data = original_data.copy()
        processed_data = self._clean_data(processed_data)
        processed_data = self._drop_columns(processed_data)
        processed_data = self._convert_categorical(processed_data)

        # Is it required to save the processed data?
        save = get_config_value(self.configuration,
                                ['data', 'saveData'],
                                DEFAULT_DATA_CONF)
        if save:
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
            dest_path = f'data/processed_data-{timestamp}.csv'
            processed_data.to_csv(dest_path, index=False)
            self.logger.info(f"Dataframe successfully saved in {dest_path}")

        return processed_data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removal of instances with errors."""
        # Check for and remove rows with missing values
        if data.isna().sum().any():
            data = data.dropna()
            self.debug("Instances with null values removed")

        # Remove instances with zero volume or non-positive prices
        initial_count = len(data)
        data = data[
            (data['x'] > 0) & (data['y'] > 0) & (data['z'] > 0) & (data['price'] > 0)]
        removed_count = initial_count - len(data)
        if removed_count > 0:
            self.logger.debug(
                f"Removed {removed_count} instances with zero volume or non-positive prices")
        return data

    def _drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are not useful for the model."""
        columns_to_drop = ['depth', 'table', 'y', 'z']
        data = data.drop(columns=columns_to_drop)
        self.logger.debug(f"Dropped columns: {columns_to_drop}")
        return data
    
    def _convert_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical variables into dummy variables."""
        categorical_columns = ['cut', 'color', 'clarity']
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
        self.logger.debug(f"Converted categorical columns into dummies: {categorical_columns}")
        return data

class TrainManager:

    # Map of model names with corresponding classes and their default parameters
    MODEL_MAPPING = {
        "linear_regression": (LinearRegression, {}),
        "logistic_regression": (LogisticRegression, {}),
        "lasso_regression": (Lasso, {}),
        "ridge_regression": (Ridge, {}),
        "decision_tree": (DecisionTreeRegressor, {}),
        "svm": (SVR, {"kernel": 'rbf'}),
        "naive_bayes": (GaussianNB, {}),
        "knn": (KNeighborsRegressor, {}),
        "random_forest": (RandomForestRegressor, {"n_estimators": 100}),
        "gradient_boosting": (GradientBoostingRegressor, {"max_depth": 3}),
        "ada_boosting": (AdaBoostRegressor, {})
    }

    METRIC_MAPPING = {
        "mean_absolute_error": (mean_absolute_error, {}),
        "mean_squared_error": (mean_squared_error, {}),
        "root_mean_squared_error": (mean_squared_error, {'squared': False}),
        "mean_absolute_percentage_error": (mean_absolute_percentage_error, {}),
        "r2_score": (r2_score, {}),
        "explained_variance_score": (explained_variance_score, {})
    }

    def __init__(self,
                 trining_config_file: Path,
                 data_config_file: Path,
                 logger: Logger) -> None:
        self.configuration = FileUtils.retrieve_config(trining_config_file,
                                                       default_conf=DEFAULT_TRAIN_CONF)
        self.logger = logger

        # Data Loading Manager
        self.dm = DataManager(data_config_file, logger)

        self.training_uuid = str(uuid.uuid4())
        self.history = {
            "id_": self.training_uuid
            }
        self.logger.info(f"Training instance {self.training_uuid}")

    def data_generation(self) -> List[pd.Series]:
        data, dataframe_name = self.dm.fetch_data()
        to_process = get_config_value(self.configuration,
                                      ['training', 'processingData'],
                                      DEFAULT_TRAIN_CONF)
        if to_process:
            data = self.dm.preprocess_data(data)

        # Separate features and target variable
        X = data.drop(columns='price')
        y = data['price']

        # Apply the transformation specified in configuration
        transformation = get_config_value(self.configuration,
                                          ['training', 'transformation'],
                                          DEFAULT_TRAIN_CONF)
        try:
            y = transform_data(y, transformation)
            self.logger.info(
                f"{snake_to_camel(transformation)} transformation was applied to the data")
        except Exception as e:
            self.logger.error(f"The transformation {snake_to_camel(transformation)} "
                              f"could not be applied to the target data: {e}")

        # Get the test size from configuration
        split_size = get_config_value(self.configuration,
                                      ['training', 'testSize'],
                                      DEFAULT_TRAIN_CONF)
        self.logger.debug(f"Using test size: {split_size}")

        self.logger.info("Data processing operation successfully completed.")

        # Update of current training history
        self.history['ts'] = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        self.history['data'] = dataframe_name
        self.history['transformation'] = transformation
        self.history['splitSize'] = split_size

        return train_test_split(X, y, test_size=split_size, random_state=42)

    # 3. Model Training
    def train_model(self) -> None:
        X_train, X_val, y_train, y_val = self.data_generation()

        model_name = get_config_value(self.configuration,
                                      ['training', 'model', 'name'],
                                      DEFAULT_TRAIN_CONF)
        model_params = get_config_value(self.configuration,
                                        ['training', 'model', 'parameters'],
                                        DEFAULT_TRAIN_CONF)
        self.logger.debug(f"For {self.training_uuid} training, the "
                          f"{snake_to_camel(model_name)} model is used.")

        model_class, default_params = self.MODEL_MAPPING[model_name]

        # Merge default parameters with specified parameters
        if model_params is None:
            model_params = {}
        params = {**default_params, **model_params}

        # Training the model
        self.logger.info("Start the model training")
        model = model_class(**params)
        start_train = time.time()
        model.fit(X_train, y_train)
        delta_train = time.time() - start_train
        self.logger.debug(f"The training lasted for {delta_train} seconds")

        # Evaluate the trained model
        self.logger.info("Evaluation of model training")
        evaluation_results = self.evaluate_model(model, X_val, y_val)
        self.logger.info(f"MAE: {evaluation_results['mean_absolute_error']}")

        # Update of current training history
        self.history['evaluation_metrics'] = evaluation_results

        self.save_training(model, results=self.history)

    # 4. Model Evaluation
    def evaluate_model(self, model: Any, X_val: pd.Series, y_val: pd.Series) -> dict:
        y_pred = model.predict(X_val)

        metrics = get_config_value(self.configuration,
                                   ['training', 'metrics'],
                                   DEFAULT_TRAIN_CONF)

        if "mean_absolute_error" not in metrics:
            # The MAE is always calculated
            metrics.append("mean_absolute_error")

        results = {}
        for metric_name in metrics:
            if metric_name not in self.METRIC_MAPPING:
                continue
            metric_func, params = self.METRIC_MAPPING[metric_name]
            results[metric_name] = metric_func(y_val, y_pred, **params)
        return results
    
    def save_training(self, model: Any, results: dict) -> None:
        # Read previous results
        stored_history = {'training': []}
        train_folder = BASE_PATH.joinpath('train')
        train_result = train_folder.joinpath('results.json')
        if not train_folder.exists():
            train_folder.mkdir(parents=False)
        elif train_result.exists():
            with open(train_result) as f:
                stored_history = json.load(f)

        # Update current results with past results and saves the new history
        stored_history['training'].append(results)
        with open(train_result, 'w', encoding='utf-8') as f:
            json.dump(stored_history, f, ensure_ascii=False, indent=4)

        # Saving the trained model
        models_folder = train_folder.joinpath('models')
        if not models_folder.exists():
            models_folder.mkdir(parents=False)
        model_name = f'model_{self.training_uuid}'
        joblib.dump(model, f'{models_folder}/{model_name}.pkl')

# 5. Automated Pipeline
def main():

    # logger_config_path = BASE_PATH.joinpath("config/logger.ini")
    # logger_file = BASE_PATH.joinpath('log/training.log')
    training_config_file = BASE_PATH.joinpath("config/train_config.json")
    data_config_file = BASE_PATH.joinpath("config/data_config.json")
    logger_manager = LoggerManager(config_path=training_config_file)
    logger = logger_manager.logger

    tm = TrainManager(trining_config_file=training_config_file,
                      data_config_file=data_config_file,
                      logger=logger)
    tm.train_model()

if __name__ == "__main__":
    main()
