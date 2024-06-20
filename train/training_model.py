from datetime import datetime
import json
from logging import Logger
from pathlib import Path
import pickle
import tempfile
import time
from typing import List, Tuple
import uuid
from matplotlib import pyplot as plt
import optuna
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.svm import SVR
import joblib
import requests
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from train.utils import (DEFAULT_TRAIN_CONF, DEFAULT_DATA_CONF, BASE_PATH,
                         LoggerManager, FileUtils, snake_to_camel, get_config_value,
                         inverse_transform_data, transform_data)

CONFIG_PATH = BASE_PATH.joinpath("config")
DATA_PATH = BASE_PATH.joinpath("data")
OUTPUT_PATH = BASE_PATH.joinpath("output")

class DataManager:
    def __init__(self, config_file: Path, logger: Logger) -> None:
        self.configuration = FileUtils.retrieve_config(config_file,
                                                       default_conf=DEFAULT_DATA_CONF)
        self.logger = logger

    # 1. Data Acquisition
    def fetch_data(self) -> Tuple[pd.DataFrame, str]:
        self.fetch_from_url = get_config_value(self.configuration,
                                               ['data', 'onlineSource'],
                                               DEFAULT_DATA_CONF)
        if self.fetch_from_url:
            # The data are to be downloaded online
            source_path = get_config_value(self.configuration,
                                           ['data', 'url'],
                                           DEFAULT_DATA_CONF)
        else:
            # Data are to be retrieved from local
            source_name = get_config_value(self.configuration,
                                           ['data', 'localPath'],
                                           DEFAULT_DATA_CONF)
            source_path = DATA_PATH.joinpath(source_name)
        self.logger.info(f"Acquiring data from '{source_path}'")

        data = (self._fetch_from_online(url=source_path)
                if self.fetch_from_url else self._fetch_from_local(path_to_data=source_path))

        return data, str(source_path).rsplit('/', 1)[-1]

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

        # Retrieve operations from config
        self.is_2b_clean = get_config_value(self.configuration,
                                       ['preparation', 'cleanData', 'active'],
                                       DEFAULT_DATA_CONF)
        self.is_2b_reduced = get_config_value(self.configuration,
                                         ['preparation', 'dropColumns', 'active'],
                                         DEFAULT_DATA_CONF)
        self.is_2b_converted_dummy = get_config_value(self.configuration,
                                                 ['preparation', 'toDummy', 'active'],
                                                 DEFAULT_DATA_CONF)
        self.is_2b_converted_ordinal = get_config_value(self.configuration,
                                                   ['preparation', 'toOrdinal', 'active'],
                                                   DEFAULT_DATA_CONF)

        # Apply the operations specified in the configuration
        processed_data = original_data.copy()
        if self.is_2b_clean:
            processed_data = self._clean_data(processed_data)
        if self.is_2b_reduced:
            self.drop_columns = get_config_value(
                self.configuration, ['preparation', 'dropColumns', 'columns'],
                DEFAULT_DATA_CONF)
            processed_data = self._drop_columns(processed_data,
                                                columns_to_drop=self.drop_columns)
        if self.is_2b_converted_dummy:
            convert_to_dummies = get_config_value(
                self.configuration, ['preparation', 'toDummy', 'columns_categories'],
                DEFAULT_DATA_CONF)
            processed_data = self._categorical_to_dummy(
                processed_data, columns_categories=convert_to_dummies)
            self.logger.debug(
                f"Converted categorical columns into dummies: {list(convert_to_dummies.keys())}")
        if self.is_2b_converted_ordinal:
            convert_to_ordinal = get_config_value(
                self.configuration, ['preparation', 'toOrdinal', 'columns_categories'],
                DEFAULT_DATA_CONF)
            processed_data = DataManager._categorical_to_ordinal(
                processed_data, columns_categories=convert_to_ordinal)
            self.logger.debug(
                f"Converted categorical columns into ordinal: {list(convert_to_ordinal.keys())}")

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

    def _drop_columns(self, data: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
        """Drop columns that are not useful for the model."""
        data = data.drop(columns=columns_to_drop)
        self.logger.debug(f"Dropped columns: {columns_to_drop}")
        return data

    @staticmethod
    def _categorical_to_dummy(data: pd.DataFrame,
                              columns_categories: list) -> pd.DataFrame:
        # """Convert categorical variables into dummy variables."""
        # data = pd.get_dummies(data, columns=columns_categories, drop_first=True)
        """Convert specified columns in a dataframe to dummy/one-hot
        encoded format based on given dictionary, in which keys are
        column names to be converted and values are lists of possible
        values.
        """
        for column, values in columns_categories.items():
            # Create dummy variables for the specified values
            for value in values:
                data[f"{column}_{value}"] = (data[column].str.upper() == value).astype(int)
            # Drop the original column if needed
            data = data.drop(column, axis=1)
        return data

    @staticmethod
    def _categorical_to_ordinal(data: pd.DataFrame,
                                columns_categories: dict) -> pd.DataFrame:
        """Convert categorical variables into ordinal variables."""
        for col, cats in columns_categories.items():
            data[col] = pd.Categorical(data[col].str.upper(), categories=cats, ordered=True)
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
        "gradient_boosting": (GradientBoostingRegressor, {}),
        "xgb_regression": (XGBRegressor, {}),
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
                 training_config_file: Path,
                 data_config_file: Path,
                 logger: Logger) -> None:
        """Initializes the TrainManager instance.

        Parameters
        ----------
        `trining_config_file` : Path
            Path to the training configuration file.
        `data_config_file` : Path
            Path to the data configuration file.
        `logger` : Logger
            Logger instance for logging information.
        """
        self.configuration = FileUtils.retrieve_config(training_config_file,
                                                       default_conf=DEFAULT_TRAIN_CONF)
        self.logger = logger

        # Data Loading Manager
        self.dm = DataManager(data_config_file, logger)

        self.training_uuid = str(uuid.uuid4())
        self.history = {
            "id": self.training_uuid
            }
        self.logger.info(f"Training instance {self.training_uuid}")

        self._retrieve_configuration()

        self.X_train, self.X_val, self.y_train, self.y_val = self._data_generation()

    def run(self, model_parameters: dict = {}):
        """Runs the model training process.

        Parameters
        ----------
        `model_parameters` : dict, optional
            Dictionary of model parameters to override default settings.
        """
        self.logger.info("Performing model training")
        results = self._train_model(model_parameters)

        # Update of current training history
        self.history['modelType'] = snake_to_camel(self.model_name)
        self.history['parameters'] = self.model_params
        self.history['evaluationMetrics'] = results

        self._save_training()

    def hyperparameters_search(self):
        """Performs hyperparameter optimization using Optuna and trains
        the model with the best found parameters."""
        self.logger.info(
            f"A Bayesian tuning of {self.number_of_trials} trials of the "
            f"model's hyperparameters is performed: {self.study_name}")
        start_study = time.time()
        study = optuna.create_study(direction='minimize',
                                    study_name=self.study_name)
        study.optimize(self._objective, n_trials=self.number_of_trials)
        delta_study = time.time() - start_study
        self.logger.debug(f"Bayesian research lasted {round(delta_study, 4)} seconds")

        self.logger.debug(f"The optimal hyperparameters are: {study.best_params}")
        self.run(model_parameters=study.best_params)

    def _retrieve_configuration(self) -> None:
        """Retrieves the model configuration from the configuration file."""

        self.logger.info(
            f"Retrieving the model configuration for training {self.training_uuid}")

        # Name of the class model
        self.model_name = get_config_value(
            self.configuration, ['training', 'model', 'name'], DEFAULT_TRAIN_CONF)
        self.logger.debug(
            f"The model to be trained is: {snake_to_camel(self.model_name)}")

        # Training parameters
        self.config_model_params = get_config_value(
            self.configuration, ['training', 'model', 'parameters'], DEFAULT_TRAIN_CONF)

        # If it is necessary to process data
        self.to_process = get_config_value(
            self.configuration, ['training', 'processingData'], DEFAULT_TRAIN_CONF)

        # Test size for the training
        self.split_size = get_config_value(
            self.configuration, ['training', 'testSize'], DEFAULT_TRAIN_CONF)
        self.logger.debug(f"Using test size: {self.split_size}")

        # Random state for the training
        self.random_state = get_config_value(
            self.configuration, ['training', 'randomState'], DEFAULT_TRAIN_CONF)

        # Training data transformation
        self.transformation = get_config_value(
            self.configuration, ['training', 'transformation'], DEFAULT_TRAIN_CONF)

        # Training evaluation metrics
        self.metrics = get_config_value(
            self.configuration, ['training', 'metrics'], DEFAULT_TRAIN_CONF)
        
        # If the hyperparameters optimisation is required
        self.optimize_model = get_config_value(
            self.configuration, ['training', 'tuning', 'active'], DEFAULT_TRAIN_CONF)
        if self.optimize_model:
            # The number of trials to be carried out for the search
            self.number_of_trials = get_config_value(
                self.configuration, ['training', 'tuning', 'trialsNumber'], DEFAULT_TRAIN_CONF)
            # The number of trials to be carried out for the search
            self.study_name = get_config_value(
                self.configuration, ['training', 'tuning', 'studyName'], DEFAULT_TRAIN_CONF)

    def _data_generation(self) -> List[pd.Series]:
        """Generates training and validation datasets.

        Returns
        -------
        List[pd.Series]
            A list containing training and validation data splits:
            [X_train, X_val, y_train, y_val]
        """
        data, dataframe_name = self.dm.fetch_data()

        if self.to_process:
            data = self.dm.preprocess_data(data)

        # Separate features and target variable
        X = data.drop(columns='price')
        y = data['price']

        self.logger.info("Data processing operation successfully completed.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.split_size, random_state=self.random_state)

        # Apply the transformation specified in configuration
        transformation_name = snake_to_camel(self.transformation)
        try:
            y_train, self.fitted_parameters = transform_data(y_train,
                                                             self.transformation)
            info_msg = "{} transformation was applied to the data".format(
                transformation_name if self.transformation else 'No')
            self.logger.info(info_msg)
        except Exception as e:
            self.logger.error(f"The transformation {transformation_name} "
                              f"could not be applied to the target data: {e}")

        # Saving the fitted transformer model
        if (self.fitted_parameters is not None and
            (any(isinstance(self.fitted_parameters, t_class)
                 for t_class in [StandardScaler, MinMaxScaler,
                                 PowerTransformer, QuantileTransformer]) or
             self.transformation == "z_score")):
            dest_path = OUTPUT_PATH.joinpath("transformer")
            model_name = f"transformer_{self.training_uuid}"
            joblib.dump(self.fitted_parameters, f'{dest_path}/{model_name}.pkl')

        # Update of current training history
        self.history['ts'] = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        self.history['data'] = {
            'name': dataframe_name,
            'onlineSource': self.dm.fetch_from_url,
            'clean': self.dm.is_2b_clean,
            'reduced': self.dm.drop_columns if self.dm.is_2b_reduced else self.dm.is_2b_reduced,
            'dummy': self.dm.is_2b_converted_dummy,
            'ordinal': self.dm.is_2b_converted_ordinal
            }
        self.history['transformation'] = transformation_name
        self.history['splitSize'] = self.split_size

        return X_train, X_val, y_train, y_val

    # 3. Model Training
    def _train_model(self, model_parameters: dict = {}) -> dict:
        """Trains the model with the given parameters.

        Parameters
        ----------
        `model_parameters` : dict, optional
            Dictionary of model parameters to override default settings.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics.
        """

        model_class, default_params = self.MODEL_MAPPING[self.model_name]

        # Merge the input parameters with those specified in the file
        # and the default parameters
        self.model_params = {**default_params,
                             **self.config_model_params,
                             **model_parameters}

        # Training the model
        self.logger.info(
            "Starting the model training with configuration: {}".format(
                self.model_params if self.model_params else 'standard'))
        self.model = model_class(**self.model_params)
        start_train = time.time()
        self.model.fit(self.X_train, self.y_train)
        delta_train = time.time() - start_train
        self.logger.debug(f"The training lasted for {round(delta_train, 4)} seconds")

        # Evaluate the trained model
        evaluation_results = self._evaluate_model()
        self.logger.info(f"MAE: {evaluation_results['mean_absolute_error']}")

        return evaluation_results

    # 4. Model Evaluation
    def _evaluate_model(self) -> dict:
        """Evaluates the trained model using the validation dataset.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics.
        """
        self.y_pred = self.model.predict(self.X_val)
        self.y_pred = inverse_transform_data(pd.Series(self.y_pred),
                                             self.transformation,
                                             self.fitted_parameters)

        if "mean_absolute_error" not in self.metrics:
            # The MAE is always calculated
            self.metrics.append("mean_absolute_error")

        results = {}
        for metric_name in self.metrics:
            if metric_name not in self.METRIC_MAPPING:
                continue
            metric_func, params = self.METRIC_MAPPING[metric_name]
            results[metric_name] = metric_func(self.y_val, self.y_pred, **params)
        return results

    def plot_gof(self, filename: str) -> None:
        """It generates the goodness-of-fit graph and saves it with the
        name `filename`."""
        plt.plot(self.y_val, self.y_pred, '.')
        plt.plot(self.y_val, self.y_val, linewidth=3, c='black')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        # Save the plot
        file_path = OUTPUT_PATH.joinpath(f'images/{filename}.png')
        plt.savefig(file_path, bbox_inches='tight')

    def _save_training(self) -> None:
        """Saves the training history and the trained model to files."""
        # Read previous results
        stored_history = {'training': []}
        train_result = OUTPUT_PATH.joinpath('results.json')
        if not OUTPUT_PATH.exists():
            OUTPUT_PATH.mkdir(parents=False)
        elif train_result.exists():
            with open(train_result) as f:
                stored_history = json.load(f)

        # Update current results with past results and saves the new history
        stored_history['training'].append(self.history)
        with open(train_result, 'w', encoding='utf-8') as f:
            json.dump(stored_history, f, ensure_ascii=False, indent=4)

        # Saving the trained model
        models_folder = OUTPUT_PATH.joinpath('models')
        if not models_folder.exists():
            models_folder.mkdir(parents=False)
        model_name = f'model_{self.training_uuid}'
        with open(f'{models_folder}/{model_name}.pkl','wb') as f:
            pickle.dump(self.model, f)

        # Elaboration of the GOF plot
        self.plot_gof(filename=model_name)
        self.logger.info("Saved all data")

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function for hyperparameter optimization.

        Parameters
        ----------
        `trial` : optuna.trial.Trial
            Optuna trial object to sample hyperparameters.

        Returns
        -------
        float
            The mean absolute error (MAE) of the model for the given trial.
        """
        epoch_params = TrainManager.process_parameters(self.config_model_params, trial)
        results = self._train_model(epoch_params)
        return results['mean_absolute_error']

    @staticmethod
    def process_parameters(params: dict, trial: optuna.trial.Trial) -> dict:
        """Processes hyperparameters for the given trial.

        Parameters
        ----------
        `params` : dict
            Dictionary of parameter names and their corresponding
            suggestion strings.
        `trial` : optuna.trial.Trial
            Optuna trial object to sample hyperparameters.

        Returns
        -------
        dict
            A dictionary of processed hyperparameters.
        """
        processed_params = {}
        for key, value in params.items():
            if (isinstance(value, str) and
                (value.startswith("self.trial.") or
                 value.startswith("trial."))):
                method_call = value.replace("self.trial.", "trial.")
                processed_params[key] = eval(method_call)
            else:
                processed_params[key] = value
        return processed_params

# 5. Automated Pipeline
def main(training_config_file: Path = CONFIG_PATH.joinpath("train_config.json"),
         data_config_file: Path = CONFIG_PATH.joinpath("data_config.json")):

    logger_manager = LoggerManager(config_path=training_config_file)
    logger = logger_manager.logger

    tm = TrainManager(training_config_file=training_config_file,
                      data_config_file=data_config_file,
                      logger=logger)

    configuration = FileUtils.retrieve_config(training_config_file,
                                              default_conf=DEFAULT_TRAIN_CONF)
    # If Bayesian optimisation is specified in the configuration
    # (`tuning.active = True`), then the TrainManager.grid_search()
    # method is executed
    is_optimisation = get_config_value(configuration,
                                       ['training', 'tuning', 'active'],
                                       DEFAULT_TRAIN_CONF)
    if is_optimisation:
        tm.hyperparameters_search()
    else:
        # Otherwise, single model training is performed
        tm.run()

if __name__ == "__main__":
    main()
