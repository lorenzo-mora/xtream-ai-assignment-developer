import json
import logging
import logging.config
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from typing import Any, Dict

DEFAULT_DATA_CONF = {
    "data": {
        "onlineSource": True,
        "url": "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv",
        "localPath": "data/diamonds.csv",
        "saveData": False
    }
}

DEFAULT_TRAIN_CONF = {
    "training": {
        "testSize": 0.2,
        "transformation": None,
        "processingData": True,
        "model": {
            "name": "linear_regression",
            "parameters": {},
            "save_path": ""
        },
        "metrics":[
            "r2_score",
            "mean_absolute_error"
        ]
    }
}

def retrieve_config(configuration_path: Path, default_conf: dict) -> dict:
    """Retrieve the configuration from the given path. If the
    configuration file does not exist at the specified path, the default
    configuration is returned.
    """
    if not configuration_path.exists():
        configuration = default_conf
    else:
        with open(configuration_path) as f:
            configuration = json.load(f)
    return configuration

def get_config_value(configuration: Dict[str, Any],
                     keys: list,
                     default_conf: Dict[str, Any]) -> Any:
    """Retrieve a nested configuration value from a given configuration
    dictionary.

    If the specified keys do not exist in the configuration, the
    function returns the corresponding value from the default configuration.

    Parameters
    ----------
    `configuration` : dict
        The configuration dictionary from which to retrieve the value.
    `keys` : list
        A list of keys representing the path to the desired value in the
        configuration dictionary.
    `default_conf` : dict, optional
        The default configuration dictionary to use if the specified
        keys do not exist in the main configuration.

    Returns
    -------
    Any
        The value from the configuration dictionary corresponding to the
        specified keys. If the keys do not exist, returns the value from
        the default configuration.

    Examples
    --------
    >>> data_configuration = {
    ...     "data": {
    ...         "onlineSource": True,
    ...         "localPath": "data/diamonds.csv",
    ...         "saveData": False
    ...     }
    ... }
    >>> keys = ['data', 'saveData']
    >>> get_config_value(data_configuration, keys)
    False
    >>> keys = ['data', 'url']
    >>> get_config_value(data_configuration, keys, {'data': {'url': 'https://diamonds.csv'}})
    'https://diamonds.csv'
    """
    conf_value = configuration
    default_value = default_conf

    for key in keys:
        conf_value = conf_value.get(key, {})
        default_value = default_value.get(key, {})

    return conf_value if conf_value else default_value

def transform_data(data: pd.Series, transformation: str) -> pd.Series:
    """Apply a specified transformation to a pandas Series.

    Parameters
    ----------
    `data` : pd.Series
        The data to transform.
    `transformation` : str
        The name of the transformation to apply. If None or an empty
        string, the original data is returned. Available transformations
        are:
        - 'log'
        - 'standardisation'
        - 'min-max scaling'
        - 'power'
        - 'square root'
        - 'exponential'
        - 'tanh'
        - 'z-score'
        - 'johnson'

    Returns
    -------
    pd.Series
        The transformed data as a pandas Series.

    Raises
    ------
    ValueError
        If the specified transformation is not recognized.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> transform_data(data, 'log')
    0    0.000000
    1    0.693147
    2    1.098612
    3    1.386294
    4    1.609438
    dtype: float64
    >>> transform_data(data, 'standardisation')
    0   -1.414214
    1   -0.707107
    2    0.000000
    3    0.707107
    4    1.414214
    dtype: float64
    >>> transform_data(data, 'min-max scaling')
    0    0.00
    1    0.25
    2    0.50
    3    0.75
    4    1.00
    dtype: float64
    >>> transform_data(data, 'nonexistent')
    Traceback (most recent call last):
        ...
    ValueError: Unknown transformation 'nonexistent'
    """
    if transformation is None or transformation == "":
        return data  # Return original data if no transformation is specified

    transformations = {
        "log": lambda x: np.log(x),
        "standardisation": lambda x: StandardScaler().fit_transform(
            x.values.reshape(-1, 1)).flatten(),
        "min_max_scaling": lambda x: MinMaxScaler().fit_transform(
            x.values.reshape(-1, 1)).flatten(),
        "power": lambda x: PowerTransformer(method='box-cox').fit_transform(
            x.values.reshape(-1, 1)).flatten(),
        "square_root": lambda x: np.sqrt(x),
        "exponential": lambda x: np.exp(x),
        "tanh": lambda x: np.tanh(x),
        "z_score": lambda x: zscore(x),
        "johnson": lambda x: QuantileTransformer(output_distribution='normal').fit_transform(
            x.values.reshape(-1, 1)).flatten()
    }

    if transformation in transformations:
        return pd.Series(transformations[transformation](data), index=data.index)
    else:
        raise ValueError(f"Unknown transformation '{transformation}'")

def snake_to_camel(snake_str: str) -> str:
    """Convert a snake_case string to Camel Case.
    
    Examples
    --------
    >>> snake_to_camel('hello_world')
    'Hello World'
    >>> snake_to_camel('convert_this_string')
    'Convert This String'
    """
    capitalized_words = [word.capitalize() for word in snake_str.split('_')]
    return ' '.join(capitalized_words)

class LoggerManager:
    def __init__(self, config_file: Path, log_file: Path):
        self.log_file = log_file

        # Check if the configuration file exists or raise an error
        if not config_file.exists():
            raise FileNotFoundError(f"The configuration file '{config_file}' does not exist.")

        self._configure_logging(config_file)

    def _configure_logging(self, config_file: Path):
        # Load logging configuration from file, pass log_file path as default
        logging.config.fileConfig(config_file, defaults={'filepath': str(self.log_file)})

    def get_logger(self, logger_name='root'):
        return logging.getLogger(logger_name)
