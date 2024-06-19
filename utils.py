import datetime as dt
from datetime import datetime, timedelta
from io import BytesIO
import json
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
import tarfile
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from typing import Any, Dict, List, Optional, Tuple

COLUMNS_CATEGORIES = {
    "cut": ["FAIR", "GOOD", "VERY GOOD", "IDEAL", "PREMIUM"],
    "color": ["D", "E", "F", "G", "H", "I", "J"],
    "clarity": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
}

DEFAULT_DATA_CONF = {
    "data": {
        "onlineSource": True,
        "url": "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv",
        "localPath": "diamonds.csv",
        "saveData": False
    },
    "preparation": {
        "cleanData": {
            "active": False
        },
        "dropColumns": {
            "active": False,
            "columns": [
                "depth",
                "table",
                "y",
                "z"
            ]
        },
        "toDummy": {
            "active": False,
            "columns_categories": COLUMNS_CATEGORIES
        },
        "toOrdinal": {
            "active": False,
            "columns_categories": COLUMNS_CATEGORIES
        }
    }
}

DEFAULT_TRAIN_CONF = {
    "training": {
        "testSize": 0.2,
        "randomState": None,
        "transformation": None,
        "processingData": True,
        "model": {
            "name": "linear_regression",
            "parameters": {}
        },
        "metrics":[
            "r2_score",
            "mean_absolute_error"
        ],
        "tuning": {
            "active": False,
            "trialsNumber": 100,
            "studyName": "Diamonds Linear Regression"
        }
    },
    "logger": {
        "configName": "logger.ini",
        "name": "train.log",
        "level": "DEBUG",
        "maxFiles": 10,
        "rotateAtTime": "23:59",
        "formatMessage": "%(asctime)s - [%(name)s:%(filename)s:%(lineno)d] - %(levelname)s: %(message)s",
        "formatDate": "%Y-%m-%d %H:%M:%S"
    }
}

def get_config_value(configuration: Dict[str, Any],
                     keys: List[str],
                     default_conf: Dict[str, Any] = {}) -> Any:
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
        if isinstance(conf_value, dict) and key in conf_value:
            conf_value = conf_value[key]
        else:
            # If key does not exist in the main configuration, switch to the default configuration path
            return default_conf.get(key, default_value.get(key))
        
        default_value = default_value.get(key, {})

    return conf_value

def transform_data(data: pd.Series,
                   transformation: str) -> Tuple[pd.Series, Any]:
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
    tuple
        The transformed data as a pandas Series and the fitted
        parameters required for the inverse transformation.

    Raises
    ------
    ValueError
        If the specified transformation is not recognized.
    """
    if transformation is None or transformation == "":
        return data, None  # Return original data if no transformation is specified

    transformers = {
        "log": lambda x: (np.log(x), None),
        "standardisation": lambda x: (
            StandardScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
            StandardScaler().fit(x.values.reshape(-1, 1))
        ),
        "min_max_scaling": lambda x: (
            MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
            MinMaxScaler().fit(x.values.reshape(-1, 1))
        ),
        "power": lambda x: (
            PowerTransformer(method='box-cox').fit_transform(x.values.reshape(-1, 1)).flatten(),
            PowerTransformer(method='box-cox').fit(x.values.reshape(-1, 1)) # Some issues
        ),
        "square_root": lambda x: (np.sqrt(x), None),
        "exponential": lambda x: (np.exp(x), None), # Some issues
        "tanh": lambda x: (np.tanh(x), None), # Some issues
        "z_score": lambda x: (zscore(x), (np.mean(x), np.std(x))),
        "johnson": lambda x: (
            QuantileTransformer(output_distribution='normal').fit_transform(x.values.reshape(-1, 1)).flatten(),
            QuantileTransformer(output_distribution='normal').fit(x.values.reshape(-1, 1))
        )
    }

    if transformation in transformers:
        transformed_data, fitted_params = transformers[transformation](data)
        return pd.Series(transformed_data, index=data.index), fitted_params
    else:
        raise ValueError(f"Unknown transformation '{transformation}'")

def inverse_transform_data(data: pd.Series,
                           transformation: str,
                           fitted_params: Any) -> pd.Series:
    """Apply the inverse of a specified transformation to a pandas Series."""
    inverse_transformers = {
        "log": lambda x: np.exp(x),
        "standardisation": lambda x: fitted_params.inverse_transform(x.values.reshape(-1, 1)).flatten(),
        "min_max_scaling": lambda x: fitted_params.inverse_transform(x.values.reshape(-1, 1)).flatten(),
        "power": lambda x: inv_boxcox(x, fitted_params.lambdas_),
        "square_root": lambda x: np.square(x),
        "exponential": lambda x: np.log(x),
        "tanh": lambda x: np.arctanh(x),
        "z_score": lambda x: x * fitted_params[1] + fitted_params[0],
        "johnson": lambda x: fitted_params.inverse_transform(x.values.reshape(-1, 1)).flatten()
    }

    if transformation is None or transformation == "":
        return data

    if transformation in inverse_transformers:
        return pd.Series(inverse_transformers[transformation](data), index=data.index)
    else:
        raise ValueError(f"Unknown transformation '{transformation}'")

def snake_to_camel(snake_str: Optional[str]) -> Optional[str]:
    """Convert a snake_case string to Camel Case.
    
    Examples
    --------
    >>> snake_to_camel('hello_world')
    'Hello World'
    >>> snake_to_camel('convert_this_string')
    'Convert This String'
    """
    if not snake_str:
        return snake_str
    capitalized_words = [word.capitalize() for word in snake_str.split('_')]
    return ' '.join(capitalized_words)

class FileUtils:
    @staticmethod
    def resize_tar(archive_path: Path,
                   num_new_files: int,
                   archive_file_num_limit: int) -> None:
        """Reduce the size of the archive by removing the oldest
        files if the total number of files, including new ones, exceeds
        the specified limit.

        Parameters
        ----------
        `archive_path` : Path
            The path to the archive to be managed.
        `num_new_files` : int
            The number of new files to be added to the archive.
        `archive_file_num_limit` : int
            The maximum allowed number of files for the archive.

        Raises
        ------
        tarfile.ReadError
            If there is an error with the tar file operation.
        """
        try:
            if not archive_path.is_file():
                return

            # Open the archive to check the current number of files
            with tarfile.open(archive_path, "r:gz") as tar:
                file_names = tar.getnames()
                num_files = len(file_names) + num_new_files

                if num_files < archive_file_num_limit:
                    return

                while num_files >= archive_file_num_limit:
                    # Get the oldest file in the archive and remove it from
                    # the archive
                    oldest_file = min(file_names,
                                      key=lambda x: tar.getmember(x).mtime)
                    tar.extract(oldest_file)
                    file_names.remove(oldest_file)
                    num_files -= 1
        except tarfile.ReadError as e:
            raise tarfile.ReadError(f"Error resizing archive: {e}") from e

    @staticmethod
    def append_file_to_tar(file_path: Path,
                           archive_path: Path,
                           replace: bool = True) -> None:
        """Append a file to an existing tar gz archive if it is not
        already present, or if `replace` is True, replace the existing
        file with the same name.

        Parameters
        ----------
        `file_path` : str
            The path to the file to be added to the archive.
        `archive_path` : Path
            The path to the archive to which the new file should be
            appended.
        `replace` : bool, optional
            Whether to replace the file if already present in the
            archive. By default True.

        Raises
        ------
        tarfile.TarError
            If there is an error with the tar file operation.
        FileNotFoundError
            If the file specified by `file_path` does not exist.
        PermissionError
            If permission is denied while accessing files or directories.
        """
        try:
            if not archive_path.is_file():
                return

            file_name = file_path.name
            with tempfile.TemporaryDirectory(dir=file_path.parent) as tempdir:
                tmp_path = Path(tempdir).joinpath('{}.gz'.format(archive_path.stem))

                with tarfile.open(archive_path, "r:gz") as tar:
                    if (not replace and
                        any(member.name == file_name for member in tar.getmembers())):
                        return

                    # Read the file to be added
                    with open(file_path, "rb") as fh:
                        file_data = BytesIO(fh.read())
                    tarinfo = tarfile.TarInfo(file_name)
                    tarinfo.size = len(file_data.getvalue())

                    with tarfile.open(tmp_path, "w:gz") as tmp:
                        for member in tar:
                            if member.name != file_name:
                                tmp.addfile(member, tar.extractfile(member.name))
                        tmp.addfile(tarinfo, file_data)
                        file_path.unlink()
                tmp_path.rename(archive_path)
        except (tarfile.TarError, FileNotFoundError, PermissionError) as e:
            raise type(e)(f"Error appending file to archive: {e}") from e

    @staticmethod
    def retrieve_config(file_path: Path, default_conf: dict) -> dict:
        """Retrieve the configuration from the given path. If the
        configuration file does not exist at the specified path, the default
        configuration is returned.
        """
        if not file_path.exists():
            configuration = default_conf
        else:
            try:
                with open(file_path) as f:
                    configuration = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Error decoding JSON") from e
            except PermissionError as e:
                raise PermissionError(f"Permission denied") from e
        return configuration

    @staticmethod
    def read_json(file_path: Path, may_not_exist: bool = False) -> Optional[dict]:
        """Reads the JSON file specified by `file_path`.

        It returns the corresponding dictionary or None if the file does
        not exist or cannot be read.

        Parameters
        ----------
        `file_path` : Path
            The path to the json file.
        `may_not_exist` : bool, optional
            If the file to be read may not exist, by default False.

        Returns
        -------
        Optional[dict]
            The contents of the json file if it could be read, otherwise
            None.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        json.JSONDecodeError
            If there is an error decoding JSON data.
        PermissionError
            If the current user does not have permission to access the
            file.
        """
        try:
            with open(file_path, 'r') as json_file:
                return json.load(json_file)
        except FileNotFoundError as e:
            if may_not_exist:
                return None
            else:
                raise FileNotFoundError(f"File not found: {file_path}") from e
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON") from e
        except PermissionError as e:
            raise PermissionError(f"Permission denied") from e

class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Custom subclass of TimedRotatingFileHandler for log file rotation
    and compression.

    This class extends TimedRotatingFileHandler to perform log file
    rollover and compress old log files when a rollover occurs.
    
    Parameters
    ----------
    `filename` : str, optional
        The filename for the log file, by default "".
    `when` : str, optional
        Specifies the type of interval, by default "midnight".
    `interval` : int, optional
        Intervals at which rollover occurs, in `when` units. By default 1.
    `backupCount` : int, optional
        Number of backup files to keep, by default 14.
    `atTime` : datetime.time, optional
        Time of day when rollover occurs, by default None.
    `encoding` : str, optional
        The encoding used for the log file, by default "utf-8".

    Methods
    -------
    `doRollover()`
        Performs log file rollover and compresses old log files.
    """
    def __init__(self,
                 filename: str = "",
                 when: int = "midnight",
                 interval: int = 1,
                 backupCount: int = 14,
                 atTime: Optional[dt.time] = None,
                 encoding: str = "utf-8"):
        super().__init__(filename=filename,
                         when=when,
                         interval=interval,
                         backupCount=backupCount,
                         atTime=atTime,
                         encoding=encoding)

    def doRollover(self):
        """Performs log file rollover and compresses old log files into a
        tar archive.

        This method performs log file rollover by rotating log files
        based on the configured criteria and compresses old log files
        into a tar archive to maintain a maximum number of backup files.

        Upon rollover, the method identifies the eligible log files to
        compress based on the log file naming convention and excludes
        today's and yesterday's log files. It then checks if a tar
        archive already exists for older log files. If an archive exists
        and its file count plus the number of log files to compress
        exceed the configured maximum backup count, the method
        iteratively removes the oldest files from the archive until the
        total file count is within the limit. Finally, the method removes
        the most recent uncompressed log files after adding them to the
        archive.

        Note
        ----
        The maximum backup count determines the number of log files to
        retain in the tar archive.
        If the archive does not exist or is not yet populated with the
        maximum number of files, the method creates a new archive and
        adds log files to it.

        Raises
        ------
        tarfile.TarError
            If an error occurs during tar archive operations.
        FileNotFoundError
            If the archive or log file cannot be found.
        PermissionError
            If the method lacks permission to read or write to files or
            directories.
        """
        super().doRollover()
        log_dir = Path(self.baseFilename).parent
        base_filename = Path(self.baseFilename).stem

        # Get the current date and yesterday's date and format them as
        # YYYY-MM-DD
        current_date = datetime.now()
        yesterday_date = current_date - timedelta(days=1)
        formatted_current_date = current_date.strftime("%Y-%m-%d")
        formatted_yesterday_date = yesterday_date.strftime("%Y-%m-%d")

        # Get files to compress
        to_compress = [f for f in log_dir.iterdir()
                       if f.name.startswith(base_filename)
                       and f.suffix not in ['.gz', '.log',
                                            f'.{formatted_current_date}',
                                            f'.{formatted_yesterday_date}']]

        # Create a tar.gz file for older log files
        if to_compress:
            tar_filename = f"{base_filename}-log.tar.gz"
            archive_path = log_dir.joinpath(tar_filename)

            if archive_path.is_file():
                # If the current archive exists, check the current number
                # of files it contains and if this value, increased by
                # the number of files to be added to the archive, exceeds
                # the `backupCount`, remove the oldest one; then all new
                # files to be added
                FileUtils.resize_tar(archive_path=archive_path,
                                     num_new_files=len(to_compress),
                                     archive_file_num_limit=self.backupCount)
                for f in to_compress:
                    FileUtils.append_file_to_tar(
                        file_path=f, archive_path=archive_path, replace=True)
            else:
                # The archive does not exist, so the archive is
                # generated by adding only the most recent log file(s)
                with tarfile.open(archive_path, "w:gz") as tar:
                    for f in to_compress:
                        tar.add(f, arcname=f.name)
                        f.unlink()  # Remove uncompressed file after archiving

class LoggerManager:
    """Custom logger configuration class for initializing logger with
    specific parameters.

    This class provides functionality to initialize a logger with
    parameters loaded from a configuration file and customize log file
    path, rotation settings, and log message format.
    """
    
    def __init__(self, config_path: Path):
        self.logger = self._initialize_logger(config_path)

    def _initialize_logger(self, config_path: Path) -> logging.Logger:
        """Initialize the logger with parameters loaded from the
        configuration file."""
        base_path = Path(__file__).resolve().parent

        # Load configuration from config.ini
        config = FileUtils.retrieve_config(config_path,
                                           default_conf=DEFAULT_TRAIN_CONF)

        # Get parameters from config
        log_filename = get_config_value(config,
                                        ['logger', 'name'],
                                        DEFAULT_TRAIN_CONF)
        log_file_path = base_path.joinpath(f"log/{log_filename}")
        severity_level = get_config_value(config,
                                          ['logger', 'level'],
                                          DEFAULT_TRAIN_CONF)
        max_files = get_config_value(config,
                                     ['logger', 'maxFiles'],
                                     DEFAULT_TRAIN_CONF)
        rtime = get_config_value(config,
                                 ['logger', 'rotateAtTime'],
                                 DEFAULT_TRAIN_CONF)
        rotate_time = datetime.strptime(rtime, '%H:%M').time()
        msg_format = get_config_value(config,
                                      ['logger', 'formatMessage'],
                                      DEFAULT_TRAIN_CONF)
        date_format = get_config_value(config,
                                       ['logger', 'formatDate'],
                                       DEFAULT_TRAIN_CONF)

        # Load logging configuration from logger.ini
        config_filename = get_config_value(config,
                                           ['logger', 'configName'],
                                           DEFAULT_TRAIN_CONF)
        config_file_path = base_path.joinpath(f"config/{config_filename}")
        logging.config.fileConfig(config_file_path, disable_existing_loggers=False)

        # Get the logger with the name 'training'
        logger = logging.getLogger('training')
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

        # Set severity level
        logger.setLevel(severity_level)

        # Configure the file handler for rotating log files
        file_handler = CustomTimedRotatingFileHandler(
            log_file_path,
            when='midnight',
            interval=1,
            backupCount=max_files,
            atTime=rotate_time,
            encoding='utf-8'
        )

        # Configure the string formatter of the handler for saving the message
        file_formatter = logging.Formatter(fmt=msg_format, datefmt=date_format, style='%')
        file_handler.setFormatter(file_formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        return logger
