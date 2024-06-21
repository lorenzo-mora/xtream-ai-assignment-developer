from pathlib import Path
import pickle
import sys
from typing import Any, List, Optional, Tuple
from fastapi import HTTPException
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

BASE_PATH = Path(__file__).resolve().parents[1]
TRAINING_PATH = BASE_PATH.joinpath("train")
OUTPUT_PATH = BASE_PATH.joinpath("output")
MODELS_PATH = OUTPUT_PATH.joinpath("models")
DATA_PATH = BASE_PATH.joinpath("data")
CONFIG_PATH = BASE_PATH.joinpath("config")
sys.path.append(str(BASE_PATH))

from train.training_model import DataManager, main
from train.utils import FileUtils, COLUMNS_CATEGORIES, inverse_transform_data

def retrieve_metadata() -> List[dict]:
    """Retrieves the training information of all models."""
    try:
        models = FileUtils.read_json(OUTPUT_PATH.joinpath("results.json"),
                                     may_not_exist=False)
    except Exception:
        raise
    # Retrieve metadata of the model
    return models['training']

def retrieve_ids(metadata: List[dict]) -> List[str]:
    """Retrieves the list of all trained model ids."""
    return [m['id'] for m in metadata]

def check_features_values(request_body: dict,
                          metadata: Optional[List[dict]] = None) -> None:
    """Checks that the features passed as input are correct, throwing an
    HTTPException in the event of an error."""

    if metadata:
        # List of available models
        available_models = retrieve_ids(metadata)

        # Check that the required id is present among the existing models
        if 'model' in request_body and request_body['model'] not in available_models:
            raise HTTPException(status_code=400, detail=[
                    {
                        "loc": ["body", "model"],
                        "msg": "Model ID is not valid",
                        "type": "value_error.not_existing"
                    }
                ])

    # Check that the specified cut is correct
    if ('cut' in request_body and
        request_body['cut'].upper() not in ['FAIR', 'GOOD', 'VERY GOOD', 'IDEAL', 'PREMIUM']):
        raise HTTPException(status_code=400, detail=[
                {
                    "loc": ["body", "cut"],
                    "msg": "Cut is not valid",
                    "type": "value_error.not_existing"
                }
            ])

    # Check that the specified colour is correct
    if ('color' in request_body and
        request_body['color'].upper() not in ['D', 'E', 'F', 'G', 'H', 'I', 'J']):
        raise HTTPException(status_code=400, detail=[
                {
                    "loc": ["body", "color"],
                    "msg": "Color is not valid",
                    "type": "value_error.not_existing"
                }
            ])

    # Check that the specified clarity is correct
    if ('clarity' in request_body and
        request_body['clarity'].upper() not in ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']):
        raise HTTPException(status_code=400, detail=[
                {
                    "loc": ["body", "clarity"],
                    "msg": "Clarity is not valid",
                    "type": "value_error.not_existing"
                }
            ])

    # Check that all specified values are positive
    feature_names = ['carat', 'depth', 'table', 'x', 'y', 'z']
    for f in feature_names:
        if f in request_body and request_body[f] < 0:
            raise HTTPException(status_code=400, detail=[
                {
                    "loc": ["body", f],
                    "msg": f"{f.capitalize()} must be positive",
                    "type": "value_error.non_positive"
                }
            ])

    # Check that the specified method is correct
    if ('method' in request_body and
        request_body['method'].lower() not in [
            'absolute difference', 'relative difference', 'squared difference',
            'z score', 'cosine similarity']):
        raise HTTPException(status_code=400, detail=[
                {
                    "loc": ["body", "method"],
                    "msg": "Method is not valid",
                    "type": "value_error.not_existing"
                }
            ])

    # Check that the specified method is correct
    if ('n' in request_body and request_body['n'] < 1):
        raise HTTPException(status_code=400, detail=[
                {
                    "loc": ["body", "n"],
                    "msg": "N must be greater than 0",
                    "type": "value_error.non_positive"
                }
            ])

    # Check the existence of the specified dataset 'dataset_name'
    if ('dataset_name' in request_body and
        not DATA_PATH.joinpath(request_body['dataset_name']).exists()):
        raise HTTPException(status_code=400, detail=[
                {
                    "loc": ["body", "dataset_name"],
                    "msg": "The dataset does not exist",
                    "type": "value_error.not_existing"
                }
            ])

    # Consistency check of configuration file name/path parameters
    required_pairs = [
        ('training_config_name', 'data_config_name'),
        ('training_config_path', 'data_config_path')
    ]
    for pair in required_pairs:
        if ((pair[0] in request_body or pair[1] in request_body) and
            bool(request_body[pair[0]]) != bool(request_body[pair[1]])):
            raise HTTPException(status_code=400, detail=[
                    {
                        "loc": ["body", pair[0] if not bool(request_body[pair[0]]) else pair[1]],
                        "msg": "The parameters must be consistent",
                        "type": "value_error.parameter_inconsistency"
                    }
                ])

def load_model(model_id: str) -> Any:
    """Loads the model with id equal to `model_id`, which is required
    for prediction."""
    model_path = MODELS_PATH.joinpath(f"model_{model_id}.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_inverse_transformer(model_id: str, transformation: Optional[str]) -> Any:
    """Loads the parameters needed to apply the inverse transformation
    of the one used for training."""
    if (transformation is None or
        transformation in ["log", "square_root", "exponential", "tanh"]):
        return
    trafo_folder = OUTPUT_PATH.joinpath("transformer")
    trafo_path = trafo_folder.joinpath(f"transformer_{model_id}.pkl")
    return joblib.load(trafo_path)
    
def prepare_data(features: dict, metadata: list) -> Tuple[pd.DataFrame, dict]:
    """Converts input features into the format required by the model."""

    # Retrieve metadata of the model
    model_metadata = [m for m in metadata if m['id'] == features['model']][0]

    # Columns that were excluded during training
    dropped_columns = model_metadata['data']['reduced']

    # Features not to be included for prediction
    excluded_features = ['model'] + dropped_columns if dropped_columns else ['model']
    diamond_features = {k: features[k] for k in features if k not in excluded_features}

    features_df = pd.DataFrame([diamond_features])
    if model_metadata['data']['dummy']:
        return DataManager._categorical_to_dummy(
            features_df, columns_categories=COLUMNS_CATEGORIES), model_metadata
    elif model_metadata['data']['ordinal']:
        return DataManager._categorical_to_ordinal(
            features_df, columns_categories=COLUMNS_CATEGORIES), model_metadata
    else:
        features_df, model_metadata

def predict_diamond_value(features: dict) -> float:
    """The value of the diamond with the features defined in the request
    is predicted through the requested model.
    
    Expected JSON format:
    {
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI2",
        "depth": 61.5,
        "table": 55,
        "x": 3.95,
        "y": 3.98,
        "z": 2.43,
        "model": "f202c0a3-f8d4-4faa-b202-7961886c0fe8"
    }
    """

    metadata = retrieve_metadata()
    check_features_values(features, metadata)

    # Retrieving the required model
    retrieved_model = load_model(features['model'])

    # Features formatted as required by the model
    data, model_metadata = prepare_data(features, metadata)

    # Retrieve the transformer to reverse the operation applied in training
    trafo_string = model_metadata['transformation']
    trafo_string = '_'.join(trafo_string.lower().split()) if isinstance(trafo_string, str) else None
    inv_trafo = load_inverse_transformer(features['model'],
                                         transformation=trafo_string)

    prediction = retrieved_model.predict(data)
    prediction = inverse_transform_data(pd.Series(prediction), trafo_string, inv_trafo)
    return float(prediction[0])

def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def absolute_difference(weight1: float, weight2: float) -> float:
    return abs(weight1 - weight2)

def relative_difference(weight1: float, weight2: float) -> float:
    return abs(weight1 - weight2) / ((weight1 + weight2) / 2) * 100

def squared_difference(weight1: float, weight2: float) -> float:
    return (weight1 - weight2) ** 2

def z_score(weight1: float, mean: float, std: float) -> float:
    return abs(weight1 - mean) / std

def cosine_similarity(weight1: float, weight2: float) -> float:
    return 1 - cosine([weight1], [weight2])

def calculate_similarity(similarity_type: str,
                         weight1: float,
                         weight2: float,
                         mean: Optional[float] = None,
                         std : Optional[float] = None) -> float:
    similarity_functions = {
        "absolute difference": absolute_difference,
        "relative difference": relative_difference,
        "squared difference": squared_difference,
        "z score": z_score,
        "cosine similarity": cosine_similarity
    }

    if similarity_type == "z score":
        return similarity_functions[similarity_type](weight1, mean, std)
    else:
        return similarity_functions[similarity_type](weight1, weight2)

def find_similar_samples(features: dict) -> dict:
    """Retrieves the N samples with the same cut, colour and clarity and
    the closest weight to the diamond with the features specified in the
    request.

    Expected JSON format:
    {
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI2",
        "n": 5,
        "method": "cosine similarity"
    }
    """
    check_features_values(features)
    
    n = features['n']
    method = features['method']
    data_name = features['dataset_name']

    # Loading the specified dataset
    data = load_data(DATA_PATH.joinpath(data_name))

    # Limit the dataset to those diamonds that have the same cut, colour
    # and clarity as the request
    filtered_data = data[
        (data['cut'].str.lower() == features['cut'].lower()) &
        (data['color'].str.lower() == features['color'].lower()) &
        (data['clarity'].str.lower() == features['clarity'].lower())
    ]

    # Calculating the similarity with respect to the weight of diamonds
    diamond_weights = filtered_data['carat']
    given_weight = float(features['carat'])
    method = method.lower()
    if method == "z score":
        mean_weight = np.mean(diamond_weights)
        std_weight = np.std(diamond_weights)
        differences = [calculate_similarity(method, weight, given_weight, mean_weight, std_weight)
                       for weight in diamond_weights]
    else:
        differences = [calculate_similarity(method, weight, given_weight)
                       for weight in diamond_weights]
    weight_diff = "_".join(method.split())
    filtered_data[weight_diff] = differences

    # Retrieve the most similar n
    similar_samples = filtered_data.nsmallest(n, weight_diff)
    return similar_samples.to_dict(orient='records')

def train_model_from_configuration(configs: dict) -> dict:
    """Trains a model based on the two specified configuration files.
    
    Expected JSON format:
    {
        "training_config_name": "train_config.json",
        "data_config_name": "data_config.json"
    }

    or
    
    {
        "training_config_path": "path/to/train_config.json",
        "data_config_path": "path/to/data_config.json"
    }
    """

    check_features_values(configs)

    message = "Successful training"
    # Retrieving the paths to the two configuration files
    if "training_config_name" in configs:
        training_config_name = configs['training_config_name']
        data_config_name = configs['data_config_name']
        training_config = CONFIG_PATH.joinpath(training_config_name)
        data_config = CONFIG_PATH.joinpath(data_config_name)
    elif "training_config_path" in configs:
        training_config = configs['training_config_path']
        data_config = configs['data_config_path']

    error_train_path = False
    error_data_path = False
    if not Path(training_config).exists():
        training_config = CONFIG_PATH.joinpath("train_config.json")
        error_train_path = True
    if not Path(data_config).exists():
        data_config = CONFIG_PATH.joinpath("data_config.json")
        error_data_path = True

    if error_train_path and error_data_path:
        message = "Both configuration files do not exist"
    elif error_train_path:
        message = "Training configuration file non-existent"
    elif error_data_path:
        message = "Data configuration file non-existent"

    # Model training
    train_object = main(training_config_file=training_config,
                        data_config_file=data_config)

    # Retrieval of training configuration metadata
    train_id = train_object.training_uuid
    model_name = train_object.history['modelType']
    hyperparameters = train_object.history['parameters']

    metadata_dataset = train_object.history['data']
    dataset_name = metadata_dataset['name']
    preprocessing_operation = {k: v
                               for k, v in metadata_dataset.items()
                               if k in ['clean', 'reduced', 'dummy', 'ordinal']}
    preprocessing_operation['transformation'] = train_object.history['transformation']
    split_size = train_object.history['splitSize']

    metrics = train_object.history['evaluationMetrics']

    training_type = train_object.configuration.get('training').get('tuning')
    tuning_active = training_type.get('active', False)
    trial_number = training_type.get('trialsNumber')

    response_dict = {
        "train_id": train_id,
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "dataset_name": dataset_name,
        "processing_operation": preprocessing_operation,
        "split_size": split_size,
        "metrics": metrics,
        "training_type": f"bayesian hyperparameter tuning [{trial_number} trials]" if tuning_active else "one-shot",
        "train_config_file": str(training_config),
        "data_config_file": str(data_config),
        "message": message
    }
    return response_dict