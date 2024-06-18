from pathlib import Path
import pickle
from typing import Any
import pandas as pd

from pipeline import DataManager
from utils import FileUtils, COLUMNS_CATEGORIES

BASE_PATH = Path(__file__).resolve().parent
TRAINING_PATH = BASE_PATH.joinpath("train")
MODELS_PATH = TRAINING_PATH.joinpath("models")

def load_model(model_id: str) -> Any:
    model_path = MODELS_PATH.joinpath(f"model_{model_id}.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("ECCO: ", model, type(model))
    return model
    
def prepare_data(features: dict) -> pd.DataFrame:
    """Converts input features into the format required by the model."""
    try:
        models = FileUtils.read_json(TRAINING_PATH.joinpath("results.json"),
                                     may_not_exist=False)
    except Exception as e:
        raise
    # Retrieve metadata of the model
    model_metadata = [m for m in models['training'] if m['id'] == features['model']][0]

    # Columns that were excluded during training
    dropped_columns = model_metadata['data']['reduced']

    # Features not to be included for prediction
    excluded_features = ['model'] + dropped_columns if dropped_columns else ['model']
    diamond_features = {k: features[k] for k in features if k not in excluded_features}

    features_df = pd.DataFrame([diamond_features])
    if model_metadata['data']['dummy']:
        return DataManager._categorical_to_dummy(features_df, columns_categories=COLUMNS_CATEGORIES)
    elif model_metadata['data']['ordinal']:
        return DataManager._categorical_to_ordinal(features_df, columns_categories=COLUMNS_CATEGORIES)

def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def predict_diamond_value(features: dict):
    # Retrieving the required model
    retrieved_model = load_model(features['model'])

    # Features formatted as required by the model
    data = prepare_data(features)

    prediction = retrieved_model.predict(data)
    # TODO Add inverse transformation to prediction 
    return float(prediction[0])

# def find_similar_samples(features, n):
#     filtered_data = data[
#         (data['cut'] == features['cut']) &
#         (data['color'] == features['color']) &
#         (data['clarity'] == features['clarity'])
#     ]
#     filtered_data['weight_diff'] = abs(filtered_data['carat'] - float(features['carat']))
#     similar_samples = filtered_data.nsmallest(n, 'weight_diff')
#     return similar_samples.to_dict(orient='records')
