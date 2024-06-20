from typing import Optional
from pydantic import BaseModel, root_validator

class DiamondFeatures(BaseModel):
    model: Optional[str] = "19349c13-b711-4440-b669-ed9b199ad5e3"
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

class SimilarRequest(BaseModel):
    n: Optional[int] = 5
    method: Optional[str] = "cosine similarity"
    dataset_name: Optional[str] = "diamonds.csv"
    carat: float
    cut: str
    color: str
    clarity: str

class TrainModel(BaseModel):
    training_config_name: Optional[str] = None
    data_config_name: Optional[str] = None
    training_config_path: Optional[str] = None
    data_config_path: Optional[str] = None

    @root_validator(pre=False)
    def check_configs(cls, values):
        training_config_name = values.get('training_config_name')
        data_config_name = values.get('data_config_name')
        training_config_path = values.get('training_config_path')
        data_config_path = values.get('data_config_path')

        if not ((training_config_name and data_config_name) or
                (training_config_path and data_config_path)):
            err_msg = ("You must specify both 'training_config_name' and "
                       "'data_config_name', or both 'training_config_path' "
                       "and 'data_config_path'.")
            raise ValueError(err_msg)

        return values