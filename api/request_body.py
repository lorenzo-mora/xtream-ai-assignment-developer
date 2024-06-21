from typing import Optional
from pydantic import BaseModel

class DiamondFeatures(BaseModel):
    model: Optional[str] = "e705470a-608d-4a65-b948-141f67284573"
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