from typing import Optional
from pydantic import BaseModel

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