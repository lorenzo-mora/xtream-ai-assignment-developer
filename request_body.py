from typing import Dict, Literal, Optional
from pydantic import BaseModel

class DiamondFeatures(BaseModel):
    model: Optional[str] = "0d665ad8-63e1-4e85-92a0-81860a9f10a2"
    carat: float
    cut: str#Literal['Fair', 'Good', 'Very Good', 'Ideal', 'Premium']
    color: str#Literal['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity: str#Literal['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
    depth: float
    table: float
    x: float
    y: float
    z: float

class SimilarRequest(BaseModel):
    features: Dict[str, str]
    n: Optional[int] = 5
    feature: Optional[str] = "carat"
    orderby: Optional[str] = "desc"