from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from core import predict_diamond_value, find_similar_samples
from request_body import DiamondFeatures, SimilarRequest

app = FastAPI()

@app.post("/predict")
async def predict(features: DiamondFeatures) -> JSONResponse:
    json_features = jsonable_encoder(features)

    # Generating a prediction
    prediction = predict_diamond_value(json_features)
    response_content = {
        "predicted_value": prediction
        }
    return JSONResponse(content=response_content, status_code=200)

@app.post("/similar")
async def similar(features: SimilarRequest) -> JSONResponse:
    json_features = jsonable_encoder(features)

    # Retrieval of the most similar diamonds.
    samples = find_similar_samples(json_features)
    response_content = {
        "samples": samples
        }
    return JSONResponse(content=response_content, status_code=200)