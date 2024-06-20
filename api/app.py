import datetime as dt
from datetime import datetime
import json
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from core import predict_diamond_value, find_similar_samples, BASE_PATH, train_model_from_configuration
from request_body import DiamondFeatures, SimilarRequest, TrainModel
from database import insert_request_response, create_tables

# Path to the SQLite DB
DB_PATH = BASE_PATH.joinpath('log/api_logs.db')

app = FastAPI()

# Create tables if they do not exist
create_tables(DB_PATH)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        # Storage of request start-up time
        start_time = datetime.now(dt.timezone.utc)

        # Copy the request body to avoid consuming the stream
        request_body = await request.body()
        request_body_dict = json.loads(request_body.decode()) if request_body else {}

        response = await call_next(request)

        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        response_body = json.loads(response_body.decode()) if response_body else {}

        # Storage of request completion time
        end_time = datetime.now(dt.timezone.utc)
        response_time = str((end_time - start_time).total_seconds())

        new_entry = {
            'timestamp': start_time.isoformat(),
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'response_time': response_time,
            'request_body': request_body_dict,
            'response_body': response_body
        }
        insert_request_response(db_path=DB_PATH, **new_entry)

        return JSONResponse(content=response_body, status_code=response.status_code, headers=dict(response.headers))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

router = APIRouter()

@router.post("/predict")
async def predict(features: DiamondFeatures) -> JSONResponse:
    try:
        json_features = jsonable_encoder(features)

        # Generating a prediction
        prediction = predict_diamond_value(json_features)
        response_content = {
            "predicted_value": prediction
            }
        return JSONResponse(content=response_content, status_code=200)
    except Exception:
        raise

@router.post("/similar")
async def similar(features: SimilarRequest) -> JSONResponse:
    try:
        json_features = jsonable_encoder(features)

        # Retrieval of the most similar diamonds.
        samples = find_similar_samples(json_features)
        response_content = {
            "samples": samples
            }
        return JSONResponse(content=response_content, status_code=200)
    except Exception:
        raise

@router.post("/train")
async def train_model(train_config: TrainModel) -> JSONResponse:
    try:
        configurations_json = jsonable_encoder(train_config)

        metadata = train_model_from_configuration(configurations_json)

        message = metadata.pop('message')

        response_content = {
            "message": message, #"Model training initiated successfully.",
            "training_config": metadata
        }
        return JSONResponse(content=response_content, status_code=200)
    except Exception:
        raise
    
app.include_router(router)