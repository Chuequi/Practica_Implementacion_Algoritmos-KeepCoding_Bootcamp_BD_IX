from fastapi import APIRouter, Depends
from starlette.requests import Request

from app.models.payload import TextPayload
from app.models.prediction import SentimentPredictionResult
from app.services.models import SentimentAnalysisModel

router = APIRouter()


@router.post("/predict", response_model=SentimentPredictionResult, name="predict")
def post_predict(
    request: Request, data: TextPayload = None,
) -> SentimentPredictionResult:

    model: SentimentAnalysisModel = request.app.state.model
    prediction: SentimentPredictionResult = model.predict(data)

    return prediction
