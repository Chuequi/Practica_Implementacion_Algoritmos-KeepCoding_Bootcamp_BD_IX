from pydantic import BaseModel

from app.core.enums import Sentiment


class SentimentPredictionResult(BaseModel):
    label: Sentiment
    score: float
    elapsed_time: float
