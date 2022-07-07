import os
import time
import tempfile

import pickle
import tensorflow as tf
from loguru import logger
from tensorflow.keras.preprocessing.sequence import pad_sequences


from app.core.messages import NO_VALID_PAYLOAD
from app.models.payload import TextPayload, payload_to_text
from app.models.prediction import SentimentPredictionResult
from app.core.enums import Sentiment
from app.core.config import (
    KERAS_MODEL,
    TOKENIZER_MODEL,
    SEQUENCE_LENGTH,
    SENTIMENT_THRESHOLD,
)


class SentimentAnalysisModel:
    def __init__(
        self, model_dir,
    ):
        self.model_dir = model_dir
        self._load_local_model()

    def _load_local_model(self):
        keras_model_path = os.path.join(self.model_dir, KERAS_MODEL)
        with tempfile.NamedTemporaryFile(suffix=".h5") as local_file:
            with tf.io.gfile.GFile(keras_model_path, mode="rb") as gcs_file:
                local_file.write(gcs_file.read())
                self.model = tf.keras.models.load_model(local_file.name, compile=False)

        tokenizer_path = os.path.join(self.model_dir, TOKENIZER_MODEL)
        self.tokenizer = pickle.load(tf.io.gfile.GFile(tokenizer_path, mode="rb"))

    def _decode_sentiment(self, score: float, include_neutral=True) -> str:
        return Sentiment.NEGATIVE.value if score < 0.5 else Sentiment.POSITIVE.value

    def _pre_process(self, payload: TextPayload) -> str:
        logger.debug("Pre-processing payload.")
        return payload_to_text(payload)

    def _predict(self, text: str) -> float:
        logger.debug("Predicting.")
        # Tokenize text
        x_test = pad_sequences(
            self.tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH
        )
        # Predict
        return self.model.predict([x_test])[0]

    def _post_process(
        self, text: str, prediction: float, start_time: float
    ) -> SentimentPredictionResult:
        logger.debug("Post-processing prediction.")
        # Decode sentiment
        label = self._decode_sentiment(prediction)

        return SentimentPredictionResult(
            label=label, score=prediction, elapsed_time=(time.time() - start_time),
        )

    def predict(self, payload: TextPayload):
        if payload is None:
            raise ValueError(NO_VALID_PAYLOAD.format(payload))

        start_at = time.time()
        pre_processed_payload = self._pre_process(payload)
        prediction = self._predict(pre_processed_payload)
        post_processed_result = self._post_process(
            pre_processed_payload, prediction, start_at
        )

        return post_processed_result
