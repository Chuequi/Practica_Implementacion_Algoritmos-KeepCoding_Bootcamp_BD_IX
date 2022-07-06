from pydantic import BaseModel


class TextPayload(BaseModel):
    text: str


def payload_to_text(payload: TextPayload) -> str:
    return payload.text
