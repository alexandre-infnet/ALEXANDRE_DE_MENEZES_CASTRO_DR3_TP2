from pydantic import BaseModel


class InputData(BaseModel):
    prompt: str
    max_length: int = 50
    num_return_sequences: int = 1
