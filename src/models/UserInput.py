from pydantic import BaseModel


class UserInput(BaseModel):
    prompt: str
