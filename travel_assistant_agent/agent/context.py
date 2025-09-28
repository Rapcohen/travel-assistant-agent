from pydantic import BaseModel, Field


class Context(BaseModel):
    model: str
    model_provider: str = Field(default='ollama')
