from pydantic import BaseModel


class RequestSchema(BaseModel):
    url: str
    query: str


class ResponseSchema(BaseModel):
    result: str
