from fastapi import FastAPI

from routes.router import router

app = FastAPI(title="Youtube chat app")

app.include_router(router)


@app.get("/")
def root():
    return {"message": "RAG API is running..."}
