from fastapi import FastAPI
from app.routers import plants

app = FastAPI(
    title="Plant Identifier API",
    description="Identifica tipos de plantas y proporciona información útil.",
    version="1.0.0"
)

app.include_router(plants.router)
