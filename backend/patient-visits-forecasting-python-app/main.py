from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.forecasting_router import router as forecasting_router


app = FastAPI(
    title="Patient Visits Forecasting API",
    description="Forecasting API with MLflow Tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# Routers
# --------------------------------------------------

app.include_router(
    forecasting_router,
    prefix="/api/v1"
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "application": "Patient Visits Forecasting API",
        "version": "1.0.0",
        "status": "Running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


