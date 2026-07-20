from pydantic import BaseModel
from pydantic import Field

from typing import Optional




"""
class PredictRequest(BaseModel):

    #model_name: Optional[str]

    patient_count: int
    admissions: int
    discharges: int
    occupancy: float
"""

class PredictRequest(BaseModel):

    model_name: str

    horizon: int = Field(
        default=7,
        ge=1,
        le=365,
        description="""
        Number of future time periods to forecast.

        Examples:
        - 1  = Forecast next day
        - 7  = Forecast next 7 days
        - 30 = Forecast next 30 days
        - 90 = Forecast next 3 months

        The model uses historical patient visit data to predict future values.
        """
    )

class RegressionPredictRequest(BaseModel):

    patient_count: int = Field(..., example=120)
    admissions: int = Field(..., example=32)
    discharges: int = Field(..., example=30)
    occupancy: float = Field(..., example=71.5)

    year: int = Field(..., example=2024)
    month: int = Field(..., example=6)
    day: int = Field(..., example=14)
    dayofweek: int = Field(...,example=4)
    quarter: int = Field(..., example=2)
    weekofyear: int = Field(..., example=24)