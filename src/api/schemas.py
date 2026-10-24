"""Request and response models. Inputs are raw customer fields; no Gender and no identifiers (extra fields are rejected)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH = 1000


class Customer(BaseModel):
    """One customer's raw attributes. Unknown fields (including Gender, CustomerId, Surname) are rejected with a 422."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "CreditScore": 619, "Geography": "France", "Age": 42, "Tenure": 2, "Balance": 0.0,
            "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88}})

    CreditScore: int = Field(ge=300, le=900)
    Geography: Literal["France", "Germany", "Spain"]
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=10, description="Years as a customer")
    Balance: float = Field(ge=0, allow_inf_nan=False)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(gt=0, allow_inf_nan=False)


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class Driver(BaseModel):
    feature: str
    value: float = Field(description="The customer's value of this model feature")
    contribution: float = Field(description="Push on the model's churn log-odds, averaged over the 5 fold models; positive raises risk")
    direction: Literal["increases churn risk", "decreases churn risk"]


class Assumptions(BaseModel):
    scenario: str
    net_interest_margin: float
    annual_fee_per_product: float
    offer_cost: float
    retention_success_rate: float
    units: str
    note: str


class Prediction(BaseModel):
    churn_probability: float = Field(description="Calibrated probability of churning, clipped to [0.005, 0.995] in the API (the model can return exactly 0 or 1)")
    risk_band: Literal["low", "medium", "high"]
    estimated_annual_value: float = Field(description="Balance x net interest margin + NumOfProducts x annual fee, in units (illustrative assumptions)")
    expected_net_benefit: float = Field(
        description="p_churn x retention_success_rate x annual value - offer cost, in units, under the base assumptions. "
                    "Positive means contacting this customer is expected to pay for itself under those assumptions.")
    top_drivers: list[Driver] | None = Field(description="The 3 largest drivers of this customer's score; null when not requested (batch default)")


class PredictResponse(Prediction):
    request_id: str
    model_version: str
    assumptions: Assumptions
    note: str


class BatchResponse(BaseModel):
    request_id: str
    model_version: str
    count: int
    assumptions: Assumptions
    note: str
    predictions: list[Prediction] = Field(description="In the same order as the request")


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    training_date: str
    calibration_method: str | None
    n_features: int
    features: list[str]
    metrics: dict[str, float]
    metrics_note: str
    probability_clip: dict = Field(description="The serving-time clip on probabilities, and test metrics before and after it")
    risk_bands: dict
