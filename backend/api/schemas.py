"""
Centralized Pydantic Request & Response Schemas (DTOs)
"""

from typing import Generic, List, Optional, TypeVar, Any
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar("T")

class StandardAPIResponse(BaseModel, Generic[T]):
    """Standardized API Response Envelope for all endpoints."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[T] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class GrowthCalculationRequest(BaseModel):
    """Input payload for Basal Area Growth calculation."""
    species: str = Field(..., example="Mh", description="Species code (e.g. Mh, By, Pw) or Forest Unit (e.g. FU_TH1)")
    initial_ba: float = Field(..., gt=0, example=25.0, description="Initial stand basal area (m^2/ha)")
    ags_fraction: float = Field(..., ge=0, le=1, example=0.75, description="Acceptable Growing Stock fraction (0-1)")
    tcr_index: Optional[int] = Field(None, ge=1, le=5, example=3, description="Temperature-Precipitation Climate Ratio index (1-5)")
    dq_cm: float = Field(..., gt=0, example=22.5, description="Quadratic Mean DBH (cm)")
    q_factor: float = Field(..., gt=0, example=1.5, description="Diameter distribution q-factor")
    cut_flag: int = Field(0, ge=0, le=1, example=0, description="Harvest cut flag (0 or 1)")
    projection_years: int = Field(5, ge=1, le=50, example=5, description="Growth projection horizon in years")

class GrowthCalculationResult(BaseModel):
    """Output payload for Basal Area Growth calculation."""
    species: str
    rgr_predicted: float
    annual_growth_rate: float
    initial_ba: float
    projected_ba: float
    projection_years: int
    trajectory: List[Dict[str, Any]]
