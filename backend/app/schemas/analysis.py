from datetime import datetime
from typing import Any

from pydantic import BaseModel

class AnalysisBase(BaseModel):
    fps: float
    modules: dict

    class Config:
        from_attributes = True


class AnalysisCreateIn(AnalysisBase):
    run_id: int

class AnalysisGetIn(BaseModel):
    run_id: int

class AnalysisOut(AnalysisBase):
    id: int
    run_id: int
    created_at: datetime
    recommendations: dict[str, Any] | None = None


class RecommendationsOut(BaseModel):
    run_id: int
    recommendations: dict[str, Any]

    class Config:
        from_attributes = True
