from datetime import datetime

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
