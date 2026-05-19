from sqlalchemy import Column, Integer, ForeignKey, DateTime, func, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)

    fps = Column(Float, nullable=False, server_default="0")
    modules = Column(JSONB)
    recommendations = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    run_id = Column(Integer, ForeignKey('runs.id', ondelete='CASCADE'), nullable=False, unique=True)
    owner = relationship("Run", back_populates="analysis_result")
