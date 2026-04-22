from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base



class Run(Base):
    __tablename__ = 'runs'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)  # Optional title for the run

    #path of the video in Supabase Storage, e.g., "user-id/uuid.mp4"
    video_path = Column(String, nullable=False, unique=True)

    #complex JSON output from AI model
    analysis_results = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Foreign key to link this run to a user
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    status = Column(String, default="pending", nullable=False)
    # Establish the relationship to the User model
    owner = relationship("User", back_populates="runs")

    analysis_result = relationship("AnalysisResult", back_populates="owner", uselist=False)