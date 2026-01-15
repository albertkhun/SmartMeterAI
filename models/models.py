from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from database.db import Base


class Upload(Base):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class UsageRecord(Base):
    __tablename__ = "usage_records"
    id = Column(Integer, primary_key=True, index=True)
    consumer_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    kwh = Column(Float)


class ConsumerRisk(Base):
    __tablename__ = "consumer_risks"
    id = Column(Integer, primary_key=True, index=True)
    consumer_id = Column(String, unique=True, index=True)
    risk_score = Column(Float)
    label = Column(String)
    reason = Column(Text)
    action = Column(String)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    consumer_id = Column(String, index=True)
    decision = Column(String)  # theft / false_alarm / need_more_data
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
