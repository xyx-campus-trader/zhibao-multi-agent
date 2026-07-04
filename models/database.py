"""
数据库模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class TaskRecord(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), unique=True, index=True)
    topic = Column(String(500))
    status = Column(String(32), default="pending")
    search_keywords = Column(JSON, default=dict)
    draft = Column(Text, default="")
    final_report = Column(Text, default="")
    review_notes = Column(Text, default="")
    user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class TaskState(Base):
    __tablename__ = "task_states"

    task_id = Column(String(64), primary_key=True)
    state = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow)
