import json
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DEMO_USERNAME = "Demo Learner"


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return "sqlite:///mandarin_trainer.db"
    # Normalize postgres:// URIs from cloud providers for SQLAlchemy 2.x + psycopg3
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


DATABASE_URL = _get_database_url()
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    native_language = Column(String(32), default="English")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    attempts = relationship("Attempt", back_populates="user", cascade="all, delete-orphan")


class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    phrase = Column(String(128), nullable=False)
    pinyin_str = Column(String(256), default="")
    overall_score = Column(Float, nullable=False)
    syllable_scores_json = Column(Text, default="[]")
    asr_heard = Column(String(128), default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="attempts")


def init_db() -> None:
    """Create tables and ensure the shared Demo Learner profile exists."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        demo = db.execute(
            select(User).where(User.username == DEMO_USERNAME)
        ).scalar_one_or_none()
        if demo is None:
            db.add(User(username=DEMO_USERNAME, native_language="Mixed"))
            db.commit()


def list_users() -> list[dict[str, Any]]:
    with SessionLocal() as db:
        users = db.execute(select(User).order_by(User.id.asc())).scalars().all()
        return [
            {
                "id": u.id,
                "username": u.username,
                "native_language": u.native_language,
            }
            for u in users
        ]


def get_or_create_user(username: str, native_language: str = "English") -> dict[str, Any]:
    clean_name = username.strip()[:64] or DEMO_USERNAME
    with SessionLocal() as db:
        user = db.execute(
            select(User).where(func.lower(User.username) == clean_name.lower())
        ).scalar_one_or_none()
        if user is None:
            user = User(username=clean_name, native_language=native_language.strip()[:32] or "English")
            db.add(user)
            db.commit()
            db.refresh(user)
        return {
            "id": user.id,
            "username": user.username,
            "native_language": user.native_language,
        }


def log_attempt(
    user_id: int,
    phrase: str,
    pinyin_str: str,
    overall_score: float,
    syllable_scores: list[float | None],
    asr_heard: str = "",
) -> None:
    with SessionLocal() as db:
        record = Attempt(
            user_id=user_id,
            phrase=phrase,
            pinyin_str=pinyin_str,
            overall_score=round(float(overall_score), 2),
            syllable_scores_json=json.dumps(syllable_scores),
            asr_heard=asr_heard,
        )
        db.add(record)
        db.commit()


def get_user_history(user_id: int, limit: int = 30) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(Attempt)
                .where(Attempt.user_id == user_id)
                .order_by(Attempt.created_at.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [
            {
                "id": r.id,
                "phrase": r.phrase,
                "pinyin": r.pinyin_str,
                "score": r.overall_score,
                "syllable_scores": json.loads(r.syllable_scores_json or "[]"),
                "heard": r.asr_heard,
                "timestamp": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
            }
            for r in rows
        ]
