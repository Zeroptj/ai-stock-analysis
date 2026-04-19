"""SQLite database setup with SQLAlchemy — cache + audit trail."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from stock_analysis.config import settings


class Base(DeclarativeBase):
    pass


class Filing(Base):
    """SEC filing metadata and content cache."""

    __tablename__ = "filings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    filing_type = Column(String(20), nullable=False)  # 10-K, 10-Q, 8-K, etc.
    filing_date = Column(String(20))
    accession_number = Column(String(30), unique=True)
    content_json = Column(Text)  # Raw JSON data
    fetched_at = Column(DateTime, default=datetime.utcnow)


class PriceHistory(Base):
    """Daily price data cache."""

    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(String(10), nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class Financial(Base):
    """Parsed financial data (income stmt, balance sheet, cash flow)."""

    __tablename__ = "financials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    period = Column(String(10), nullable=False)  # e.g. "2024-Q4", "2024-FY"
    statement_type = Column(String(20), nullable=False)  # income, balance, cashflow
    data_json = Column(Text, nullable=False)
    source = Column(String(20))  # sec, yfinance
    fetched_at = Column(DateTime, default=datetime.utcnow)


class Valuation(Base):
    """Valuation results cache."""

    __tablename__ = "valuations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    model_type = Column(String(30), nullable=False)  # dcf_fcff, ddm, affo, etc.
    fair_value = Column(Float)
    current_price = Column(Float)
    upside_pct = Column(Float)
    assumptions_json = Column(Text)
    result_json = Column(Text)
    calculated_at = Column(DateTime, default=datetime.utcnow)


class Report(Base):
    """Generated report metadata."""

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    report_type = Column(String(20), default="full")
    file_path = Column(String(500))
    generated_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="completed")


class CacheEntry(Base):
    """Generic TTL-backed JSON cache for external API responses."""

    __tablename__ = "cache"

    key = Column(String(200), primary_key=True)  # e.g. "yf_info:AAPL"
    value_json = Column(Text, nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def get_engine():
    """Create SQLAlchemy engine."""
    db_path = Path(settings.database.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
