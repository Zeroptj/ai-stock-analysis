"""TTL-backed cache for external API responses (yfinance, SEC EDGAR).

Usage:
    data = cached("yf_info:AAPL", ttl_hours=12, fetcher=lambda: yf.Ticker("AAPL").info)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Callable

import structlog

from stock_analysis.database import CacheEntry, get_session

logger = structlog.get_logger(__name__)


def cached(
    key: str,
    ttl_hours: float,
    fetcher: Callable[[], Any],
    skip_cache: bool = False,
) -> Any:
    """Return cached JSON value if fresh, else call `fetcher()` and store result.

    Returns None if fetcher raises — caller decides how to handle.
    """
    session = get_session()
    try:
        if not skip_cache:
            entry = session.query(CacheEntry).filter_by(key=key).first()
            if entry:
                age = datetime.utcnow() - entry.fetched_at
                if age < timedelta(hours=ttl_hours):
                    logger.debug("cache_hit", key=key, age_hours=round(age.total_seconds() / 3600, 2))
                    return json.loads(entry.value_json)
                logger.debug("cache_stale", key=key, age_hours=round(age.total_seconds() / 3600, 2))

        # Cache miss or skip — fetch fresh
        try:
            value = fetcher()
        except Exception as e:
            logger.warning("cache_fetch_error", key=key, error=str(e))
            # Fall back to stale cache if we have one
            entry = session.query(CacheEntry).filter_by(key=key).first()
            if entry:
                logger.info("cache_stale_fallback", key=key)
                return json.loads(entry.value_json)
            return None

        # Store (upsert)
        entry = session.query(CacheEntry).filter_by(key=key).first()
        if entry:
            entry.value_json = json.dumps(value, default=str)
            entry.fetched_at = datetime.utcnow()
        else:
            session.add(
                CacheEntry(
                    key=key,
                    value_json=json.dumps(value, default=str),
                    fetched_at=datetime.utcnow(),
                )
            )
        session.commit()
        logger.debug("cache_store", key=key)
        return value
    finally:
        session.close()


def invalidate(key: str) -> None:
    """Remove a cache entry by key."""
    session = get_session()
    try:
        session.query(CacheEntry).filter_by(key=key).delete()
        session.commit()
    finally:
        session.close()
