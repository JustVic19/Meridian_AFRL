"""SQLAlchemy engine from DATABASE_URL. Connection is lazy."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


def get_engine() -> Engine:
    return create_engine(get_database_url())
