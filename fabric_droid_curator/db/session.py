from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, SchemaMigration

SCHEMA_VERSION = 1


class Database:
    def __init__(self, url: str):
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, future=True, connect_args={"check_same_thread": False})
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._sqlite_pragmas)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    @staticmethod
    def _sqlite_pragmas(connection, _record) -> None:  # type: ignore[no-untyped-def]
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    def migrate(self) -> None:
        Base.metadata.create_all(self.engine)
        with self.session() as db:
            if db.get(SchemaMigration, SCHEMA_VERSION) is None:
                db.add(SchemaMigration(version=SCHEMA_VERSION, name="initial_curator_schema"))

    @contextmanager
    def session(self) -> Iterator[Session]:
        db = self.session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
