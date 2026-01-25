from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


JsonLike = Union[Dict[str, Any], List[Any]]


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationHistory(Base):
    """
    Таблица для истории диалога.
    """
    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    message_txt: Mapped[str] = mapped_column(Text, nullable=False)
    message_from: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    user_state: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class UserMetadata(Base):
    """
    Таблица для истории диалога.
    """
    __tablename__ = "user_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


@dataclass(frozen=True)
class RepositoryConfig:
    db_url: str = "sqlite:///app.sqlite3"
    echo: bool = False


class Repository:
    """
    Репозиторий: сохранить / получить / удалить данные из 4 таблиц.

    Дизайн специально простой:
    - на каждый user_id можно хранить много "снимков" (каждый save создаёт новую строку)
    - get_latest_* возвращает самый свежий снимок
    - delete_* удаляет все записи пользователя по типу
    """

    def __init__(self, config: RepositoryConfig = RepositoryConfig()):
        self._engine = create_engine(config.db_url, echo=config.echo, future=True)
        self._Session = sessionmaker(bind=self._engine, autoflush=False, autocommit=False, future=True)

    def create_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    # ---------- utils ----------
    @staticmethod
    def _dump_json(payload: JsonLike) -> str:
        """
        ensure_ascii=False — чтобы русский сохранялся нормально.
        default=str — чтобы не падать на "сложных" типах (на всякий).
        """
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _load_json(data_json: str) -> JsonLike:
        return json.loads(data_json)

    def get_conversation_history(self, user_id: str):
        """Получить историю диалога пользователя из БД"""

        with self._Session() as s:
            conversation_history = (
                select(ConversationHistory)
                .where(ConversationHistory.user_id == str(user_id))
                .order_by(ConversationHistory.created_at.asc())

            )
            conversation_history = s.execute(conversation_history).scalars().all()
            if not conversation_history:
                return []

            result = []
            for msg in list(conversation_history):
                result.append({"role": msg.message_from, "text": msg.message_txt})
            return result

    def add_conversation_history(self, user_id: str, message_txt: str, message_from: str, created_at: datetime,
                                 user_state: str):
        """Добавить запись в историю диалога пользователя"""
        with self._Session() as s:
            row = ConversationHistory(user_id=str(user_id), message_txt=message_txt, message_from=str(message_from),
                                      created_at=created_at, user_state=user_state)
            s.add(row)
            s.commit()
            s.refresh(row)
            return row.id

    def clean_conversation_history(self, user_id: str):
        """Очистить историю диалога пользователя"""
        with self._Session() as s:
            stmt = delete(ConversationHistory).where(ConversationHistory.user_id == str(user_id))
            res = s.execute(stmt)
            s.commit()
            return res.rowcount or 0

    def get_metadata(self, user_id: str):
        """Получить последнюю запись с метаданными пользователя"""
        with self._Session() as s:
            stmt = (
                select(UserMetadata)
                .where(UserMetadata.user_id == str(user_id))
                .order_by(UserMetadata.created_at.desc(), UserMetadata.id.desc())
                .limit(1)
            )
            row = s.execute(stmt).scalar_one_or_none()
            return {} if row is None else self._load_json(row.metadata_json)

    def save_metadata(self, user_id: str, user_metadata: dict):
        """Добавить запись с метаданными пользователя"""
        with self._Session() as s:
            row = UserMetadata(user_id=str(user_id), metadata_json=self._dump_json(user_metadata), created_at=datetime.now())
            s.add(row)
            s.commit()
            s.refresh(row)
            return row.id

    def clean_metadata(self, user_id: str):
        """Очистить метаданные пользователя"""
        with self._Session() as s:
            stmt = delete(UserMetadata).where(UserMetadata.user_id == str(user_id))
            res = s.execute(stmt)
            s.commit()
            return res.rowcount or 0





