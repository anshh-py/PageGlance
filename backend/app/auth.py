"""Installation-level bearer-token authentication for the PageGlance API.

Design (Option A from the M9 security audit):

- The Chrome extension generates a cryptographically secure token on first use
  and stores it only in ``chrome.storage.local`` (never hardcoded).
- The backend stores only a salted-domain SHA-256 hash of the token, never the
  raw token, so a leaked database does not expose usable credentials.
- Each token maps to one installation record. All ingestion, retrieval and
  stats routes require a valid bearer token, and every document is scoped to
  the authenticated installation's ``owner_id``.

This provides isolation between installations. It does not gate the service
against new self-registrations (that is the separate volume/abuse concern in
finding C2, handled by rate limiting, not here).
"""

import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import create_session_factory
from app.models import InstallationRecord

logger = logging.getLogger(__name__)

# 256-bit random bearer token.
TOKEN_BYTES = 32
MIN_TOKEN_CHARS = 32
MAX_TOKEN_CHARS = 256
# Domain-separated hash prefix so token hashes can never be confused with
# content hashes or other hex digests elsewhere in the schema.
_HASH_PREFIX = "pga1:"


@dataclass(frozen=True)
class Installation:
    """The verified caller identity for a request (installation-level)."""

    id: str
    token_hash: str


def generate_token() -> str:
    """Return a fresh cryptographic bearer token (never persisted in cleartext)."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """Return the secure, domain-separated hash stored for a token."""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


# Reassignable session factory so tests can inject a SQLite-backed factory and
# production uses the configured PostgreSQL engine.
get_session_factory = create_session_factory


def _find_installation(session: Session, token_hash: str) -> InstallationRecord | None:
    return session.scalar(
        select(InstallationRecord).where(InstallationRecord.token_hash == token_hash)
    )


def register_installation(token: str) -> Installation:
    """Register a token idempotently and return its installation identity.

    Registration is idempotent by token hash: re-registering the same token
    returns the existing installation rather than minting a new owner, so a
    client that re-registers after a network failure keeps access to its own
    previously persisted documents.
    """
    token = token.strip()
    if not MIN_TOKEN_CHARS <= len(token) <= MAX_TOKEN_CHARS:
        raise HTTPException(status_code=422, detail="The installation token is invalid.")
    token_hash = hash_token(token)
    with get_session_factory()() as session:
        record = _find_installation(session, token_hash)
        if record is not None:
            return Installation(id=record.id, token_hash=record.token_hash)
        installation_id = str(uuid.uuid4())
        session.add(
            InstallationRecord(
                id=installation_id,
                token_hash=token_hash,
                created_at=datetime.now(timezone.utc),
            )
        )
        session.commit()
        logger.info("installation_registered id=%s", installation_id)
        return Installation(id=installation_id, token_hash=token_hash)


def require_installation(
    authorization: str | None = Header(default=None),
) -> Installation:
    """FastAPI dependency: resolve and verify the bearer installation token."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    token_hash = hash_token(token)
    try:
        with get_session_factory()() as session:
            record = _find_installation(session, token_hash)
    except Exception:
        # Treat a backend/database unavailability as a service-level failure,
        # not as an authentication decision; never leak database details.
        raise HTTPException(status_code=503, detail="Authentication is temporarily unavailable.")
    if record is None:
        raise HTTPException(status_code=403, detail="Invalid credentials.")
    return Installation(id=record.id, token_hash=record.token_hash)
