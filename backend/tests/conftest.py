"""Shared fixtures for backend tests.

The HTTP API now requires an installation bearer token on every ingestion,
retrieval and stats route. These fixtures install a FastAPI dependency override
so pre-existing route tests keep exercising the application logic without a
live authentication database. Dedicated auth/ownership tests clear this override
explicitly to exercise the real enforcement.
"""

import pytest

from app import main
from app.auth import Installation

TEST_INSTALLATION = Installation(id="test-install", token_hash="pga1:test-hash")


@pytest.fixture(autouse=True)
def _bypass_auth_for_existing_route_tests(monkeypatch):
    """Make authenticated routes reachable for tests that are not about auth."""
    main.app.dependency_overrides[main.require_installation] = (
        lambda: TEST_INSTALLATION
    )
    yield
    main.app.dependency_overrides.pop(main.require_installation, None)
