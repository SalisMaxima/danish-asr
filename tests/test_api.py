"""Tests for API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from danish_asr import api


def test_health_endpoint():
    """Test health endpoint returns ok status."""
    client = TestClient(api.app, raise_server_exceptions=False)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "device" in data
