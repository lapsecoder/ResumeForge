"""Smoke tests for the foundation FastAPI app.

These validate that the empty skeleton is wired correctly; they are written to
be replaced/extended as application endpoints are added.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_preflight_allows_frontend_origin() -> None:
    response = client.options(
        "/api/v1/resumes/parse",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_blocks_disallowed_origin() -> None:
    response = client.options(
        "/api/v1/resumes/parse",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") is None
