"""API tests for POST /api/v1/resumes/extract."""

from fastapi.testclient import TestClient

from app.ingestion.validators import MAX_FILE_SIZE_BYTES
from app.main import app
from tests.conftest import (
    make_docx_bytes,
    make_malformed_docx_bytes,
    make_pdf_bytes,
    make_text_bytes,
)

client = TestClient(app)


def _upload(payload: bytes, filename: str, content_type: str):
    return client.post(
        "/api/v1/resumes/extract",
        files={"file": (filename, payload, content_type)},
    )


class TestSuccessfulExtraction:
    def test_pdf(self) -> None:
        content = make_pdf_bytes("John Doe", "Software Engineer")
        resp = _upload(content, "resume.pdf", "application/pdf")
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_type"] == "pdf"
        assert body["page_count"] == 2
        assert body["status"] == "success"
        assert "John Doe" in body["extracted_text"]
        assert body["character_count"] == len(body["extracted_text"])
        assert body["word_count"] > 0

    def test_docx(self) -> None:
        content = make_docx_bytes("John Doe", "Engineer")
        resp = _upload(
            content,
            "resume.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_type"] == "docx"
        assert body["status"] == "success"
        assert "John Doe" in body["extracted_text"]

    def test_txt(self) -> None:
        content = make_text_bytes("John Doe\nSoftware Engineer")
        resp = _upload(content, "resume.txt", "text/plain")
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_type"] == "txt"
        assert body["status"] == "success"
        assert "John Doe" in body["extracted_text"]


class TestValidationFailures:
    def test_unsupported_extension(self) -> None:
        resp = _upload(b"data", "resume.exe", "application/octet-stream")
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "unsupported_file_type"

    def test_missing_file(self) -> None:
        resp = client.post("/api/v1/resumes/extract", files={})
        assert resp.status_code == 422

    def test_oversized_file(self) -> None:
        payload = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        resp = _upload(payload, "resume.pdf", "application/pdf")
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "file_too_large"

    def test_empty_file(self) -> None:
        resp = _upload(b"", "resume.pdf", "application/pdf")
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "empty_file"

    def test_malformed_pdf(self) -> None:
        resp = _upload(b"This is not a pdf", "resume.pdf", "application/pdf")
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "malformed_file"

    def test_malformed_docx(self) -> None:
        resp = _upload(
            make_malformed_docx_bytes(),
            "resume.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "malformed_file"


class TestNoInternalInfoExposed:
    def test_error_does_not_expose_traceback_or_paths(self) -> None:
        resp = _upload(b"This is not a pdf", "resume.pdf", "application/pdf")
        body = resp.json()["error"]
        assert "traceback" not in body["message"].lower()
        assert body["message"].count("\\") == 0
        assert "c:" not in body["message"].lower()
        assert "app\\ingestion" not in body["message"].lower()


class TestHealthEndpoint:
    def test_health_unchanged(self) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
