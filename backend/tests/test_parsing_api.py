"""API tests for POST /api/v1/resumes/parse.

Verifies end-to-end parse flow, error handling, path-traversal safety, and
that no permanent files are created.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import (
    install_tempfile_tracker,
    make_docx_bytes,
    make_pdf_bytes,
    make_text_bytes,
)

client = TestClient(app)

RESUME_PDF_TEXT = (
    "John Doe\n"
    "john.doe@example.com\n"
    "\n"
    "SKILLS\n"
    "Python, PostgreSQL\n"
    "\n"
    "WORK EXPERIENCE\n"
    "Senior Engineer | Acme Corp\n"
    "Jun 2021 - Present\n"
    "- Led the platform team\n"
)


class TestParseSuccess:
    def test_parse_pdf(self) -> None:
        resp = client.post(
            "/api/v1/resumes/parse",
            files={
                "file": (
                    "resume.pdf",
                    make_pdf_bytes(RESUME_PDF_TEXT),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["contact"]["name"] == "John Doe"
        assert body["contact"]["email"] == "john.doe@example.com"
        assert body["skills"]["all"] == ["Python", "PostgreSQL"]
        assert body["experience"][0]["company"] == "Acme Corp"
        assert body["experience"][0]["end_date"] == "Present"
        assert body["metadata"]["file_type"] == "pdf"

    def test_parse_docx(self) -> None:
        docx_text = RESUME_PDF_TEXT.replace("John Doe", "Jane Smith").replace(
            "john.doe@example.com", "jane@example.com"
        )
        resp = client.post(
            "/api/v1/resumes/parse",
            files={
                "file": (
                    "resume.docx",
                    make_docx_bytes(docx_text),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["contact"]["email"] == "jane@example.com"

    def test_parse_txt(self) -> None:
        resp = client.post(
            "/api/v1/resumes/parse",
            files={
                "file": (
                    "resume.txt",
                    make_text_bytes(RESUME_PDF_TEXT),
                    "text/plain",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["metadata"]["file_type"] == "txt"
        assert body["experience"][0]["title"] == "Senior Engineer"


class TestParseErrors:
    def test_unsupported_extension(self) -> None:
        resp = client.post(
            "/api/v1/resumes/parse",
            files={"file": ("resume.xlsx", b"whatever", "application/octet-stream")},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "unsupported_file_type"

    def test_malformed_pdf(self) -> None:
        resp = client.post(
            "/api/v1/resumes/parse",
            files={"file": ("resume.pdf", b"this is not a pdf", "application/pdf")},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "malformed_file"

    def test_empty_file(self) -> None:
        resp = client.post(
            "/api/v1/resumes/parse",
            files={"file": ("resume.txt", b"", "text/plain")},
        )
        assert resp.status_code == 422

    def test_error_does_not_expose_paths_or_traceback(self) -> None:
        resp = client.post(
            "/api/v1/resumes/parse",
            files={"file": ("resume.pdf", b"garbage", "application/pdf")},
        )
        body = resp.json()["error"]
        assert "Traceback" not in body["message"]
        assert "\\" not in body["message"]
        assert "C:" not in body["message"]


class TestParseSecurity:
    def test_no_permanent_file_created(self, monkeypatch) -> None:
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)
        resp = client.post(
            "/api/v1/resumes/parse",
            files={
                "file": (
                    "resume.pdf",
                    make_pdf_bytes(RESUME_PDF_TEXT),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 200
        # Any temp dir the ingestion step created must already be removed.
        for p in created:
            assert not p.exists(), f"Temp dir {p} leaked after request"

    def test_path_traversal_filename_safe(self, monkeypatch) -> None:
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)
        resp = client.post(
            "/api/v1/resumes/parse",
            files={
                "file": (
                    "../../../etc/passwd.pdf",
                    make_pdf_bytes(RESUME_PDF_TEXT),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        assert created
        for p in created:
            assert p.name.startswith("resumeforge_"), f"Unexpected temp name: {p}"

    def test_parse_endpoint_does_not_log_content(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.DEBUG, logger="app.api"):
            client.post(
                "/api/v1/resumes/parse",
                files={
                    "file": (
                        "resume.txt",
                        make_text_bytes("SECRET-VALUE-42\nSKILLS\nPython\n"),
                        "text/plain",
                    )
                },
            )
        for record in caplog.records:
            assert "SECRET-VALUE-42" not in record.getMessage()
