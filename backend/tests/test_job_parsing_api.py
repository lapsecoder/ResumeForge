"""API tests for POST /api/v1/jobs/parse.

Verifies end-to-end job-description parsing, error handling, path-traversal
safety, temp-file cleanup, and that JD content is never logged.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.ingestion.validators import MAX_FILE_SIZE_BYTES
from app.main import app
from tests.conftest import (
    install_tempfile_tracker,
    make_docx_bytes,
    make_pdf_bytes,
    make_text_bytes,
)

client = TestClient(app)

JD_TEXT = (
    "Senior Backend Engineer\n"
    "Company: Acme Corp\n"
    "Location: Bangalore\n"
    "\n"
    "SUMMARY\n"
    "Build scalable services.\n"
    "\n"
    "RESPONSIBILITIES\n"
    "- Design and build APIs\n"
    "- Lead the platform team\n"
    "\n"
    "SKILLS\n"
    "Python, Go, PostgreSQL\n"
    "\n"
    "EXPERIENCE\n"
    "5+ years of experience with Python\n"
    "\n"
    "EDUCATION\n"
    "Bachelor's degree in Computer Science\n"
    "\n"
    "Salary\n"
    "$110k-$130k annual\n"
)


class TestJobParseSuccess:
    def test_parse_pdf(self) -> None:
        resp = client.post(
            "/api/v1/jobs/parse",
            files={
                "file": (
                    "job.pdf",
                    make_pdf_bytes(JD_TEXT),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["title"] == "Senior Backend Engineer"
        assert body["company"] == "Acme Corp"
        assert body["location"] == "Bangalore"
        assert body["summary"] == "Build scalable services."
        assert body["responsibilities"] == [
            "Design and build APIs",
            "Lead the platform team",
        ]
        assert body["required_skills"] == ["Python", "Go", "PostgreSQL"]
        assert "5+ years of experience with Python" in body["experience_requirements"]
        assert any(
            "Bachelor" in item and "Computer Science" in item
            for item in body["education_requirements"]
        )
        assert body["salary"][0]["range_text"] == "110k–130k"
        assert body["salary"][0]["currency"] == "$"
        assert body["metadata"]["file_type"] == "pdf"
        assert body["metadata"]["overall_confidence"] == "high"

    def test_parse_docx(self) -> None:
        resp = client.post(
            "/api/v1/jobs/parse",
            files={
                "file": (
                    "job.docx",
                    make_docx_bytes(JD_TEXT),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["metadata"]["file_type"] == "docx"
        assert body["required_skills"] == ["Python", "Go", "PostgreSQL"]

    def test_parse_txt(self) -> None:
        resp = client.post(
            "/api/v1/jobs/parse",
            files={
                "file": (
                    "job.txt",
                    make_text_bytes(JD_TEXT),
                    "text/plain",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["metadata"]["file_type"] == "txt"
        assert body["metadata"]["word_count"] > 0
        assert body["certifications"] == []
        assert body["nice_to_have"] == []


class TestJobParseErrors:
    def test_unsupported_extension(self) -> None:
        resp = client.post(
            "/api/v1/jobs/parse",
            files={"file": ("job.xlsx", b"whatever", "application/octet-stream")},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "unsupported_file_type"

    def test_malformed_pdf(self) -> None:
        resp = client.post(
            "/api/v1/jobs/parse",
            files={"file": ("job.pdf", b"this is not a pdf", "application/pdf")},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "malformed_file"

    def test_empty_file(self) -> None:
        resp = client.post(
            "/api/v1/jobs/parse",
            files={"file": ("job.txt", b"", "text/plain")},
        )
        assert resp.status_code == 422

    def test_oversized_file(self) -> None:
        resp = client.post(
            "/api/v1/jobs/parse",
            files={
                "file": ("job.pdf", b"x" * (MAX_FILE_SIZE_BYTES + 1), "application/pdf")
            },
        )
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "file_too_large"

    def test_missing_file(self) -> None:
        resp = client.post("/api/v1/jobs/parse", files={})
        assert resp.status_code == 422

    def test_error_does_not_expose_paths_or_traceback(self) -> None:
        resp = client.post(
            "/api/v1/jobs/parse",
            files={"file": ("job.pdf", b"garbage", "application/pdf")},
        )
        body = resp.json()["error"]
        assert "Traceback" not in body["message"]
        assert "\\" not in body["message"]
        assert "C:" not in body["message"]


class TestJobParseSecurity:
    def test_no_permanent_file_created(self, monkeypatch) -> None:
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)
        resp = client.post(
            "/api/v1/jobs/parse",
            files={
                "file": (
                    "job.txt",
                    make_text_bytes(JD_TEXT),
                    "text/plain",
                )
            },
        )
        assert resp.status_code == 200
        for p in created:
            assert not p.exists(), f"Temp dir {p} leaked after request"

    def test_path_traversal_filename_safe(self, monkeypatch) -> None:
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)
        resp = client.post(
            "/api/v1/jobs/parse",
            files={
                "file": (
                    "../../../etc/passwd.pdf",
                    make_pdf_bytes(JD_TEXT),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        for p in created:
            assert p.name.startswith("resumeforge_"), f"Unexpected temp name: {p}"

    def test_job_parse_endpoint_does_not_log_content(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.DEBUG, logger="app.api"):
            client.post(
                "/api/v1/jobs/parse",
                files={
                    "file": (
                        "job.txt",
                        make_text_bytes("JOB-CONFIDENTIAL-XYZ\nSKILLS\nPython\n"),
                        "text/plain",
                    )
                },
            )
        for record in caplog.records:
            assert "JOB-CONFIDENTIAL-XYZ" not in record.getMessage()
