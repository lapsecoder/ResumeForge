# tests/

Cross-cutting / shared test assets for ResumeForge.

## Purpose

This directory holds fixtures and assets shared across the backend and
frontend test suites, primarily **sample PDF and DOCX resume files** used to
validate parsing and processing. It is intentionally not tied to a single
language or framework.

## Layout (to grow over time)

- `fixtures/` — sample resume and job-description files (PDF/DOCX/TXT).
- `sample-data/` — representative structured payloads for tests.

## Current status

Foundation stage — no fixtures added yet. Preserve and grow the sample files
here as parsing is implemented.
