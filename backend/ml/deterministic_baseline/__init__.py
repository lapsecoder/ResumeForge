"""Deterministic Phase-5A baseline adapter for synthetic ground-truth evaluation.

Maps audited dataset rows to Resume / JobDescription with NO fabrication of
missing fields (experience dates, education institutions, certifications).
The continuous Phase-5A score is thresholded into a binary label for direct
comparison with the supervised ML experiments.
"""
