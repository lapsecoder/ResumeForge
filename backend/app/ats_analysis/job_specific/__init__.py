"""Job-Specific ATS Coverage analysis (Phase 6B).

Deterministic, explainable measurement of how completely the *terminology* of
a job description is represented in a parsed resume. Measures explicit
representation only — it does NOT infer skill possession and never predicts
hiring or ATS-pass outcomes. Inputs are transient in-memory models; nothing is
persisted, logged, or sent over the network.
"""
