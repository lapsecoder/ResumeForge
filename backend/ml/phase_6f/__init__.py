"""Phase 6F — model evaluation + error analysis package.

Analysis-only.  Never modifies production matching/ATS behaviour, never
retrains the full Phase 6E suite, and never touches the original 6E artifacts.
All heavy lifting reuses the Phase 6E pipeline unchanged (splits, feature
builders, trainer, evaluator).
"""
