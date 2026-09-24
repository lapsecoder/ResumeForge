"""Provider tests using a fake sentence-transformer class.

No real model library is imported or downloaded: ``_get_st_class`` is
monkeypatched. The only real call happens in one test asserting the pure CPU
fallback path on this machine (no torch installed here).
"""

from __future__ import annotations

import pytest

from app.semantic_matching.model import (
    InferenceError,
    LocalSentenceTransformerProvider,
    ModelLoadError,
    get_embedding_provider,
    reset_embedding_provider,
)


class FakeSentenceTransformer:
    instances = 0
    encode_calls = 0

    def __init__(self, model_name: str, device: str | None = None) -> None:
        FakeSentenceTransformer.instances += 1
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, sentences, batch_size: int | None = None):
        FakeSentenceTransformer.encode_calls += 1
        return [[1.0, 0.0] if sentence else [0.0, 0.0] for sentence in sentences]


@pytest.fixture(autouse=True)
def _patch_st_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        LocalSentenceTransformerProvider,
        "_get_st_class",
        lambda self: FakeSentenceTransformer,
    )
    monkeypatch.setattr(
        LocalSentenceTransformerProvider, "_select_device", lambda self: "cpu"
    )
    FakeSentenceTransformer.instances = 0
    FakeSentenceTransformer.encode_calls = 0
    yield
    reset_embedding_provider()


class TestLazyLoading:
    def test_construction_does_not_load_model(self) -> None:
        provider = LocalSentenceTransformerProvider()
        assert provider._model is None
        assert FakeSentenceTransformer.instances == 0

    def test_first_embedding_loads_exactly_once(self) -> None:
        provider = LocalSentenceTransformerProvider()
        provider.embed_texts(["hello"])
        assert FakeSentenceTransformer.instances == 1
        assert provider._model is not None

    def test_model_is_reused_across_embedding_calls(self) -> None:
        provider = LocalSentenceTransformerProvider()
        provider.embed_texts(["one"])
        provider.embed_texts(["two"])
        assert FakeSentenceTransformer.instances == 1
        assert FakeSentenceTransformer.encode_calls == 2

    def test_importing_module_does_not_load_model(self) -> None:
        from app import semantic_matching  # noqa: PLC0415

        assert semantic_matching is not None
        assert FakeSentenceTransformer.instances == 0


class TestEmbeddingBehaviour:
    def test_embedding_shape(self) -> None:
        provider = LocalSentenceTransformerProvider()
        vectors = provider.embed_texts(["a", "b", "c"])
        assert len(vectors) == 3
        assert all(len(vector) == 2 for vector in vectors)
        assert [1.0, 0.0] in vectors

    def test_empty_input_yields_empty_output(self) -> None:
        provider = LocalSentenceTransformerProvider()
        assert provider.embed_texts([]) == []

    def test_device_selection_is_recorded_in_metadata(self) -> None:
        provider = LocalSentenceTransformerProvider()
        provider.embed_texts(["x"])
        assert provider.metadata().device == "cpu"
        assert provider.metadata().model_dimension == 2

    def test_metadata_exposes_model_facts_only(self) -> None:
        provider = LocalSentenceTransformerProvider()
        metadata = provider.metadata()
        assert metadata.model_name.startswith("sentence-transformers/")
        assert metadata.source == "local Hugging Face cache"
        assert metadata.license == "Apache-2.0"


class TestErrorHandling:
    def test_missing_library_raises_model_load_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _missing(*args, **kwargs):
            raise ImportError("no sentence_transformers module")

        monkeypatch.setattr(LocalSentenceTransformerProvider, "_get_st_class", _missing)
        provider = LocalSentenceTransformerProvider()
        with pytest.raises(ModelLoadError):
            provider.embed_texts(["x"])

    def test_inference_failure_raises_inference_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BrokenSentenceTransformer(FakeSentenceTransformer):
            def encode(self, sentences, batch_size: int | None = None):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            LocalSentenceTransformerProvider,
            "_get_st_class",
            lambda self: BrokenSentenceTransformer,
        )
        provider = LocalSentenceTransformerProvider()
        with pytest.raises(InferenceError):
            provider.embed_texts(["x"])


class TestRealCpuFallback:
    def test_no_torch_machine_selects_cpu(self) -> None:
        provider = LocalSentenceTransformerProvider()
        assert provider._select_device() == "cpu"


class TestSingleton:
    def test_get_embedding_provider_returns_same_instance(self) -> None:
        first = get_embedding_provider()
        second = get_embedding_provider()
        assert first is second
        reset_embedding_provider()
        assert get_embedding_provider() is not first
