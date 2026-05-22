from __future__ import annotations

from types import SimpleNamespace

from rag_flow.config import RetrievalConfig
from rag_flow.retrieval import (
    RetrievalEngine,
    RetrievedImage,
    build_final_output,
    _candidate_min_score,
    _colpali_maxsim_score,
    _payload_position_key,
    _visual_chunk_alignment_score,
    _visual_page_query_filter,
)


def test_candidate_min_score_uses_allowed_drop_ratio():
    assert _candidate_min_score(best_score=10.0, min_candidate_score=0.0, min_score_ratio=0.0) == 10.0
    assert _candidate_min_score(best_score=10.0, min_candidate_score=0.0, min_score_ratio=0.2) == 8.0
    assert _candidate_min_score(best_score=10.0, min_candidate_score=0.0, min_score_ratio=0.5) == 5.0
    assert _candidate_min_score(best_score=10.0, min_candidate_score=0.0, min_score_ratio=1.0) == 0.0


def test_candidate_min_score_respects_absolute_floor():
    assert _candidate_min_score(best_score=10.0, min_candidate_score=9.0, min_score_ratio=0.5) == 9.0


def test_visual_alignment_keeps_page_prior_on_single_page_chunk():
    visual_payload = {
        "is_visual_page": True,
        "page_idx": 4,
        "chunk_ids_on_page": ["manual-chunk-00001"],
    }
    chunk_payload = {
        "chunk_id": "manual-chunk-00001",
        "page_indices": [4],
        "bboxes_by_page": {"4": [[100, 100, 900, 300]]},
    }

    assert _visual_chunk_alignment_score(chunk_payload, visual_payload) == 1.0


def test_visual_alignment_attenuates_cross_page_chunks():
    visual_payload = {
        "is_visual_page": True,
        "page_idx": 5,
        "chunk_ids_on_page": ["manual-chunk-00002"],
    }
    chunk_payload = {
        "chunk_id": "manual-chunk-00002",
        "page_indices": [4, 5],
        "bboxes_by_page": {
            "4": [[100, 100, 900, 900]],
            "5": [[100, 100, 300, 200]],
        },
    }

    score = _visual_chunk_alignment_score(chunk_payload, visual_payload)

    assert 0.5 < score < 1.0


def test_visual_alignment_rejects_chunks_outside_visual_page():
    visual_payload = {"is_visual_page": True, "page_idx": 5, "chunk_ids_on_page": ["other"]}
    chunk_payload = {
        "chunk_id": "manual-chunk-00003",
        "page_indices": [6],
        "bboxes_by_page": {"6": [[100, 100, 900, 300]]},
    }

    assert _visual_chunk_alignment_score(chunk_payload, visual_payload) == 0.0


def test_payload_position_key_sorts_by_page_then_bbox_top_left():
    upper = {
        "chunk_id": "upper",
        "chunk_idx": 2,
        "page_indices": [3],
        "bboxes_by_page": {"3": [[200, 100, 500, 150]]},
    }
    lower = {
        "chunk_id": "lower",
        "chunk_idx": 1,
        "page_indices": [3],
        "bboxes_by_page": {"3": [[100, 300, 500, 350]]},
    }

    assert _payload_position_key(upper, 3) < _payload_position_key(lower, 3)


def test_rrf_keeps_route_contributions_for_visual_prior():
    config = SimpleNamespace(retrieval=RetrievalConfig(10, 3, 60, 1.5, False))
    engine = RetrievalEngine(config)  # type: ignore[arg-type]
    dense_hit = SimpleNamespace(id="chunk-1", payload={"chunk_id": "chunk-1"})
    visual_hit = SimpleNamespace(id="page-1", payload={"is_visual_page": True, "page_idx": 0})

    ranking = engine._compute_rrf([dense_hit], [], [visual_hit])

    visual = next(item for item in ranking if item["payload"].get("is_visual_page"))
    assert visual["routes"]["visual"] == 1.5 / 61


def test_candidate_mode_normalizes_direct_aliases():
    config = SimpleNamespace(retrieval=RetrievalConfig(10, 3, 60, 1.5, False, candidate_mode="direct-rank"))
    engine = RetrievalEngine(config)  # type: ignore[arg-type]

    assert engine._candidate_mode() == "direct"


def test_candidate_mode_rejects_removed_seed_expansion():
    config = SimpleNamespace(retrieval=RetrievalConfig(10, 3, 60, 1.5, False, candidate_mode="seed"))
    engine = RetrievalEngine(config)  # type: ignore[arg-type]

    try:
        engine._candidate_mode()
    except ValueError as exc:
        assert "Seed expansion has been removed" in str(exc)
    else:
        raise AssertionError("candidate_mode='seed' should be rejected")


def test_context_block_includes_breadcrumb_when_chunk_content_lacks_it():
    config = SimpleNamespace(retrieval=RetrievalConfig(10, 3, 60, 1.5, False))
    engine = RetrievalEngine(config)  # type: ignore[arg-type]
    block = engine._format_context_block(
        {
            "visual_alignment_score": 0.0,
            "payload": {
                "source_relpath": "DSS/manual.pdf",
                "page_idx": 4,
                "page_start": 4,
                "page_end": 4,
                "section_path": ["1 Overview", "1.1 Login"],
                "section_title": "1.1 Login",
                "chunk_content": "Login body",
            },
        }
    )

    assert "[Breadcrumb: DSS > manual.pdf > 1 Overview > 1.1 Login]" in block
    assert "[Source: DSS/manual.pdf, Page: 5]\n" in block
    assert "Section:" not in block


def test_context_block_does_not_duplicate_existing_breadcrumb():
    config = SimpleNamespace(retrieval=RetrievalConfig(10, 3, 60, 1.5, False))
    engine = RetrievalEngine(config)  # type: ignore[arg-type]
    block = engine._format_context_block(
        {
            "visual_alignment_score": 0.0,
            "payload": {
                "source_relpath": "DSS/manual.pdf",
                "breadcrumb": "DSS > manual.pdf > 1 Overview",
                "page_idx": 0,
                "chunk_content": "[Breadcrumb: DSS > manual.pdf > 1 Overview]\nLogin body",
            },
        }
    )

    assert block.count("[Breadcrumb:") == 1


def test_image_references_include_all_images_with_policies(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "recommended.png").write_bytes(b"png")
    config = SimpleNamespace(
        paths=SimpleNamespace(base_dir=tmp_path),
        retrieval=RetrievalConfig(10, 3, 60, 1.5, False),
    )
    engine = RetrievalEngine(config)  # type: ignore[arg-type]
    payload = {
        "source_relpath": "DSS/manual.pdf",
        "chunk_id": "manual-chunk-00001",
        "page_idx": 2,
        "image_answering_evidence": [
            {
                "img_path": "images/optional.png",
                "page_idx": 1,
                "image_answering_policy": "image_optional",
            },
            {
                "img_path": "images/recommended.png",
                "page_idx": 2,
                "bbox": [1, 2, 3, 4],
                "image_caption": "Login",
                "image_description_vlm": "A login form.",
                "image_answering_policy": "image_recommended",
                "image_answering_confidence": "high",
                "image_answering_reason": "Visible labels are important.",
            },
            {
                "img_path": "images/required.png",
                "page_idx": 3,
                "image_answering_policy": "required",
            },
        ],
    }

    references = engine._image_references_for_payload(payload, hit_rank=4)

    assert [reference.img_path for reference in references] == [
        "images/optional.png",
        "images/recommended.png",
        "images/required.png",
    ]
    assert [reference.image_answering_policy for reference in references] == [
        "image_optional",
        "image_recommended",
        "required",
    ]
    assert references[1].image_path == str(tmp_path / "images" / "recommended.png")
    assert references[1].image_exists is True
    assert references[1].bbox == [1.0, 2.0, 3.0, 4.0]
    assert references[1].hit_rank == 4
    assert references[1].chunk_id == "manual-chunk-00001"
    assert references[2].image_exists is False


def test_image_references_fall_back_to_config_base_dir(tmp_path):
    config = SimpleNamespace(
        paths=SimpleNamespace(base_dir=tmp_path),
        retrieval=RetrievalConfig(10, 3, 60, 1.5, False),
    )
    engine = RetrievalEngine(config)  # type: ignore[arg-type]
    payload = {
        "source_relpath": "manual.pdf",
        "image_answering_evidence": [
            {"img_path": "images/required.png", "image_answering_policy": "image_required"}
        ],
    }

    references = engine._image_references_for_payload(payload, hit_rank=1)

    assert len(references) == 1
    assert references[0].image_path == str(tmp_path / "images" / "required.png")


def test_build_final_output_is_text_only_by_default(tmp_path):
    image = RetrievedImage(
        hit_rank=1,
        chunk_id="chunk-1",
        source_relpath="manual.pdf",
        img_path="images/recommended.png",
        image_path=str(tmp_path / "images" / "recommended.png"),
        image_exists=True,
        page_idx=0,
        page_number=1,
        bbox=[],
        image_answering_policy="image_recommended",
    )

    final_output = build_final_output(context="ctx", images=(image,), include_images=False)

    assert final_output.mode == "context_only"
    assert final_output.content == ({"type": "text", "text": "ctx"},)
    assert final_output.images == ()


def test_build_final_output_filters_to_existing_recommended_images(tmp_path):
    optional = RetrievedImage(
        hit_rank=1,
        chunk_id="chunk-1",
        source_relpath="manual.pdf",
        img_path="images/optional.png",
        image_path=str(tmp_path / "images" / "optional.png"),
        image_exists=True,
        page_idx=0,
        page_number=1,
        bbox=[],
        image_answering_policy="image_optional",
    )
    recommended = RetrievedImage(
        hit_rank=1,
        chunk_id="chunk-1",
        source_relpath="manual.pdf",
        img_path="images/recommended.png",
        image_path=str(tmp_path / "images" / "recommended.png"),
        image_exists=True,
        page_idx=0,
        page_number=1,
        bbox=[],
        image_answering_policy="image_recommended",
    )
    required_missing = RetrievedImage(
        hit_rank=1,
        chunk_id="chunk-1",
        source_relpath="manual.pdf",
        img_path="images/required.png",
        image_path=str(tmp_path / "images" / "required.png"),
        image_exists=False,
        page_idx=0,
        page_number=1,
        bbox=[],
        image_answering_policy="image_required",
    )

    final_output = build_final_output(
        context="ctx",
        images=(optional, recommended, required_missing),
        include_images=True,
    )

    assert final_output.mode == "openai_compatible_multimodal"
    assert final_output.content == (
        {"type": "text", "text": "ctx"},
        {"type": "image_url", "image_url": {"url": str(tmp_path / "images" / "recommended.png")}},
    )
    assert final_output.images == (recommended,)


def test_direct_rank_candidates_skip_visual_pages():
    config = SimpleNamespace(retrieval=RetrievalConfig(10, 3, 60, 1.5, False, candidate_mode="direct"))
    engine = RetrievalEngine(config)  # type: ignore[arg-type]
    final_ranking = [
        {
            "id": "visual-page-1",
            "score": 0.9,
            "payload": {"is_visual_page": True, "source_relpath": "manual.pdf", "page_idx": 3},
            "routes": {"visual": 0.9},
        },
        {
            "id": "chunk-1",
            "score": 0.4,
            "payload": {
                "source_relpath": "manual.pdf",
                "chunk_id": "chunk-1",
                "page_idx": 3,
                "page_indices": [3],
            },
            "routes": {"dense": 0.25, "sparse": 0.15},
        },
    ]

    candidates = engine._direct_rank_candidates(final_ranking)

    assert len(candidates) == 1
    assert candidates[0]["payload"]["chunk_id"] == "chunk-1"
    assert candidates[0]["score"] == 0.4
    assert candidates[0]["visual_score"] == 0.0


def test_visual_page_local_candidates_stay_on_hit_page():
    class FakeClient:
        def scroll(self, **kwargs):
            return [
                SimpleNamespace(
                    id="chunk-a",
                    payload={
                        "source_relpath": "manual.pdf",
                        "chunk_id": "chunk-a",
                        "page_idx": 3,
                        "page_indices": [3],
                        "bboxes_by_page": {"3": [[10, 10, 100, 100]]},
                    },
                ),
                SimpleNamespace(
                    id="chunk-b",
                    payload={
                        "source_relpath": "manual.pdf",
                        "chunk_id": "chunk-b",
                        "page_idx": 4,
                        "page_indices": [4],
                        "bboxes_by_page": {"4": [[10, 10, 100, 100]]},
                    },
                ),
            ], None

    config = SimpleNamespace(retrieval=RetrievalConfig(10, 3, 60, 0.5, False, candidate_mode="visual-page-local-bbox"))
    engine = RetrievalEngine(config)  # type: ignore[arg-type]
    engine.client = FakeClient()
    fake_models = SimpleNamespace(
        FieldCondition=lambda **kwargs: SimpleNamespace(**kwargs),
        Filter=lambda **kwargs: SimpleNamespace(**kwargs),
        MatchValue=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    candidates = engine._visual_page_local_candidates(
        collection_name="manuals",
        final_ranking=[],
        visual_hits=[
            SimpleNamespace(
                id="visual-page-3",
                payload={
                    "is_visual_page": True,
                    "source_relpath": "manual.pdf",
                    "page_idx": 3,
                    "chunk_ids_on_page": ["chunk-a"],
                },
                score=99.0,
            )
        ],
        candidate_mode="visual-page-local-bbox",
        models=fake_models,
    )

    assert [candidate["payload"]["chunk_id"] for candidate in candidates] == ["chunk-a"]
    assert candidates[0]["visual_alignment_score"] == 1.0
    assert candidates[0]["visual_score"] > 0.0


def test_visual_page_query_filter_limits_colpali_to_visual_pages():
    class FakeMatchValue:
        def __init__(self, value):
            self.value = value

    class FakeFieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class FakeFilter:
        def __init__(self, must):
            self.must = must

    fake_models = SimpleNamespace(
        FieldCondition=FakeFieldCondition,
        Filter=FakeFilter,
        MatchValue=FakeMatchValue,
    )

    point_filter = _visual_page_query_filter(fake_models)

    assert point_filter.must[0].key == "is_visual_page"
    assert point_filter.must[0].match.value is True


def test_colpali_maxsim_scores_best_patch_per_query_token():
    query = [[1.0, 0.0], [0.0, 1.0]]
    page = [[1.0, 0.0], [0.0, 0.5]]

    assert _colpali_maxsim_score(query, page) == 2.0


def test_visual_page_scroll_query_sorts_by_local_maxsim():
    class FakeClient:
        def scroll(self, **kwargs):
            records = [
                SimpleNamespace(
                    id="low",
                    payload={"is_visual_page": True, "page_idx": 1},
                    vector={"page-image-colpali": [[0.0, 1.0]]},
                ),
                SimpleNamespace(
                    id="high",
                    payload={"is_visual_page": True, "page_idx": 2},
                    vector={"page-image-colpali": [[1.0, 0.0]]},
                ),
            ]
            return records, None

    config = SimpleNamespace(retrieval=RetrievalConfig(10, 3, 60, 1.5, False))
    engine = RetrievalEngine(config)  # type: ignore[arg-type]
    engine.client = FakeClient()
    fake_models = SimpleNamespace(
        FieldCondition=lambda **kwargs: SimpleNamespace(**kwargs),
        Filter=lambda **kwargs: SimpleNamespace(**kwargs),
        MatchValue=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    hits = engine._query_visual_pages_by_scroll(
        collection_name="manuals",
        visual_query=[[1.0, 0.0]],
        limit=1,
        models=fake_models,
    )

    assert [hit.id for hit in hits] == ["high"]
