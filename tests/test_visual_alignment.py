from __future__ import annotations

from rag_flow.visual_alignment import (
    PatchEvidence,
    cosine_similarity_matrix,
    patch_bbox_for_index,
    rank_chunk_alignments,
    token_topk_patch_evidence,
)


def test_cosine_similarity_matrix_scores_query_tokens_against_patches():
    matrix = cosine_similarity_matrix(
        query_vectors=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        patch_vectors=[
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ],
    )

    assert matrix[0][0] == 1.0
    assert round(matrix[0][1], 3) == 0.707
    assert matrix[0][2] == -1.0
    assert matrix[1][0] == 0.0
    assert round(matrix[1][1], 3) == 0.707


def test_token_topk_patch_evidence_keeps_each_tokens_best_patches():
    similarity = [
        [0.1, 0.9, 0.2, 0.3],
        [0.8, 0.1, 0.7, 0.2],
    ]

    evidence = token_topk_patch_evidence(similarity, grid_width=2, grid_height=2, top_k=1)

    assert [(item.token_idx, item.patch_idx, item.score) for item in evidence] == [
        (0, 1, 0.9),
        (1, 0, 0.8),
    ]
    assert evidence[0].bbox == patch_bbox_for_index(1, grid_width=2, grid_height=2)


def test_rank_chunk_alignments_uses_lexicographic_metrics():
    evidence = token_topk_patch_evidence(
        [
            [0.1, 0.9, 0.2, 0.3],
            [0.8, 0.1, 0.7, 0.2],
        ],
        grid_width=2,
        grid_height=2,
        top_k=1,
    )
    chunks = [
        {
            "metadata": {
                "chunk_id": "left",
                "chunk_idx": 0,
                "bboxes_by_page": {"0": [[0, 0, 500, 500]]},
            }
        },
        {
            "metadata": {
                "chunk_id": "right",
                "chunk_idx": 1,
                "bboxes_by_page": {"0": [[500, 0, 1000, 500]]},
            }
        },
    ]

    ranked = rank_chunk_alignments(chunks, evidence, page_idx=0, total_query_tokens=2)

    assert [item.chunk_id for item in ranked] == ["right", "left"]
    assert round(ranked[0].chunk_score, 3) == round(0.9 / 1.7, 3)
    assert ranked[0].token_coverage == 0.5
    assert ranked[0].density_score > ranked[0].chunk_score


def test_rank_chunk_alignments_uses_token_coverage_as_tie_breaker():
    evidence = (
        PatchEvidence(0, 0, 0.9, patch_bbox_for_index(0, grid_width=2, grid_height=2)),
        PatchEvidence(0, 1, 0.9, patch_bbox_for_index(1, grid_width=2, grid_height=2)),
        PatchEvidence(1, 2, 0.9, patch_bbox_for_index(2, grid_width=2, grid_height=2)),
    )
    chunks = [
        {
            "metadata": {
                "chunk_id": "two-tokens",
                "chunk_idx": 0,
                "bboxes_by_page": {"0": [[500, 0, 1000, 500], [0, 500, 500, 1000]]},
            }
        },
        {
            "metadata": {
                "chunk_id": "one-token",
                "chunk_idx": 1,
                "bboxes_by_page": {"0": [[0, 0, 500, 500], [500, 0, 1000, 500]]},
            }
        },
    ]

    ranked = rank_chunk_alignments(chunks, evidence, page_idx=0, total_query_tokens=2)

    assert ranked[0].chunk_id == "two-tokens"
    assert ranked[0].chunk_score == ranked[1].chunk_score
    assert ranked[0].token_coverage > ranked[1].token_coverage
