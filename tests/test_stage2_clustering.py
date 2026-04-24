from __future__ import annotations

import numpy as np

from grpocredit.voi.stage2_semantic import (
    cluster_sizes_from_labels,
    connected_component_clusters,
    semantic_entropy,
)


def test_identical_vectors_single_cluster() -> None:
    v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    labels = connected_component_clusters(v, cosine_threshold=0.5)
    assert len(set(labels)) == 1


def test_orthogonal_vectors_separate_clusters() -> None:
    v = np.eye(3)
    labels = connected_component_clusters(v, cosine_threshold=0.5)
    assert len(set(labels)) == 3


def test_cluster_sizes_sum_to_n() -> None:
    labels = [0, 0, 1, 1, 1, 2]
    sizes = cluster_sizes_from_labels(labels)
    assert sum(sizes) == len(labels)
    assert sorted(sizes, reverse=True) == [3, 2, 1]


def test_semantic_entropy_single_cluster_zero() -> None:
    assert semantic_entropy([4]) == 0.0


def test_semantic_entropy_uniform_partition() -> None:
    # 4 continuations, 4 singletons → H = log(4)
    import math

    h = semantic_entropy([1, 1, 1, 1])
    assert abs(h - math.log(4)) < 1e-9
