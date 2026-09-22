"""
LombokVector — Pure Python vector math for RAG systems.
Zero dependencies. Part of LombokRAGFrameworks (@codinglombok).
License: Apache-2.0
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

__version__ = "0.1.0"
__all__ = [
    "VectorError",
    "cosine_similarity",
    "dot_product",
    "l2_distance",
    "inner_product",
    "l2_norm",
    "normalize",
    "vec_add",
    "vec_sub",
    "vec_mul_scalar",
    "batch_cosine",
    "batch_l2",
    "distance_matrix_cosine",
]


class VectorError(Exception):
    """Raised on invalid vector operations."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _check_pair(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) == 0:
        raise VectorError("EMPTY_VECTOR", "empty vector")
    if len(a) != len(b):
        raise VectorError(
            "DIMENSION_MISMATCH",
            f"dimension mismatch: expected {len(a)}, got {len(b)}",
        )


def _check_single(a: Sequence[float]) -> None:
    if len(a) == 0:
        raise VectorError("EMPTY_VECTOR", "empty vector")


# ── Core kernels (4× unrolled) ──


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    chunks = n & ~3
    s = 0.0
    i = 0
    while i < chunks:
        s += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3]
        i += 4
    while i < n:
        s += a[i] * b[i]
        i += 1
    return s


def _sum_sq(a: Sequence[float]) -> float:
    n = len(a)
    chunks = n & ~3
    s = 0.0
    i = 0
    while i < chunks:
        s += a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] + a[i + 3] * a[i + 3]
        i += 4
    while i < n:
        s += a[i] * a[i]
        i += 1
    return s


def _l2_sq(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    chunks = n & ~3
    s = 0.0
    i = 0
    while i < chunks:
        d0 = a[i] - b[i]
        d1 = a[i + 1] - b[i + 1]
        d2 = a[i + 2] - b[i + 2]
        d3 = a[i + 3] - b[i + 3]
        s += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
        i += 4
    while i < n:
        d = a[i] - b[i]
        s += d * d
        i += 1
    return s


# ── Public API ──


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]."""
    _check_pair(a, b)
    dot = _dot(a, b)
    na = math.sqrt(_sum_sq(a))
    nb = math.sqrt(_sum_sq(b))
    if na == 0 or nb == 0:
        raise VectorError("ZERO_MAGNITUDE", "zero magnitude vector")
    return dot / (na * nb)


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product."""
    _check_pair(a, b)
    return _dot(a, b)


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean (L2) distance."""
    _check_pair(a, b)
    return math.sqrt(_l2_sq(a, b))


def inner_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Inner product (alias for dot product)."""
    return dot_product(a, b)


def l2_norm(a: Sequence[float]) -> float:
    """L2 norm (magnitude)."""
    _check_single(a)
    return math.sqrt(_sum_sq(a))


def normalize(a: Sequence[float]) -> List[float]:
    """Normalize to unit length."""
    _check_single(a)
    norm = math.sqrt(_sum_sq(a))
    if norm == 0:
        raise VectorError("ZERO_MAGNITUDE", "zero magnitude vector")
    inv = 1.0 / norm
    return [x * inv for x in a]


def vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Element-wise addition."""
    _check_pair(a, b)
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Element-wise subtraction."""
    _check_pair(a, b)
    return [x - y for x, y in zip(a, b)]


def vec_mul_scalar(a: Sequence[float], s: float) -> List[float]:
    """Scalar multiplication."""
    _check_single(a)
    return [x * s for x in a]


# ── Batch operations ──


def batch_cosine(
    query: Sequence[float], candidates: Sequence[Sequence[float]]
) -> List[Tuple[int, float]]:
    """Query vs N candidates, sorted descending by cosine similarity."""
    _check_single(query)
    qn = math.sqrt(_sum_sq(query))
    if qn == 0:
        raise VectorError("ZERO_MAGNITUDE", "zero magnitude query")
    results: List[Tuple[int, float]] = []
    for i, c in enumerate(candidates):
        if len(c) != len(query):
            raise VectorError(
                "DIMENSION_MISMATCH",
                f"candidate {i}: expected {len(query)}, got {len(c)}",
            )
        dot = _dot(query, c)
        cn = math.sqrt(_sum_sq(c))
        score = dot / (qn * cn) if cn != 0 else 0.0
        results.append((i, score))
    results.sort(key=lambda x: -x[1])
    return results


def batch_l2(
    query: Sequence[float], candidates: Sequence[Sequence[float]]
) -> List[Tuple[int, float]]:
    """Query vs N candidates, sorted ascending by L2 distance."""
    _check_single(query)
    results: List[Tuple[int, float]] = []
    for i, c in enumerate(candidates):
        if len(c) != len(query):
            raise VectorError(
                "DIMENSION_MISMATCH",
                f"candidate {i}: expected {len(query)}, got {len(c)}",
            )
        results.append((i, math.sqrt(_l2_sq(query, c))))
    results.sort(key=lambda x: x[1])
    return results


def distance_matrix_cosine(
    vectors_a: Sequence[Sequence[float]],
    vectors_b: Sequence[Sequence[float]],
) -> List[List[float]]:
    """N×M cosine similarity matrix."""
    if not vectors_a or not vectors_b:
        raise VectorError("EMPTY_VECTOR", "empty vector set")
    norms_a = [math.sqrt(_sum_sq(v)) for v in vectors_a]
    norms_b = [math.sqrt(_sum_sq(v)) for v in vectors_b]
    matrix: List[List[float]] = []
    for i, va in enumerate(vectors_a):
        row: List[float] = []
        for j, vb in enumerate(vectors_b):
            denom = norms_a[i] * norms_b[j]
            row.append(_dot(va, vb) / denom if denom != 0 else 0.0)
        matrix.append(row)
    return matrix
