"""LombokVector for Python: binary64 vector math whose results are
bit-identical to the Rust, TypeScript, Go and PHP ports. Every sum uses the
8-lane order of docs/SPEC_LombokVector_v0.2.0.md section 2. Pure Python,
no dependencies."""

from __future__ import annotations

import math
from typing import List, NamedTuple, Sequence

__version__ = "0.2.0"
__all__ = [
    "ScoredIndex",
    "VectorError",
    "batch_cosine",
    "batch_dot",
    "batch_l2",
    "cosine_similarity",
    "distance_matrix_cosine",
    "dot_product",
    "inner_product",
    "l2_distance",
    "l2_norm",
    "normalize",
    "vec_add",
    "vec_mul_scalar",
    "vec_sub",
]

Vector = Sequence[float]


class VectorError(ValueError):
    """Raised by every operation; ``code`` is stable across ports:
    ``DIMENSION_MISMATCH``, ``EMPTY_VECTOR``, ``ZERO_MAGNITUDE``, ``NON_FINITE``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class ScoredIndex(NamedTuple):
    """A candidate's position in the input and its score."""

    index: int
    score: float


def _check_pair(a: Vector, b: Vector) -> None:
    if len(a) == 0:
        raise VectorError("EMPTY_VECTOR", "vector has no elements")
    if len(a) != len(b):
        raise VectorError("DIMENSION_MISMATCH", f"expected {len(a)} elements, got {len(b)}")


def _check_single(a: Vector) -> None:
    if len(a) == 0:
        raise VectorError("EMPTY_VECTOR", "vector has no elements")


def _finite(x: float) -> float:
    if not math.isfinite(x):
        raise VectorError("NON_FINITE", "result is infinite or NaN")
    return x


# --- SPEC section 2: the 8-lane reduction ------------------------------------


def _combine(s: List[float]) -> float:
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]))


def _dot(a: Vector, b: Vector) -> float:
    s = [0.0] * 8
    for i in range(len(a)):
        s[i & 7] += float(a[i]) * float(b[i])
    return _combine(s)


def _sumsq(a: Vector) -> float:
    s = [0.0] * 8
    for i in range(len(a)):
        x = float(a[i])
        s[i & 7] += x * x
    return _combine(s)


def _l2sq(a: Vector, b: Vector) -> float:
    s = [0.0] * 8
    for i in range(len(a)):
        d = float(a[i]) - float(b[i])
        s[i & 7] += d * d
    return _combine(s)


def _sqrt(x: float) -> float:
    # x is a finite sum of squares, so it is never negative
    return _finite(math.sqrt(x))


def _div(x: float, y: float) -> float:
    if y == 0.0:  # IEEE 754 gives an infinity or NaN; Python raises instead
        raise VectorError("NON_FINITE", "result is infinite or NaN")
    return _finite(x / y)


def _cos(d: float, na: float, nb: float) -> float:
    c = _div(d, na * nb)
    return -1.0 if c < -1.0 else 1.0 if c > 1.0 else c


# --- SPEC section 3: operations ----------------------------------------------


def dot_product(a: Vector, b: Vector) -> float:
    """Dot product."""
    _check_pair(a, b)
    return _finite(_dot(a, b))


def inner_product(a: Vector, b: Vector) -> float:
    """Same as :func:`dot_product` for real vectors."""
    return dot_product(a, b)


def l2_norm(a: Vector) -> float:
    """Euclidean norm."""
    _check_single(a)
    return _sqrt(_finite(_sumsq(a)))


def l2_distance(a: Vector, b: Vector) -> float:
    """Euclidean distance."""
    _check_pair(a, b)
    return _sqrt(_finite(_l2sq(a, b)))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity clamped to [-1, 1]; ``ZERO_MAGNITUDE`` when either norm is 0."""
    _check_pair(a, b)
    d = _finite(_dot(a, b))
    na = _sqrt(_finite(_sumsq(a)))
    nb = _sqrt(_finite(_sumsq(b)))
    if na == 0.0 or nb == 0.0:
        raise VectorError("ZERO_MAGNITUDE", "vector has magnitude zero")
    return _cos(d, na, nb)


def normalize(a: Vector) -> List[float]:
    """Unit-length copy: ``a[i] * (1 / norm)``."""
    _check_single(a)
    n = _sqrt(_finite(_sumsq(a)))
    if n == 0.0:
        raise VectorError("ZERO_MAGNITUDE", "vector has magnitude zero")
    inv = _div(1.0, n)
    return [_finite(float(x) * inv) for x in a]


def vec_add(a: Vector, b: Vector) -> List[float]:
    """Element-wise sum."""
    _check_pair(a, b)
    return [_finite(float(x) + float(y)) for x, y in zip(a, b)]


def vec_sub(a: Vector, b: Vector) -> List[float]:
    """Element-wise difference."""
    _check_pair(a, b)
    return [_finite(float(x) - float(y)) for x, y in zip(a, b)]


def vec_mul_scalar(a: Vector, s: float) -> List[float]:
    """Every element multiplied by ``s``."""
    _check_single(a)
    s = float(s)
    return [_finite(float(x) * s) for x in a]


# --- batch and matrix ----------------------------------------------------------


def _rank(scores: List[float], descending: bool) -> List[ScoredIndex]:
    # sorted() is stable, so equal scores keep index order
    order = sorted(range(len(scores)), key=(lambda i: -scores[i]) if descending else (lambda i: scores[i]))
    return [ScoredIndex(i, scores[i]) for i in order]


def _check_candidate(query: Vector, c: Vector) -> None:
    if len(c) != len(query):
        raise VectorError("DIMENSION_MISMATCH", f"expected {len(query)} elements, got {len(c)}")


def batch_cosine(query: Vector, candidates: Sequence[Vector]) -> List[ScoredIndex]:
    """Cosine similarity of ``query`` with each candidate, descending; a zero candidate scores 0."""
    _check_single(query)
    qn = _sqrt(_finite(_sumsq(query)))
    if qn == 0.0:
        raise VectorError("ZERO_MAGNITUDE", "query has magnitude zero")
    scores = []
    for c in candidates:
        _check_candidate(query, c)
        d = _finite(_dot(query, c))
        cn = _sqrt(_finite(_sumsq(c)))
        scores.append(0.0 if cn == 0.0 else _cos(d, qn, cn))
    return _rank(scores, True)


def batch_dot(query: Vector, candidates: Sequence[Vector]) -> List[ScoredIndex]:
    """Dot product of ``query`` with each candidate, descending."""
    _check_single(query)
    scores = []
    for c in candidates:
        _check_candidate(query, c)
        scores.append(_finite(_dot(query, c)))
    return _rank(scores, True)


def batch_l2(query: Vector, candidates: Sequence[Vector]) -> List[ScoredIndex]:
    """Euclidean distance from ``query`` to each candidate, ascending."""
    _check_single(query)
    scores = []
    for c in candidates:
        _check_candidate(query, c)
        scores.append(_sqrt(_finite(_l2sq(query, c))))
    return _rank(scores, False)


def distance_matrix_cosine(a: Sequence[Vector], b: Sequence[Vector]) -> List[List[float]]:
    """``out[i][j]`` = cosine similarity of ``a[i]`` and ``b[j]``; 0 when either has magnitude zero."""
    if len(a) == 0 or len(b) == 0:
        raise VectorError("EMPTY_VECTOR", "vector set is empty")
    first = a[0]
    _check_single(first)
    for v in list(a) + list(b):
        _check_candidate(first, v)
    na = [_sqrt(_finite(_sumsq(v))) for v in a]
    nb = [_sqrt(_finite(_sumsq(v))) for v in b]
    out = []
    for i, va in enumerate(a):
        row = []
        for j, vb in enumerate(b):
            d = _finite(_dot(va, vb))
            row.append(0.0 if na[i] == 0.0 or nb[j] == 0.0 else _cos(d, na[i], nb[j]))
        out.append(row)
    return out
