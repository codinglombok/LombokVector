#!/usr/bin/env python3
"""Builds vectors/lombokvector-vectors-v1.json (GP-11).

Expected outputs come from a reference implementation of SPEC_LombokVector
sections 2-5 written in this file from the SPEC: plain Python floats for
binary64 and an exact binary32 emulation (struct round trips) for binary32.
Every result is also checked against an exact rational computation
(fractions.Fraction) so that the reference itself is known to be close to
the true value. Ports must reproduce every expected bit pattern exactly.

After editing, run:

    python3 vectors/build_vectors.py
    sha256sum vectors/lombokvector-vectors-v1.json > vectors/SHA256SUMS

and update the hash in docs/SPEC_LombokVector_v<version>.md.
"""
import json
import math
import pathlib
import random
import struct
from fractions import Fraction

LANES = 8

# ---------------------------------------------------------------------------
# Arithmetic in one precision. binary32 results are exact: binary64 holds the
# exact product of two binary32 values, and rounding a binary64 sum, quotient
# or square root of binary32 operands to binary32 is correctly rounded because
# 53 >= 2*24 + 2.
# ---------------------------------------------------------------------------


def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


class P64:
    name = "f64"

    @staticmethod
    def r(x):
        return x

    @staticmethod
    def bits(x):
        return "0x%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


class P32:
    name = "f32"
    r = staticmethod(f32)

    @staticmethod
    def bits(x):
        return "0x%08x" % struct.unpack("<I", struct.pack("<f", x))[0]


def add(p, x, y):
    return p.r(x + y)


def mul(p, x, y):
    return p.r(x * y)


def sub(p, x, y):
    return p.r(x - y)


def div(p, x, y):
    if y == 0:
        if x == 0 or math.isnan(x):
            return math.nan
        return math.copysign(math.inf, x) * math.copysign(1.0, y)
    return p.r(x / y)


def sqrt(p, x):
    return p.r(math.sqrt(x)) if x >= 0 else math.nan


# ---------------------------------------------------------------------------
# SPEC section 2: the 8-lane reduction
# ---------------------------------------------------------------------------


def reduce8(p, terms):
    s = [0.0] * LANES
    for i, t in enumerate(terms):
        s[i % LANES] = add(p, s[i % LANES], t)
    left = add(p, add(p, s[0], s[1]), add(p, s[2], s[3]))
    right = add(p, add(p, s[4], s[5]), add(p, s[6], s[7]))
    return add(p, left, right)


def dot_raw(p, a, b):
    return reduce8(p, [mul(p, x, y) for x, y in zip(a, b)])


def sumsq_raw(p, a):
    return reduce8(p, [mul(p, x, x) for x in a])


def l2sq_raw(p, a, b):
    return reduce8(p, [mul(p, d, d) for d in (sub(p, x, y) for x, y in zip(a, b))])


class VErr(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def finite(x):
    if not math.isfinite(x):
        raise VErr("NON_FINITE")
    return x


def check_pair(a, b):
    if len(a) == 0:
        raise VErr("EMPTY_VECTOR")
    if len(a) != len(b):
        raise VErr("DIMENSION_MISMATCH")


def check_single(a):
    if len(a) == 0:
        raise VErr("EMPTY_VECTOR")


def clamp1(x):
    return -1.0 if x < -1.0 else 1.0 if x > 1.0 else x


# ---------------------------------------------------------------------------
# SPEC section 3: operations
# ---------------------------------------------------------------------------


def op_dot(p, a, b):
    check_pair(a, b)
    return finite(dot_raw(p, a, b))


def op_norm(p, a):
    check_single(a)
    return finite(sqrt(p, sumsq_raw(p, a)))


def op_l2(p, a, b):
    check_pair(a, b)
    return finite(sqrt(p, l2sq_raw(p, a, b)))


def cos_value(p, d, na, nb):
    return clamp1(finite(div(p, d, mul(p, na, nb))))


def op_cosine(p, a, b):
    check_pair(a, b)
    d = finite(dot_raw(p, a, b))
    na = finite(sqrt(p, sumsq_raw(p, a)))
    nb = finite(sqrt(p, sumsq_raw(p, b)))
    if na == 0 or nb == 0:
        raise VErr("ZERO_MAGNITUDE")
    return cos_value(p, d, na, nb)


def op_normalize(p, a):
    check_single(a)
    n = finite(sqrt(p, sumsq_raw(p, a)))
    if n == 0:
        raise VErr("ZERO_MAGNITUDE")
    inv = finite(div(p, 1.0, n))
    return [finite(mul(p, x, inv)) for x in a]


def op_add(p, a, b):
    check_pair(a, b)
    return [finite(add(p, x, y)) for x, y in zip(a, b)]


def op_sub(p, a, b):
    check_pair(a, b)
    return [finite(sub(p, x, y)) for x, y in zip(a, b)]


def op_scale(p, a, s):
    check_single(a)
    return [finite(mul(p, x, s)) for x in a]


def ranked(scores, descending):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i] if descending else scores[i], i))
    return [[i, scores[i]] for i in order]


def op_batch_cosine(p, q, cands):
    check_single(q)
    qn = finite(sqrt(p, sumsq_raw(p, q)))
    if qn == 0:
        raise VErr("ZERO_MAGNITUDE")
    scores = []
    for c in cands:
        if len(c) != len(q):
            raise VErr("DIMENSION_MISMATCH")
        d = finite(dot_raw(p, q, c))
        cn = finite(sqrt(p, sumsq_raw(p, c)))
        scores.append(0.0 if cn == 0 else cos_value(p, d, qn, cn))
    return ranked(scores, True)


def op_batch_dot(p, q, cands):
    check_single(q)
    scores = []
    for c in cands:
        if len(c) != len(q):
            raise VErr("DIMENSION_MISMATCH")
        scores.append(finite(dot_raw(p, q, c)))
    return ranked(scores, True)


def op_batch_l2(p, q, cands):
    check_single(q)
    scores = []
    for c in cands:
        if len(c) != len(q):
            raise VErr("DIMENSION_MISMATCH")
        scores.append(finite(sqrt(p, l2sq_raw(p, q, c))))
    return ranked(scores, False)


def op_matrix_cosine(p, A, B):
    if len(A) == 0 or len(B) == 0:
        raise VErr("EMPTY_VECTOR")
    dim = len(A[0])
    if dim == 0:
        raise VErr("EMPTY_VECTOR")
    for v in A + B:
        if len(v) != dim:
            raise VErr("DIMENSION_MISMATCH")
    na = [finite(sqrt(p, sumsq_raw(p, v))) for v in A]
    nb = [finite(sqrt(p, sumsq_raw(p, v))) for v in B]
    out = []
    for i, va in enumerate(A):
        row = []
        for j, vb in enumerate(B):
            d = finite(dot_raw(p, va, vb))
            row.append(0.0 if na[i] == 0 or nb[j] == 0 else cos_value(p, d, na[i], nb[j]))
        out.append(row)
    return out


OPS = {
    "dot": (op_dot, 2), "norm": (op_norm, 1), "l2": (op_l2, 2), "cosine": (op_cosine, 2),
    "normalize": (op_normalize, 1), "add": (op_add, 2), "sub": (op_sub, 2), "scale": (op_scale, 2),
    "batch_cosine": (op_batch_cosine, 2), "batch_dot": (op_batch_dot, 2), "batch_l2": (op_batch_l2, 2),
    "matrix_cosine": (op_matrix_cosine, 2),
}

# ---------------------------------------------------------------------------
# Exact cross-check of the reference (independent of the lane order)
# ---------------------------------------------------------------------------


def exact_check(op, args, value):
    if op not in ("dot", "l2", "norm", "cosine"):
        return
    F = Fraction
    a = [F(x) for x in args[0]]
    if op == "dot":
        true = sum(x * y for x, y in zip(a, [F(y) for y in args[1]]))
        scale = sum(abs(x * F(y)) for x, y in zip(a, args[1]))
        assert abs(F(value) - true) <= scale * F(len(a) + 2, 2**21), (op, value, float(true))
    elif op in ("l2", "norm"):
        b = [F(y) for y in args[1]] if op == "l2" else [F(0)] * len(a)
        true = math.sqrt(float(sum((x - y) ** 2 for x, y in zip(a, b))))
        assert abs(value - true) <= abs(true) * (len(a) + 2) * 2.0**-21, (op, value, true)
    elif op == "cosine":
        b = [F(y) for y in args[1]]
        d = float(sum(x * y for x, y in zip(a, b)))
        true = d / math.sqrt(float(sum(x * x for x in a)) * float(sum(y * y for y in b)))
        assert abs(value - clamp1(true)) <= (len(a) + 4) * 2.0**-20, (op, value, true)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

cases = []


def encode(p, value):
    if isinstance(value, list):
        if value and isinstance(value[0], list) and len(value[0]) == 2 and isinstance(value[0][0], int):
            return [[i, p.bits(s)] for i, s in value]
        if value and isinstance(value[0], list):
            return [[p.bits(x) for x in row] for row in value]
        return [p.bits(x) for x in value]
    return p.bits(value)


def add_case(op, args, precisions=("f64", "f32"), note=None):
    fn, arity = OPS[op]
    assert len(args) == arity
    for pname in precisions:
        p = P64 if pname == "f64" else P32
        if pname == "f32":
            cargs = []
            for a in args:
                if isinstance(a, list) and a and isinstance(a[0], list):
                    cargs.append([[f32(x) for x in v] for v in a])
                elif isinstance(a, list):
                    cargs.append([f32(x) for x in a])
                else:
                    cargs.append(f32(a))
        else:
            cargs = []
            for a in args:
                if isinstance(a, list) and a and isinstance(a[0], list):
                    cargs.append([[float(x) for x in v] for v in a])
                elif isinstance(a, list):
                    cargs.append([float(x) for x in a])
                else:
                    cargs.append(float(a))
        case = {"id": f"{op}-{pname}-{sum(1 for c in cases if c['op'] == op and c['precision'] == pname) + 1:03d}",
                "op": op, "precision": pname}
        if note:
            case["note"] = note
        case["args"] = cargs
        try:
            value = fn(p, *cargs)
            exact_check(op, cargs, value)
            case["expected"] = {"value": encode(p, value)}
        except VErr as e:
            case["expected"] = {"error": e.code}
        cases.append(case)


rng = random.Random(20261002)


def rvec(n, lo=-1.0, hi=1.0):
    return [rng.uniform(lo, hi) for _ in range(n)]


def short(x):
    # keep the JSON small: 6 significant digits, still exact binary64 values
    return float(f"{x:.6g}")


def svec(n, lo=-1.0, hi=1.0):
    return [short(x) for x in rvec(n, lo, hi)]


# --- hand-picked inputs with simple exact answers --------------------------
add_case("dot", [[1, 2, 3], [4, 5, 6]], note="32")
add_case("dot", [[1, 0, 0], [0, 1, 0]], note="orthogonal")
add_case("dot", [[-1.5, 2.5], [2, -4]], note="-13")
add_case("dot", [[0.1] * 10, [0.1] * 10], note="rounding visible: lane order matters")
add_case("dot", [[1e16, 1, -1e16, 1], [1, 1, 1, 1]], note="cancellation across lanes")
add_case("dot", [list(range(1, 18)), [1] * 17], note="17 elements: second pass over lanes 0")
add_case("norm", [[3, 4]], note="5")
add_case("norm", [[1, 1, 1, 1]], note="2")
add_case("norm", [[0, 0, 0]], note="zero vector has norm 0")
add_case("l2", [[1, 2, 3], [4, 5, 6]], note="sqrt(27)")
add_case("l2", [[1, 2, 3], [1, 2, 3]], note="0")
add_case("cosine", [[1, 2, 3], [4, 5, 6]])
add_case("cosine", [[1, 0, 0], [1, 0, 0]], note="identical: 1")
add_case("cosine", [[1, 0, 0], [0, 1, 0]], note="orthogonal: 0")
add_case("cosine", [[1, 2, 3], [-1, -2, -3]], note="opposite: -1")
add_case("cosine", [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], note="clamped to 1 if rounding exceeds it")
add_case("cosine", [[3, 4], [6, 8]], note="parallel, different length")
add_case("normalize", [[3, 4]])
add_case("normalize", [[1, 2, 3]])
add_case("normalize", [[-2, 0, 0, 0]])
add_case("add", [[1, 2, 3], [0.5, -2, 1e-3]])
add_case("sub", [[1, 2, 3], [0.5, -2, 1e-3]])
add_case("scale", [[1, -2, 3.5], 0.1])
add_case("scale", [[1, -2, 3.5], 0])

# --- errors ---------------------------------------------------------------------
add_case("dot", [[], []], note="EMPTY_VECTOR")
add_case("dot", [[1, 2], [1, 2, 3]], note="DIMENSION_MISMATCH")
add_case("l2", [[1], []], note="DIMENSION_MISMATCH")
add_case("cosine", [[], [1]], note="empty is checked before the length")
add_case("cosine", [[0, 0], [1, 2]], note="ZERO_MAGNITUDE")
add_case("cosine", [[1, 2], [0, 0]], note="ZERO_MAGNITUDE")
add_case("normalize", [[0, 0, 0]], note="ZERO_MAGNITUDE")
add_case("normalize", [[]], note="EMPTY_VECTOR")
add_case("norm", [[]], note="EMPTY_VECTOR")
add_case("add", [[1], [1, 2]], note="DIMENSION_MISMATCH")
add_case("scale", [[], 2], note="EMPTY_VECTOR")
add_case("dot", [[1e200, 1e200], [1e200, 1e200]], ("f64",), note="overflow: NON_FINITE")
add_case("dot", [[1e30, 1e30], [1e30, 1e30]], ("f32",), note="binary32 overflow: NON_FINITE")
add_case("norm", [[1e300, 1e300]], ("f64",), note="sum of squares overflows: NON_FINITE")
add_case("add", [[1.7e308], [1.7e308]], ("f64",), note="NON_FINITE")
add_case("cosine", [[1e-200, 0], [1e-200, 0]], ("f64",), note="norms underflow to 0: ZERO_MAGNITUDE")

# --- random inputs of many sizes ------------------------------------------------------
for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 127]:
    a, b = svec(n), svec(n)
    add_case("dot", [a, b])
    add_case("cosine", [a, b])
    add_case("l2", [a, b])
for n in [6, 13, 24, 50]:
    a = svec(n, -100, 100)
    add_case("norm", [a])
    add_case("normalize", [a])
for n in [384, 768]:
    a, b = rvec(n), rvec(n)
    add_case("cosine", [a, b], note=f"{n} dimensions, full-precision inputs")
    add_case("dot", [a, b], note=f"{n} dimensions, full-precision inputs")
# integer-based generator (no libm functions, so the file rebuilds identically everywhere)
a = [((i * 37) % 101) / 50 - 1 for i in range(768)]
b = [((i * 53 + 7) % 97) / 48 - 1 for i in range(768)]
add_case("cosine", [a, b], note="a[i] = ((37 i) mod 101) / 50 - 1, b[i] = ((53 i + 7) mod 97) / 48 - 1, 768 dimensions")
add_case("l2", [a, b], note="a[i] = ((37 i) mod 101) / 50 - 1, b[i] = ((53 i + 7) mod 97) / 48 - 1, 768 dimensions")

# --- batch and matrix -------------------------------------------------------------------
q = svec(10)
cands = [svec(10) for _ in range(12)]
cands[3] = [0.0] * 10
cands[7] = list(cands[2])
add_case("batch_cosine", [q, cands], note="zero candidate scores 0; equal scores keep index order")
add_case("batch_dot", [q, cands], note="descending, ties by index")
add_case("batch_l2", [q, cands], note="ascending, ties by index")
add_case("batch_cosine", [q, []], note="no candidates")
add_case("batch_cosine", [[0.0] * 10, cands], note="ZERO_MAGNITUDE query")
add_case("batch_cosine", [q, [svec(10), svec(9)]], note="DIMENSION_MISMATCH")
add_case("batch_l2", [[], []], note="EMPTY_VECTOR")
add_case("batch_dot", [[1, 1], [[1, 0], [0, 1], [2, -1], [-1, 2]]], note="four equal scores keep index order")
add_case("matrix_cosine", [[svec(6) for _ in range(3)], [svec(6) for _ in range(4)]])
add_case("matrix_cosine", [[[1, 0], [0, 0]], [[1, 0], [0, 1], [1, 1]]], note="zero row gives 0")
add_case("matrix_cosine", [[], [[1, 0]]], note="EMPTY_VECTOR")
add_case("matrix_cosine", [[[1, 0]], [[1, 0, 0]]], note="DIMENSION_MISMATCH")

doc = {
    "name": "lombokvector-vectors",
    "version": 1,
    "spec": "docs/SPEC_LombokVector (sections 2-5)",
    "encoding": "expected values are IEEE 754 bit patterns: 0x + 16 hex digits (binary64) or 8 hex digits (binary32); "
                "batch results are [index, bits] pairs; matrix results are rows of bits",
    "precisions": {"f64": "every port", "f32": "ports with a binary32 API (Rust)"},
    "cases": cases,
}
out = pathlib.Path(__file__).with_name("lombokvector-vectors-v1.json")
out.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False).replace('{"id"', '\n{"id"') + "\n", encoding="utf-8")
print(f"{len(cases)} cases ({sum(c['precision'] == 'f64' for c in cases)} f64, "
      f"{sum(c['precision'] == 'f32' for c in cases)} f32) -> {out} ({out.stat().st_size // 1024} KiB)")
