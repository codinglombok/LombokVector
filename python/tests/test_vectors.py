"""Runs the binary64 cases of the shared vectors: every bit pattern must match."""

import json
import pathlib
import struct

import pytest

import lombokvector as lv

DOC = json.loads((pathlib.Path(__file__).resolve().parents[2] / "vectors" / "lombokvector-vectors-v1.json").read_text(encoding="utf-8"))
CASES = [c for c in DOC["cases"] if c["precision"] == "f64"]


def bits(x):
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def run(op, a):
    if op == "dot":
        return bits(lv.dot_product(a[0], a[1]))
    if op == "norm":
        return bits(lv.l2_norm(a[0]))
    if op == "l2":
        return bits(lv.l2_distance(a[0], a[1]))
    if op == "cosine":
        return bits(lv.cosine_similarity(a[0], a[1]))
    if op == "normalize":
        return [bits(x) for x in lv.normalize(a[0])]
    if op == "add":
        return [bits(x) for x in lv.vec_add(a[0], a[1])]
    if op == "sub":
        return [bits(x) for x in lv.vec_sub(a[0], a[1])]
    if op == "scale":
        return [bits(x) for x in lv.vec_mul_scalar(a[0], a[1])]
    if op in ("batch_cosine", "batch_dot", "batch_l2"):
        fn = {"batch_cosine": lv.batch_cosine, "batch_dot": lv.batch_dot, "batch_l2": lv.batch_l2}[op]
        return [[r.index, bits(r.score)] for r in fn(a[0], a[1])]
    if op == "matrix_cosine":
        return [[bits(x) for x in row] for row in lv.distance_matrix_cosine(a[0], a[1])]
    raise ValueError(op)


def test_at_least_100_cases():
    assert len(CASES) >= 100


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_vector(case):
    try:
        got = {"value": run(case["op"], case["args"])}
    except lv.VectorError as e:
        got = {"error": e.code}
    assert got == case["expected"]
