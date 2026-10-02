import math

import pytest

import lombokvector as lv


def code(c):
    return pytest.raises(lv.VectorError, match=f"^{c}:")


def test_basic_values():
    assert lv.dot_product([1, 2, 3], [4, 5, 6]) == 32.0
    assert lv.inner_product([1, 2, 3], [4, 5, 6]) == 32.0
    assert lv.l2_norm([3, 4]) == 5.0
    assert lv.l2_distance([1, 2, 3], [4, 5, 6]) == math.sqrt(27)
    assert lv.cosine_similarity([1, 0], [0, 1]) == 0.0
    assert lv.normalize([3, 4]) == [3 * 0.2, 4 * 0.2]
    assert lv.vec_add([1, 2], [3, 4]) == [4.0, 6.0]
    assert lv.vec_sub([1, 2], [3, 4]) == [-2.0, -2.0]
    assert lv.vec_mul_scalar([1, 2], 3) == [3.0, 6.0]
    assert lv.vec_mul_scalar([-2], 0) == [-0.0] and math.copysign(1, lv.vec_mul_scalar([-2], 0)[0]) == -1


def test_lane_order():
    assert lv.dot_product([1e16, 1, -1e16, 1], [1, 1, 1, 1]) == 0.0


def test_errors():
    with code("EMPTY_VECTOR"):
        lv.dot_product([], [])
    with code("DIMENSION_MISMATCH"):
        lv.dot_product([1], [1, 2])
    with code("ZERO_MAGNITUDE"):
        lv.cosine_similarity([0, 0], [1, 1])
    with code("ZERO_MAGNITUDE"):
        lv.normalize([0])
    with code("NON_FINITE"):
        lv.l2_norm([math.nan])
    with code("NON_FINITE"):
        lv.vec_add([math.inf], [1])
    with code("NON_FINITE"):
        lv.cosine_similarity([1e200, 1e200], [1, 1])
    with code("EMPTY_VECTOR"):
        lv.vec_mul_scalar([], 1)
    with code("ZERO_MAGNITUDE"):
        lv.batch_cosine([0, 0], [])
    with code("DIMENSION_MISMATCH"):
        lv.batch_dot([1], [[1, 2]])
    with code("DIMENSION_MISMATCH"):
        lv.batch_l2([1], [[1, 2]])
    with code("EMPTY_VECTOR"):
        lv.distance_matrix_cosine([], [[1]])
    with code("EMPTY_VECTOR"):
        lv.distance_matrix_cosine([[]], [[]])
    with code("DIMENSION_MISMATCH"):
        lv.distance_matrix_cosine([[1, 0]], [[1]])
    e = lv.VectorError("NON_FINITE", "x")
    assert isinstance(e, ValueError) and e.code == "NON_FINITE" and str(e) == "NON_FINITE: x"


def test_batch_ranking():
    r = lv.batch_dot([1, 1], [[1, 0], [0, 1], [2, -1], [-1, 2]])
    assert [x.index for x in r] == [0, 1, 2, 3]
    assert lv.batch_l2([0, 0], [[3, 4], [1, 0], [0, 1]]) == [(1, 1.0), (2, 1.0), (0, 5.0)]
    assert [x.index for x in lv.batch_cosine([1, 1], [[0, 0], [1, 1]])] == [1, 0]
    assert lv.distance_matrix_cosine([[1, 0], [0, 0]], [[1, 0]]) == [[1.0], [0.0]]
    assert lv.__version__ == "0.2.0"
