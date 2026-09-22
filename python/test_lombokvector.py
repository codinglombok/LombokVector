"""LombokVector — Python tests (matches lombokvector-vectors-v1.json)"""

import math
import unittest
from lombokvector import (
    VectorError,
    cosine_similarity,
    dot_product,
    l2_distance,
    l2_norm,
    normalize,
    vec_add,
    vec_sub,
    vec_mul_scalar,
    batch_cosine,
    batch_l2,
    distance_matrix_cosine,
)

TOL = 1e-6
TOL64 = 1e-14


class TestCosine(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [4, 5, 6]), 0.9746318461970762, places=6)

    def test_identical(self):
        self.assertAlmostEqual(cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0, places=10)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0, 0], [0, 1, 0]), 0.0, places=10)

    def test_opposite(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [-1, -2, -3]), -1.0, places=10)

    def test_negative_mixed(self):
        r = cosine_similarity([-0.5, 0.3, -0.8, 0.1], [0.2, -0.7, 0.4, 0.9])
        self.assertAlmostEqual(r, -0.4431293675255979, places=12)

    def test_768d(self):
        a = [math.sin(i * 0.1) for i in range(768)]
        b = [math.cos(i * 0.1) for i in range(768)]
        r = cosine_similarity(a, b)
        self.assertTrue(abs(r - 0.012394344943011405) < 1e-10)


class TestDot(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(dot_product([1, 2, 3], [4, 5, 6]), 32.0)

    def test_zeros(self):
        self.assertAlmostEqual(dot_product([0, 0, 0], [1, 2, 3]), 0.0)

    def test_negative(self):
        self.assertAlmostEqual(dot_product([1, -2, 3], [-4, 5, -6]), -32.0)

    def test_single(self):
        self.assertAlmostEqual(dot_product([7], [3]), 21.0)


class TestL2(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(l2_distance([1, 2, 3], [4, 5, 6]), 5.196152422706632, places=6)

    def test_identical(self):
        self.assertAlmostEqual(l2_distance([1, 2, 3], [1, 2, 3]), 0.0)

    def test_unit_axes(self):
        self.assertAlmostEqual(l2_distance([1, 0, 0], [0, 1, 0]), 1.4142135623730951, places=6)


class TestNormalize(unittest.TestCase):
    def test_basic(self):
        r = normalize([3, 4])
        self.assertAlmostEqual(r[0], 0.6, places=10)
        self.assertAlmostEqual(r[1], 0.8, places=10)

    def test_3d(self):
        r = normalize([1, 2, 3])
        self.assertAlmostEqual(r[0], 0.2672612419124244, places=14)
        self.assertAlmostEqual(r[1], 0.5345224838248488, places=14)
        self.assertAlmostEqual(r[2], 0.8017837257372732, places=14)


class TestL2Norm(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(l2_norm([3, 4]), 5.0)

    def test_3d(self):
        self.assertAlmostEqual(l2_norm([1, 2, 3]), 3.7416573867739413, places=14)


class TestArithmetic(unittest.TestCase):
    def test_add(self):
        self.assertEqual(vec_add([1, 2, 3], [4, 5, 6]), [5, 7, 9])

    def test_sub(self):
        self.assertEqual(vec_sub([4, 5, 6], [1, 2, 3]), [3, 3, 3])

    def test_mul_scalar(self):
        r = vec_mul_scalar([1, 2, 3], 2.5)
        self.assertAlmostEqual(r[0], 2.5)
        self.assertAlmostEqual(r[1], 5.0)
        self.assertAlmostEqual(r[2], 7.5)


class TestErrors(unittest.TestCase):
    def test_dimension_mismatch(self):
        with self.assertRaises(VectorError):
            cosine_similarity([1, 2], [1, 2, 3])

    def test_empty(self):
        with self.assertRaises(VectorError):
            cosine_similarity([], [])

    def test_zero_magnitude(self):
        with self.assertRaises(VectorError):
            cosine_similarity([0, 0, 0], [1, 2, 3])

    def test_normalize_zero(self):
        with self.assertRaises(VectorError):
            normalize([0, 0])


class TestBatch(unittest.TestCase):
    def test_batch_cosine(self):
        r = batch_cosine([1, 0, 0], [[1, 0, 0], [0, 1, 0], [0.707, 0.707, 0]])
        self.assertEqual(r[0][0], 0)
        self.assertEqual(r[2][0], 1)

    def test_batch_l2(self):
        r = batch_l2([0, 0, 0], [[1, 0, 0], [3, 4, 0]])
        self.assertEqual(r[0][0], 0)
        self.assertAlmostEqual(r[0][1], 1.0)
        self.assertAlmostEqual(r[1][1], 5.0)

    def test_distance_matrix(self):
        m = distance_matrix_cosine([[1, 0], [0, 1]], [[1, 0], [0.707, 0.707]])
        self.assertAlmostEqual(m[0][0], 1.0, places=6)
        self.assertAlmostEqual(m[1][0], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
