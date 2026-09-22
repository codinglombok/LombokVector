package lombokvector

import (
	"math"
	"testing"
)

const tol = 1e-14

func assertClose(t *testing.T, name string, got, want, eps float64) {
	t.Helper()
	if math.Abs(got-want) > eps {
		t.Errorf("%s: got %v, want %v (diff %v)", name, got, want, math.Abs(got-want))
	}
}

// ── Cosine similarity ──

func TestCosineBasic(t *testing.T) {
	r, err := CosineSimilarity([]float64{1, 2, 3}, []float64{4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "cos_basic", r, 0.9746318461970762, tol)
}

func TestCosineIdentical(t *testing.T) {
	r, err := CosineSimilarity([]float64{1, 0, 0}, []float64{1, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "cos_identical", r, 1.0, tol)
}

func TestCosineOrthogonal(t *testing.T) {
	r, err := CosineSimilarity([]float64{1, 0, 0}, []float64{0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "cos_orthogonal", r, 0.0, tol)
}

func TestCosineOpposite(t *testing.T) {
	r, err := CosineSimilarity([]float64{1, 2, 3}, []float64{-1, -2, -3})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "cos_opposite", r, -1.0, tol)
}

func TestCosineNegativeMixed(t *testing.T) {
	r, err := CosineSimilarity([]float64{-0.5, 0.3, -0.8, 0.1}, []float64{0.2, -0.7, 0.4, 0.9})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "cos_negative_mixed", r, -0.4431293675255979, tol)
}

func TestCosine768d(t *testing.T) {
	a := make([]float64, 768)
	b := make([]float64, 768)
	for i := 0; i < 768; i++ {
		a[i] = math.Sin(float64(i) * 0.1)
		b[i] = math.Cos(float64(i) * 0.1)
	}
	r, err := CosineSimilarity(a, b)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "cos_768d", r, 0.012394344943011405, 1e-10)
}

// ── Dot product ──

func TestDotBasic(t *testing.T) {
	r, err := DotProduct([]float64{1, 2, 3}, []float64{4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "dot_basic", r, 32.0, tol)
}

func TestDotZeros(t *testing.T) {
	r, err := DotProduct([]float64{0, 0, 0}, []float64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "dot_zeros", r, 0.0, tol)
}

func TestDotNegative(t *testing.T) {
	r, err := DotProduct([]float64{1, -2, 3}, []float64{-4, 5, -6})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "dot_negative", r, -32.0, tol)
}

func TestDotSingle(t *testing.T) {
	r, err := DotProduct([]float64{7}, []float64{3})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "dot_single", r, 21.0, tol)
}

// ── L2 distance ──

func TestL2Basic(t *testing.T) {
	r, err := L2Distance([]float64{1, 2, 3}, []float64{4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "l2_basic", r, 5.196152422706632, tol)
}

func TestL2Identical(t *testing.T) {
	r, err := L2Distance([]float64{1, 2, 3}, []float64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "l2_identical", r, 0.0, tol)
}

func TestL2UnitAxes(t *testing.T) {
	r, err := L2Distance([]float64{1, 0, 0}, []float64{0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "l2_unit_axes", r, 1.4142135623730951, tol)
}

// ── L2 norm ──

func TestL2NormBasic(t *testing.T) {
	r, err := L2Norm([]float64{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "l2norm_basic", r, 5.0, tol)
}

func TestL2Norm3d(t *testing.T) {
	r, err := L2Norm([]float64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "l2norm_3d", r, 3.7416573867739413, tol)
}

// ── Normalize ──

func TestNormalizeBasic(t *testing.T) {
	r, err := Normalize([]float64{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "norm[0]", r[0], 0.6, tol)
	assertClose(t, "norm[1]", r[1], 0.8, tol)
}

func TestNormalize3d(t *testing.T) {
	expected := []float64{0.2672612419124244, 0.5345224838248488, 0.8017837257372732}
	r, err := Normalize([]float64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		assertClose(t, "norm_3d", r[i], expected[i], tol)
	}
}

// ── Arithmetic ──

func TestVecAdd(t *testing.T) {
	r, err := VecAdd([]float64{1, 2, 3}, []float64{4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{5, 7, 9}
	for i := range expected {
		assertClose(t, "add", r[i], expected[i], tol)
	}
}

func TestVecSub(t *testing.T) {
	r, err := VecSub([]float64{4, 5, 6}, []float64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{3, 3, 3}
	for i := range expected {
		assertClose(t, "sub", r[i], expected[i], tol)
	}
}

func TestVecMulScalar(t *testing.T) {
	r, err := VecMulScalar([]float64{1, 2, 3}, 2.5)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{2.5, 5.0, 7.5}
	for i := range expected {
		assertClose(t, "mul_scalar", r[i], expected[i], tol)
	}
}

// ── Batch ──

func TestBatchCosine(t *testing.T) {
	q := []float64{1, 0, 0}
	cands := [][]float64{
		{0, 1, 0},
		{1, 0, 0},
		{1, 1, 0},
	}
	r, err := BatchCosine(q, cands)
	if err != nil {
		t.Fatal(err)
	}
	if r[0].Index != 1 {
		t.Errorf("batch_cosine[0]: expected index 1, got %d", r[0].Index)
	}
	assertClose(t, "batch_cosine[0].score", r[0].Score, 1.0, tol)
}

func TestBatchL2(t *testing.T) {
	q := []float64{1, 0, 0}
	cands := [][]float64{
		{0, 1, 0},
		{1, 0, 0},
		{2, 0, 0},
	}
	r, err := BatchL2(q, cands)
	if err != nil {
		t.Fatal(err)
	}
	if r[0].Index != 1 {
		t.Errorf("batch_l2[0]: expected index 1, got %d", r[0].Index)
	}
	assertClose(t, "batch_l2[0].score", r[0].Score, 0.0, tol)
}

// ── Error handling ──

func TestEmptyVector(t *testing.T) {
	_, err := CosineSimilarity([]float64{}, []float64{})
	if err != ErrEmptyVector {
		t.Errorf("expected ErrEmptyVector, got %v", err)
	}
}

func TestDimensionMismatch(t *testing.T) {
	_, err := DotProduct([]float64{1, 2}, []float64{1, 2, 3})
	if err != ErrDimensionMismatch {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestZeroMagnitude(t *testing.T) {
	_, err := Normalize([]float64{0, 0, 0})
	if err != ErrZeroMagnitude {
		t.Errorf("expected ErrZeroMagnitude, got %v", err)
	}
}
