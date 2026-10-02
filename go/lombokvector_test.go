package lombokvector

import (
	"errors"
	"math"
	"testing"
)

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func wantErr(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("got %v, want %v", err, target)
	}
}

func TestFloat64API(t *testing.T) {
	if must(DotProduct([]float64{1, 2, 3}, []float64{4, 5, 6})) != 32 {
		t.Fatal("dot")
	}
	if must(InnerProduct([]float64{1, 2}, []float64{3, 4})) != 11 {
		t.Fatal("inner")
	}
	if must(L2Norm([]float64{3, 4})) != 5 {
		t.Fatal("norm")
	}
	if must(L2Distance([]float64{1, 2, 3}, []float64{4, 5, 6})) != math.Sqrt(27) {
		t.Fatal("l2")
	}
	if must(CosineSimilarity([]float64{1, 0}, []float64{0, 1})) != 0 {
		t.Fatal("cosine")
	}
	n := must(Normalize([]float64{3, 4}))
	if n[0] != 0.6000000000000001 || n[1] != 0.8 { // a[i] * (1 / norm)
		t.Fatal(n)
	}
	if v := must(VecAdd([]float64{1, 2}, []float64{3, 4})); v[0] != 4 || v[1] != 6 {
		t.Fatal(v)
	}
	if v := must(VecSub([]float64{1, 2}, []float64{3, 4})); v[0] != -2 || v[1] != -2 {
		t.Fatal(v)
	}
	if v := must(VecMulScalar([]float64{1, 2}, 3)); v[1] != 6 {
		t.Fatal(v)
	}
	if must(DotProduct([]float64{1e16, 1, -1e16, 1}, []float64{1, 1, 1, 1})) != 0 {
		t.Fatal("lane order")
	}
}

func TestFloat32API(t *testing.T) {
	if must(DotProductF32([]float32{1, 2, 3}, []float32{4, 5, 6})) != 32 {
		t.Fatal("dot")
	}
	if must(L2NormF32([]float32{3, 4})) != 5 {
		t.Fatal("norm")
	}
	if must(L2DistanceF32([]float32{0, 0}, []float32{3, 4})) != 5 {
		t.Fatal("l2")
	}
	if must(CosineSimilarityF32([]float32{1, 1}, []float32{2, 2})) != 1 {
		t.Fatal("cosine")
	}
	if v := must(NormalizeF32([]float32{3, 4})); v[0] != float32(3)*float32(0.2) {
		t.Fatal(v)
	}
	if v := must(VecAddF32([]float32{1}, []float32{2})); v[0] != 3 {
		t.Fatal(v)
	}
	if v := must(VecSubF32([]float32{1}, []float32{2})); v[0] != -1 {
		t.Fatal(v)
	}
	if v := must(VecMulScalarF32([]float32{1}, 2)); v[0] != 2 {
		t.Fatal(v)
	}
	r := must(BatchCosineF32([]float32{1, 1}, [][]float32{{0, 0}, {1, 1}}))
	if r[0].Index != 1 || r[1].Score != 0 {
		t.Fatal(r)
	}
	if r := must(BatchDotF32([]float32{1}, [][]float32{{1}, {2}})); r[0].Index != 1 {
		t.Fatal(r)
	}
	if r := must(BatchL2F32([]float32{0}, [][]float32{{2}, {1}})); r[0].Index != 1 {
		t.Fatal(r)
	}
	if m := must(DistanceMatrixCosineF32([][]float32{{1, 0}}, [][]float32{{1, 0}, {0, 0}})); m[0][0] != 1 || m[0][1] != 0 {
		t.Fatal(m)
	}
}

func TestBatchAndMatrix(t *testing.T) {
	r := must(BatchDot([]float64{1, 1}, [][]float64{{1, 0}, {0, 1}, {2, -1}, {-1, 2}}))
	for i, x := range r {
		if x.Index != i {
			t.Fatal("ties must keep index order", r)
		}
	}
	l := must(BatchL2([]float64{0, 0}, [][]float64{{3, 4}, {1, 0}, {0, 1}}))
	if l[0].Index != 1 || l[1].Index != 2 || l[2].Score != 5 {
		t.Fatal(l)
	}
	c := must(BatchCosine([]float64{1, 1}, [][]float64{{0, 0}, {1, 1}}))
	if c[0].Index != 1 {
		t.Fatal(c)
	}
	m := must(DistanceMatrixCosine([][]float64{{1, 0}, {0, 0}}, [][]float64{{1, 0}}))
	if m[0][0] != 1 || m[1][0] != 0 {
		t.Fatal(m)
	}
}

func TestErrors(t *testing.T) {
	_, err := DotProduct(nil, nil)
	wantErr(t, err, ErrEmptyVector)
	_, err = DotProduct([]float64{1}, []float64{1, 2})
	wantErr(t, err, ErrDimensionMismatch)
	if err.Error() != "DIMENSION_MISMATCH: expected 1 elements, got 2" {
		t.Fatal(err.Error())
	}
	_, err = CosineSimilarity([]float64{0, 0}, []float64{1, 1})
	wantErr(t, err, ErrZeroMagnitude)
	_, err = CosineSimilarity([]float64{1, math.NaN()}, []float64{1, 1})
	wantErr(t, err, ErrNonFinite)
	_, err = CosineSimilarity([]float64{1, 1}, []float64{math.Inf(1), 1})
	wantErr(t, err, ErrNonFinite)
	_, err = CosineSimilarity([]float64{1e200, 1e200}, []float64{1, 1})
	wantErr(t, err, ErrNonFinite)
	_, err = L2Norm(nil)
	wantErr(t, err, ErrEmptyVector)
	_, err = L2Distance([]float64{math.MaxFloat64}, []float64{-math.MaxFloat64})
	wantErr(t, err, ErrNonFinite)
	_, err = Normalize(nil)
	wantErr(t, err, ErrEmptyVector)
	_, err = Normalize([]float64{0})
	wantErr(t, err, ErrZeroMagnitude)
	_, err = Normalize([]float64{math.Inf(1)})
	wantErr(t, err, ErrNonFinite)
	_, err = VecAdd([]float64{math.MaxFloat64}, []float64{math.MaxFloat64})
	wantErr(t, err, ErrNonFinite)
	_, err = VecAdd([]float64{1}, nil)
	wantErr(t, err, ErrDimensionMismatch)
	_, err = VecMulScalar(nil, 1)
	wantErr(t, err, ErrEmptyVector)
	_, err = VecMulScalar([]float64{2}, math.MaxFloat64)
	wantErr(t, err, ErrNonFinite)
	_, err = BatchCosine(nil, nil)
	wantErr(t, err, ErrEmptyVector)
	_, err = BatchCosine([]float64{0}, nil)
	wantErr(t, err, ErrZeroMagnitude)
	_, err = BatchCosine([]float64{math.Inf(1)}, nil)
	wantErr(t, err, ErrNonFinite)
	_, err = BatchCosine([]float64{1}, [][]float64{{1, 2}})
	wantErr(t, err, ErrDimensionMismatch)
	_, err = BatchCosine([]float64{1}, [][]float64{{math.NaN()}})
	wantErr(t, err, ErrNonFinite)
	_, err = BatchCosine([]float64{0, 1}, [][]float64{{1e200, 0}})
	wantErr(t, err, ErrNonFinite)
	_, err = BatchDot(nil, nil)
	wantErr(t, err, ErrEmptyVector)
	_, err = BatchDot([]float64{1}, [][]float64{{1, 2}})
	wantErr(t, err, ErrDimensionMismatch)
	_, err = BatchDot([]float64{math.MaxFloat64}, [][]float64{{2}})
	wantErr(t, err, ErrNonFinite)
	_, err = DistanceMatrixCosine(nil, [][]float64{{1}})
	wantErr(t, err, ErrEmptyVector)
	_, err = DistanceMatrixCosine([][]float64{{}}, [][]float64{{}})
	wantErr(t, err, ErrEmptyVector)
	_, err = DistanceMatrixCosine([][]float64{{1, 0}}, [][]float64{{1}})
	wantErr(t, err, ErrDimensionMismatch)
	_, err = DistanceMatrixCosine([][]float64{{math.NaN()}}, [][]float64{{1}})
	wantErr(t, err, ErrNonFinite)
	_, err = DistanceMatrixCosine([][]float64{{1}}, [][]float64{{math.Inf(1)}})
	wantErr(t, err, ErrNonFinite)
	_, err = DistanceMatrixCosine([][]float64{{1e200}}, [][]float64{{1e200}})
	wantErr(t, err, ErrNonFinite)
	if itoa(0) != "0" || itoa(1536) != "1536" {
		t.Fatal("itoa")
	}
}
