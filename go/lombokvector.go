// Package lombokvector provides SIMD-style vector math for RAG systems.
// Zero dependencies. Part of LombokRAGFrameworks (@codinglombok).
// License: Apache-2.0
package lombokvector

import (
	"errors"
	"math"
	"sort"
)

var (
	ErrDimensionMismatch = errors.New("lombokvector: dimension mismatch")
	ErrEmptyVector       = errors.New("lombokvector: empty vector")
	ErrZeroMagnitude     = errors.New("lombokvector: zero magnitude vector")
)

// ── Core kernels ──

func dotKernel(a, b []float64) float64 {
	n := len(a)
	chunks := n &^ 3
	var s float64
	i := 0
	for ; i < chunks; i += 4 {
		s += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
	}
	for ; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

func sumSqKernel(a []float64) float64 {
	n := len(a)
	chunks := n &^ 3
	var s float64
	i := 0
	for ; i < chunks; i += 4 {
		s += a[i]*a[i] + a[i+1]*a[i+1] + a[i+2]*a[i+2] + a[i+3]*a[i+3]
	}
	for ; i < n; i++ {
		s += a[i] * a[i]
	}
	return s
}

func l2SqKernel(a, b []float64) float64 {
	n := len(a)
	chunks := n &^ 3
	var s float64
	i := 0
	for ; i < chunks; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s += d0*d0 + d1*d1 + d2*d2 + d3*d3
	}
	for ; i < n; i++ {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func checkPair(a, b []float64) error {
	if len(a) == 0 {
		return ErrEmptyVector
	}
	if len(a) != len(b) {
		return ErrDimensionMismatch
	}
	return nil
}

func checkSingle(a []float64) error {
	if len(a) == 0 {
		return ErrEmptyVector
	}
	return nil
}

// ── Public API ──

// CosineSimilarity returns cosine similarity in [-1, 1].
func CosineSimilarity(a, b []float64) (float64, error) {
	if err := checkPair(a, b); err != nil {
		return 0, err
	}
	dot := dotKernel(a, b)
	na := math.Sqrt(sumSqKernel(a))
	nb := math.Sqrt(sumSqKernel(b))
	if na == 0 || nb == 0 {
		return 0, ErrZeroMagnitude
	}
	return dot / (na * nb), nil
}

// DotProduct returns the dot product.
func DotProduct(a, b []float64) (float64, error) {
	if err := checkPair(a, b); err != nil {
		return 0, err
	}
	return dotKernel(a, b), nil
}

// L2Distance returns the Euclidean distance.
func L2Distance(a, b []float64) (float64, error) {
	if err := checkPair(a, b); err != nil {
		return 0, err
	}
	return math.Sqrt(l2SqKernel(a, b)), nil
}

// InnerProduct is an alias for DotProduct.
func InnerProduct(a, b []float64) (float64, error) {
	return DotProduct(a, b)
}

// L2Norm returns the L2 norm (magnitude).
func L2Norm(a []float64) (float64, error) {
	if err := checkSingle(a); err != nil {
		return 0, err
	}
	return math.Sqrt(sumSqKernel(a)), nil
}

// Normalize returns a unit-length copy.
func Normalize(a []float64) ([]float64, error) {
	if err := checkSingle(a); err != nil {
		return nil, err
	}
	norm := math.Sqrt(sumSqKernel(a))
	if norm == 0 {
		return nil, ErrZeroMagnitude
	}
	inv := 1.0 / norm
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = v * inv
	}
	return out, nil
}

// VecAdd returns element-wise sum.
func VecAdd(a, b []float64) ([]float64, error) {
	if err := checkPair(a, b); err != nil {
		return nil, err
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out, nil
}

// VecSub returns element-wise difference.
func VecSub(a, b []float64) ([]float64, error) {
	if err := checkPair(a, b); err != nil {
		return nil, err
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out, nil
}

// VecMulScalar returns scalar multiplication.
func VecMulScalar(a []float64, s float64) ([]float64, error) {
	if err := checkSingle(a); err != nil {
		return nil, err
	}
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = v * s
	}
	return out, nil
}

// ── Batch operations ──

// ScoredIndex holds an index and its score.
type ScoredIndex struct {
	Index int
	Score float64
}

// BatchCosine computes cosine similarity for query vs candidates, sorted descending.
func BatchCosine(query []float64, candidates [][]float64) ([]ScoredIndex, error) {
	if err := checkSingle(query); err != nil {
		return nil, err
	}
	qn := math.Sqrt(sumSqKernel(query))
	if qn == 0 {
		return nil, ErrZeroMagnitude
	}
	results := make([]ScoredIndex, len(candidates))
	for i, c := range candidates {
		if len(c) != len(query) {
			return nil, ErrDimensionMismatch
		}
		dot := dotKernel(query, c)
		cn := math.Sqrt(sumSqKernel(c))
		score := 0.0
		if cn != 0 {
			score = dot / (qn * cn)
		}
		results[i] = ScoredIndex{Index: i, Score: score}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	return results, nil
}

// BatchL2 computes L2 distance for query vs candidates, sorted ascending.
func BatchL2(query []float64, candidates [][]float64) ([]ScoredIndex, error) {
	if err := checkSingle(query); err != nil {
		return nil, err
	}
	results := make([]ScoredIndex, len(candidates))
	for i, c := range candidates {
		if len(c) != len(query) {
			return nil, ErrDimensionMismatch
		}
		results[i] = ScoredIndex{Index: i, Score: math.Sqrt(l2SqKernel(query, c))}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score < results[j].Score })
	return results, nil
}
