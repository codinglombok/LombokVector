// Package lombokvector provides vector math whose results are bit-identical
// to the Rust, TypeScript, Python and PHP ports of LombokVector: dot product,
// cosine similarity, Euclidean distance, normalisation, and batch ranking.
//
// Every sum uses the 8-lane order of docs/SPEC_LombokVector_v0.2.0.md
// section 2. Products are converted explicitly (T(x*y)) before they are
// added, which the Go specification defines as rounding to the target type
// and which prevents the compiler from fusing them into a multiply-add.
package lombokvector

import (
	"math"
	"sort"
)

// Error is the error type of this package. Code is stable across ports.
type Error struct {
	Code string
	Msg  string
}

func (e *Error) Error() string { return e.Code + ": " + e.Msg }

// Is matches errors with the same Code, so errors.Is(err, ErrDimensionMismatch)
// works even though the message carries the lengths.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Code == e.Code
}

// Sentinel errors; compare with errors.Is.
var (
	ErrDimensionMismatch = &Error{"DIMENSION_MISMATCH", "vectors have different lengths"}
	ErrEmptyVector       = &Error{"EMPTY_VECTOR", "vector has no elements"}
	ErrZeroMagnitude     = &Error{"ZERO_MAGNITUDE", "vector has magnitude zero"}
	ErrNonFinite         = &Error{"NON_FINITE", "result is infinite or NaN"}
)

// Float is the element type: float32 or float64.
type Float interface{ ~float32 | ~float64 }

func mismatch(expected, actual int) error {
	return &Error{"DIMENSION_MISMATCH", "expected " + itoa(expected) + " elements, got " + itoa(actual)}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func checkPair[T Float](a, b []T) error {
	if len(a) == 0 {
		return ErrEmptyVector
	}
	if len(a) != len(b) {
		return mismatch(len(a), len(b))
	}
	return nil
}

func checkSingle[T Float](a []T) error {
	if len(a) == 0 {
		return ErrEmptyVector
	}
	return nil
}

func isFinite[T Float](x T) bool {
	f := float64(x)
	return !math.IsInf(f, 0) && !math.IsNaN(f)
}

func finite[T Float](x T) (T, error) {
	if !isFinite(x) {
		return 0, ErrNonFinite
	}
	return x, nil
}

// --- SPEC section 2: the 8-lane reduction ---------------------------------

func combine[T Float](s *[8]T) T {
	l := T(T(s[0]+s[1]) + T(s[2]+s[3]))
	r := T(T(s[4]+s[5]) + T(s[6]+s[7]))
	return T(l + r)
}

func dotRaw[T Float](a, b []T) T {
	var s [8]T
	for i := range a {
		p := T(a[i] * b[i])
		s[i&7] = T(s[i&7] + p)
	}
	return combine(&s)
}

func sumSqRaw[T Float](a []T) T {
	var s [8]T
	for i := range a {
		p := T(a[i] * a[i])
		s[i&7] = T(s[i&7] + p)
	}
	return combine(&s)
}

func l2SqRaw[T Float](a, b []T) T {
	var s [8]T
	for i := range a {
		d := T(a[i] - b[i])
		p := T(d * d)
		s[i&7] = T(s[i&7] + p)
	}
	return combine(&s)
}

// sqrt is correctly rounded for both widths: binary64 holds a binary32 value
// exactly and 53 >= 2*24 + 2 makes the second rounding harmless.
func sqrt[T Float](x T) T { return T(math.Sqrt(float64(x))) }

func cosValue[T Float](d, na, nb T) (T, error) {
	c, err := finite(T(d / T(na*nb)))
	if err != nil {
		return 0, err
	}
	if c < -1 {
		return -1, nil
	}
	if c > 1 {
		return 1, nil
	}
	return c, nil
}

// --- SPEC section 3: generic operations -----------------------------------

func dot[T Float](a, b []T) (T, error) {
	if err := checkPair(a, b); err != nil {
		return 0, err
	}
	return finite(dotRaw(a, b))
}

func norm[T Float](a []T) (T, error) {
	if err := checkSingle(a); err != nil {
		return 0, err
	}
	return finite(sqrt(sumSqRaw(a)))
}

func l2[T Float](a, b []T) (T, error) {
	if err := checkPair(a, b); err != nil {
		return 0, err
	}
	return finite(sqrt(l2SqRaw(a, b)))
}

func cosine[T Float](a, b []T) (T, error) {
	if err := checkPair(a, b); err != nil {
		return 0, err
	}
	d, err := finite(dotRaw(a, b))
	if err != nil {
		return 0, err
	}
	na, err := finite(sqrt(sumSqRaw(a)))
	if err != nil {
		return 0, err
	}
	nb, err := finite(sqrt(sumSqRaw(b)))
	if err != nil {
		return 0, err
	}
	if na == 0 || nb == 0 {
		return 0, ErrZeroMagnitude
	}
	return cosValue(d, na, nb)
}

func normalize[T Float](a []T) ([]T, error) {
	if err := checkSingle(a); err != nil {
		return nil, err
	}
	n, err := finite(sqrt(sumSqRaw(a)))
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, ErrZeroMagnitude
	}
	inv, err := finite(T(1 / n))
	if err != nil {
		return nil, err
	}
	out := make([]T, len(a))
	for i, x := range a {
		if out[i], err = finite(T(x * inv)); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func elementwise[T Float](a, b []T, f func(x, y T) T) ([]T, error) {
	if err := checkPair(a, b); err != nil {
		return nil, err
	}
	out := make([]T, len(a))
	for i := range a {
		v, err := finite(f(a[i], b[i]))
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func scale[T Float](a []T, s T) ([]T, error) {
	if err := checkSingle(a); err != nil {
		return nil, err
	}
	out := make([]T, len(a))
	for i, x := range a {
		v, err := finite(T(x * s))
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// ScoredIndex is a candidate's position in the input and its score.
type ScoredIndex[T Float] struct {
	Index int
	Score T
}

func rank[T Float](scores []T, descending bool) []ScoredIndex[T] {
	out := make([]ScoredIndex[T], len(scores))
	for i, s := range scores {
		out[i] = ScoredIndex[T]{i, s}
	}
	// stable: equal scores keep index order
	sort.SliceStable(out, func(i, j int) bool {
		if descending {
			return out[i].Score > out[j].Score
		}
		return out[i].Score < out[j].Score
	})
	return out
}

func checkCandidate[T Float](query, c []T) error {
	if len(c) != len(query) {
		return mismatch(len(query), len(c))
	}
	return nil
}

func batchCosine[T Float](query []T, candidates [][]T) ([]ScoredIndex[T], error) {
	if err := checkSingle(query); err != nil {
		return nil, err
	}
	qn, err := finite(sqrt(sumSqRaw(query)))
	if err != nil {
		return nil, err
	}
	if qn == 0 {
		return nil, ErrZeroMagnitude
	}
	scores := make([]T, len(candidates))
	for i, c := range candidates {
		if err := checkCandidate(query, c); err != nil {
			return nil, err
		}
		d, err := finite(dotRaw(query, c))
		if err != nil {
			return nil, err
		}
		cn, err := finite(sqrt(sumSqRaw(c)))
		if err != nil {
			return nil, err
		}
		if cn != 0 {
			if scores[i], err = cosValue(d, qn, cn); err != nil {
				return nil, err
			}
		}
	}
	return rank(scores, true), nil
}

func batchBy[T Float](query []T, candidates [][]T, f func(q, c []T) T, descending bool) ([]ScoredIndex[T], error) {
	if err := checkSingle(query); err != nil {
		return nil, err
	}
	scores := make([]T, len(candidates))
	for i, c := range candidates {
		if err := checkCandidate(query, c); err != nil {
			return nil, err
		}
		v, err := finite(f(query, c))
		if err != nil {
			return nil, err
		}
		scores[i] = v
	}
	return rank(scores, descending), nil
}

func matrixCosine[T Float](a, b [][]T) ([][]T, error) {
	if len(a) == 0 || len(b) == 0 {
		return nil, ErrEmptyVector
	}
	first := a[0]
	if err := checkSingle(first); err != nil {
		return nil, err
	}
	for _, set := range [][][]T{a, b} {
		for _, v := range set {
			if err := checkCandidate(first, v); err != nil {
				return nil, err
			}
		}
	}
	norms := func(set [][]T) ([]T, error) {
		out := make([]T, len(set))
		for i, v := range set {
			n, err := finite(sqrt(sumSqRaw(v)))
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	}
	na, err := norms(a)
	if err != nil {
		return nil, err
	}
	nb, err := norms(b)
	if err != nil {
		return nil, err
	}
	out := make([][]T, len(a))
	for i, va := range a {
		out[i] = make([]T, len(b))
		for j, vb := range b {
			d, err := finite(dotRaw(va, vb))
			if err != nil {
				return nil, err
			}
			if na[i] != 0 && nb[j] != 0 {
				if out[i][j], err = cosValue(d, na[i], nb[j]); err != nil {
					return nil, err
				}
			}
		}
	}
	return out, nil
}
