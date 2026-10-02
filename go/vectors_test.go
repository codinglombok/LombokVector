package lombokvector

// Runs the shared vectors (binary64 and binary32): every bit pattern must match.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"
)

type vcase struct {
	ID        string            `json:"id"`
	Op        string            `json:"op"`
	Precision string            `json:"precision"`
	Args      []json.RawMessage `json:"args"`
	Expected  any               `json:"expected"`
}

func bits64(x float64) string { return fmt.Sprintf("0x%016x", math.Float64bits(x)) }
func bits32(x float32) string { return fmt.Sprintf("0x%08x", math.Float32bits(x)) }

func decode[V any](raw json.RawMessage) V {
	var v V
	if err := json.Unmarshal(raw, &v); err != nil {
		panic(err)
	}
	return v
}

func to32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

func runCase[T Float](op string, args []json.RawMessage, conv func([]float64) []T, bits func(T) string) (any, error) {
	vec := func(i int) []T { return conv(decode[[]float64](args[i])) }
	set := func(i int) [][]T {
		raw := decode[[][]float64](args[i])
		out := make([][]T, len(raw))
		for j, v := range raw {
			out[j] = conv(v)
		}
		return out
	}
	list := func(v []T) []any {
		out := make([]any, len(v))
		for i, x := range v {
			out[i] = bits(x)
		}
		return out
	}
	ranked := func(v []ScoredIndex[T]) []any {
		out := make([]any, len(v))
		for i, r := range v {
			out[i] = []any{float64(r.Index), bits(r.Score)}
		}
		return out
	}
	scalar := func(x T, err error) (any, error) { return bits(x), err }
	vector := func(v []T, err error) (any, error) { return list(v), err }
	rankedR := func(v []ScoredIndex[T], err error) (any, error) { return ranked(v), err }
	switch op {
	case "dot":
		return scalar(dot(vec(0), vec(1)))
	case "norm":
		return scalar(norm(vec(0)))
	case "l2":
		return scalar(l2(vec(0), vec(1)))
	case "cosine":
		return scalar(cosine(vec(0), vec(1)))
	case "normalize":
		return vector(normalize(vec(0)))
	case "add":
		return vector(elementwise(vec(0), vec(1), add[T]))
	case "sub":
		return vector(elementwise(vec(0), vec(1), sub[T]))
	case "scale":
		return vector(scale(vec(0), conv([]float64{decode[float64](args[1])})[0]))
	case "batch_cosine":
		return rankedR(batchCosine(vec(0), set(1)))
	case "batch_dot":
		return rankedR(batchBy(vec(0), set(1), dotRaw[T], true))
	case "batch_l2":
		return rankedR(batchBy(vec(0), set(1), l2Of[T], false))
	case "matrix_cosine":
		m, err := matrixCosine(set(0), set(1))
		rows := make([]any, len(m))
		for i, r := range m {
			rows[i] = list(r)
		}
		return rows, err
	}
	panic("unknown op " + op)
}

func TestVectors(t *testing.T) {
	raw, err := os.ReadFile("../vectors/lombokvector-vectors-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct{ Cases []vcase }
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc.Cases) < 100 {
		t.Fatalf("only %d cases", len(doc.Cases))
	}
	for _, c := range doc.Cases {
		var v any
		var err error
		if c.Precision == "f64" {
			v, err = runCase(c.Op, c.Args, func(x []float64) []float64 { return x }, bits64)
		} else {
			v, err = runCase(c.Op, c.Args, to32, bits32)
		}
		var got map[string]any
		var e *Error
		switch {
		case err == nil:
			got = map[string]any{"value": v}
		case errors.As(err, &e):
			got = map[string]any{"error": e.Code}
		default:
			t.Fatalf("%s: unexpected error %v", c.ID, err)
		}
		if !reflect.DeepEqual(got, c.Expected) {
			gj, _ := json.Marshal(got)
			wj, _ := json.Marshal(c.Expected)
			t.Errorf("%s: got %s want %s", c.ID, gj, wj)
		}
	}
}
