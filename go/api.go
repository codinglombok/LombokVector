package lombokvector

func add[T Float](x, y T) T    { return T(x + y) }
func sub[T Float](x, y T) T    { return T(x - y) }
func l2Of[T Float](q, c []T) T { return sqrt(l2SqRaw(q, c)) }

// DotProduct returns the dot product (SPEC 3.1).
func DotProduct(a, b []float64) (float64, error) { return dot(a, b) }

// InnerProduct is DotProduct for real vectors.
func InnerProduct(a, b []float64) (float64, error) { return dot(a, b) }

// L2Norm returns the Euclidean norm (SPEC 3.2).
func L2Norm(a []float64) (float64, error) { return norm(a) }

// L2Distance returns the Euclidean distance (SPEC 3.3).
func L2Distance(a, b []float64) (float64, error) { return l2(a, b) }

// CosineSimilarity returns the cosine similarity clamped to [-1, 1] (SPEC 3.4).
func CosineSimilarity(a, b []float64) (float64, error) { return cosine(a, b) }

// Normalize returns a[i] * (1 / norm) (SPEC 3.5).
func Normalize(a []float64) ([]float64, error) { return normalize(a) }

// VecAdd returns the element-wise sum (SPEC 3.6).
func VecAdd(a, b []float64) ([]float64, error) { return elementwise(a, b, add[float64]) }

// VecSub returns the element-wise difference (SPEC 3.6).
func VecSub(a, b []float64) ([]float64, error) { return elementwise(a, b, sub[float64]) }

// VecMulScalar returns every element multiplied by s (SPEC 3.6).
func VecMulScalar(a []float64, s float64) ([]float64, error) { return scale(a, s) }

// BatchCosine ranks candidates by cosine similarity, descending; a zero
// candidate scores 0 and equal scores keep index order (SPEC 3.7).
func BatchCosine(query []float64, candidates [][]float64) ([]ScoredIndex[float64], error) {
	return batchCosine(query, candidates)
}

// BatchDot ranks candidates by dot product, descending (SPEC 3.7).
func BatchDot(query []float64, candidates [][]float64) ([]ScoredIndex[float64], error) {
	return batchBy(query, candidates, dotRaw[float64], true)
}

// BatchL2 ranks candidates by Euclidean distance, ascending (SPEC 3.7).
func BatchL2(query []float64, candidates [][]float64) ([]ScoredIndex[float64], error) {
	return batchBy(query, candidates, l2Of[float64], false)
}

// DistanceMatrixCosine returns out[i][j] = cosine(a[i], b[j]), 0 when either
// has magnitude zero (SPEC 3.8).
func DistanceMatrixCosine(a, b [][]float64) ([][]float64, error) { return matrixCosine(a, b) }

// DotProductF32 is DotProduct in binary32.
func DotProductF32(a, b []float32) (float32, error) { return dot(a, b) }

// L2NormF32 is L2Norm in binary32.
func L2NormF32(a []float32) (float32, error) { return norm(a) }

// L2DistanceF32 is L2Distance in binary32.
func L2DistanceF32(a, b []float32) (float32, error) { return l2(a, b) }

// CosineSimilarityF32 is CosineSimilarity in binary32.
func CosineSimilarityF32(a, b []float32) (float32, error) { return cosine(a, b) }

// NormalizeF32 is Normalize in binary32.
func NormalizeF32(a []float32) ([]float32, error) { return normalize(a) }

// VecAddF32 is VecAdd in binary32.
func VecAddF32(a, b []float32) ([]float32, error) { return elementwise(a, b, add[float32]) }

// VecSubF32 is VecSub in binary32.
func VecSubF32(a, b []float32) ([]float32, error) { return elementwise(a, b, sub[float32]) }

// VecMulScalarF32 is VecMulScalar in binary32.
func VecMulScalarF32(a []float32, s float32) ([]float32, error) { return scale(a, s) }

// BatchCosineF32 is BatchCosine in binary32.
func BatchCosineF32(query []float32, candidates [][]float32) ([]ScoredIndex[float32], error) {
	return batchCosine(query, candidates)
}

// BatchDotF32 is BatchDot in binary32.
func BatchDotF32(query []float32, candidates [][]float32) ([]ScoredIndex[float32], error) {
	return batchBy(query, candidates, dotRaw[float32], true)
}

// BatchL2F32 is BatchL2 in binary32.
func BatchL2F32(query []float32, candidates [][]float32) ([]ScoredIndex[float32], error) {
	return batchBy(query, candidates, l2Of[float32], false)
}

// DistanceMatrixCosineF32 is DistanceMatrixCosine in binary32.
func DistanceMatrixCosineF32(a, b [][]float32) ([][]float32, error) { return matrixCosine(a, b) }
