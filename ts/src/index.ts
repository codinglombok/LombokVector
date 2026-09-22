/**
 * LombokVector — Pure TypeScript vector math for RAG systems.
 * Zero dependencies. Part of LombokRAGFrameworks (@codinglombok).
 * License: Apache-2.0
 */

// ============================================================================
// Error
// ============================================================================

export class VectorError extends Error {
  constructor(
    public readonly code: 'DIMENSION_MISMATCH' | 'EMPTY_VECTOR' | 'ZERO_MAGNITUDE',
    message: string,
  ) {
    super(message);
    this.name = 'VectorError';
  }
}

function checkPair(a: ArrayLike<number>, b: ArrayLike<number>): void {
  if (a.length === 0) throw new VectorError('EMPTY_VECTOR', 'empty vector');
  if (a.length !== b.length)
    throw new VectorError(
      'DIMENSION_MISMATCH',
      `dimension mismatch: expected ${a.length}, got ${b.length}`,
    );
}

function checkSingle(a: ArrayLike<number>): void {
  if (a.length === 0) throw new VectorError('EMPTY_VECTOR', 'empty vector');
}

// ============================================================================
// Core kernels (scalar, 4× unrolled for ILP)
// ============================================================================

function dotKernel(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const len = a.length;
  const chunks = (len >>> 2) << 2; // floor to multiple of 4
  let sum = 0;
  let i = 0;
  for (; i < chunks; i += 4) {
    sum += a[i]! * b[i]! + a[i + 1]! * b[i + 1]! + a[i + 2]! * b[i + 2]! + a[i + 3]! * b[i + 3]!;
  }
  for (; i < len; i++) {
    sum += a[i]! * b[i]!;
  }
  return sum;
}

function sumSqKernel(a: ArrayLike<number>): number {
  const len = a.length;
  const chunks = (len >>> 2) << 2;
  let sum = 0;
  let i = 0;
  for (; i < chunks; i += 4) {
    sum += a[i]! * a[i]! + a[i + 1]! * a[i + 1]! + a[i + 2]! * a[i + 2]! + a[i + 3]! * a[i + 3]!;
  }
  for (; i < len; i++) {
    sum += a[i]! * a[i]!;
  }
  return sum;
}

function l2SqKernel(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const len = a.length;
  const chunks = (len >>> 2) << 2;
  let sum = 0;
  let i = 0;
  for (; i < chunks; i += 4) {
    const d0 = a[i]! - b[i]!;
    const d1 = a[i + 1]! - b[i + 1]!;
    const d2 = a[i + 2]! - b[i + 2]!;
    const d3 = a[i + 3]! - b[i + 3]!;
    sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
  }
  for (; i < len; i++) {
    const d = a[i]! - b[i]!;
    sum += d * d;
  }
  return sum;
}

// ============================================================================
// Public API
// ============================================================================

/** Cosine similarity. Returns value in [-1, 1]. */
export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  checkPair(a, b);
  const dot = dotKernel(a, b);
  const normA = Math.sqrt(sumSqKernel(a));
  const normB = Math.sqrt(sumSqKernel(b));
  if (normA === 0 || normB === 0) throw new VectorError('ZERO_MAGNITUDE', 'zero magnitude');
  return dot / (normA * normB);
}

/** Dot product. */
export function dotProduct(a: ArrayLike<number>, b: ArrayLike<number>): number {
  checkPair(a, b);
  return dotKernel(a, b);
}

/** Euclidean (L2) distance. */
export function l2Distance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  checkPair(a, b);
  return Math.sqrt(l2SqKernel(a, b));
}

/** Inner product (alias for dot product on real vectors). */
export function innerProduct(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return dotProduct(a, b);
}

/** L2 norm (magnitude). */
export function l2Norm(a: ArrayLike<number>): number {
  checkSingle(a);
  return Math.sqrt(sumSqKernel(a));
}

/** Normalize to unit length. Returns new Float64Array. */
export function normalize(a: ArrayLike<number>): Float64Array {
  checkSingle(a);
  const norm = Math.sqrt(sumSqKernel(a));
  if (norm === 0) throw new VectorError('ZERO_MAGNITUDE', 'zero magnitude');
  const inv = 1 / norm;
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! * inv;
  return out;
}

/** Element-wise addition. */
export function vecAdd(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  checkPair(a, b);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! + b[i]!;
  return out;
}

/** Element-wise subtraction. */
export function vecSub(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  checkPair(a, b);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! - b[i]!;
  return out;
}

/** Scalar multiplication. */
export function vecMulScalar(a: ArrayLike<number>, s: number): Float64Array {
  checkSingle(a);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! * s;
  return out;
}

// ============================================================================
// Batch operations
// ============================================================================

export interface ScoredIndex {
  index: number;
  score: number;
}

/** Batch cosine similarity: query vs N candidates, sorted descending. */
export function batchCosine(query: ArrayLike<number>, candidates: ArrayLike<number>[]): ScoredIndex[] {
  checkSingle(query);
  const qNorm = Math.sqrt(sumSqKernel(query));
  if (qNorm === 0) throw new VectorError('ZERO_MAGNITUDE', 'zero magnitude query');

  const results: ScoredIndex[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i]!;
    if (c.length !== query.length)
      throw new VectorError('DIMENSION_MISMATCH', `candidate ${i}: expected ${query.length}, got ${c.length}`);
    const dot = dotKernel(query, c);
    const cNorm = Math.sqrt(sumSqKernel(c));
    results.push({ index: i, score: cNorm === 0 ? 0 : dot / (qNorm * cNorm) });
  }
  return results.sort((a, b) => b.score - a.score);
}

/** Batch L2 distance: query vs N candidates, sorted ascending. */
export function batchL2(query: ArrayLike<number>, candidates: ArrayLike<number>[]): ScoredIndex[] {
  checkSingle(query);
  const results: ScoredIndex[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i]!;
    if (c.length !== query.length)
      throw new VectorError('DIMENSION_MISMATCH', `candidate ${i}: expected ${query.length}, got ${c.length}`);
    results.push({ index: i, score: Math.sqrt(l2SqKernel(query, c)) });
  }
  return results.sort((a, b) => a.score - b.score);
}

/** N×M cosine similarity matrix. */
export function distanceMatrixCosine(
  vectorsA: ArrayLike<number>[],
  vectorsB: ArrayLike<number>[],
): number[][] {
  if (vectorsA.length === 0 || vectorsB.length === 0)
    throw new VectorError('EMPTY_VECTOR', 'empty vector set');

  const normsA = vectorsA.map((v) => Math.sqrt(sumSqKernel(v)));
  const normsB = vectorsB.map((v) => Math.sqrt(sumSqKernel(v)));

  return vectorsA.map((va, i) =>
    vectorsB.map((vb, j) => {
      const denom = normsA[i]! * normsB[j]!;
      return denom === 0 ? 0 : dotKernel(va, vb) / denom;
    }),
  );
}
