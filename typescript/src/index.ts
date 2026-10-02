/**
 * LombokVector for TypeScript/JavaScript: binary64 vector math whose results
 * are bit-identical to the Rust, Python, Go and PHP ports. Every sum uses the
 * 8-lane order of docs/SPEC_LombokVector_v0.2.0.md section 2.
 */

/** Error codes shared by every port (SPEC section 4). */
export type VectorErrorCode = 'DIMENSION_MISMATCH' | 'EMPTY_VECTOR' | 'ZERO_MAGNITUDE' | 'NON_FINITE';

/** Error thrown by LombokVector; `code` is stable across ports. */
export class VectorError extends Error {
    readonly code: VectorErrorCode;

    constructor(code: VectorErrorCode, message: string) {
        super(`${code}: ${message}`);
        this.name = 'VectorError';
        this.code = code;
    }
}

/** Any array-like of numbers: `number[]`, `Float64Array`, `Float32Array`, ... */
export type Vector = ArrayLike<number>;

function checkPair(a: Vector, b: Vector): void {
    if (a.length === 0) throw new VectorError('EMPTY_VECTOR', 'vector has no elements');
    if (a.length !== b.length) {
        throw new VectorError('DIMENSION_MISMATCH', `expected ${a.length} elements, got ${b.length}`);
    }
}

function checkSingle(a: Vector): void {
    if (a.length === 0) throw new VectorError('EMPTY_VECTOR', 'vector has no elements');
}

function finite(x: number): number {
    if (!Number.isFinite(x)) throw new VectorError('NON_FINITE', 'result is infinite or NaN');
    return x;
}

// --- SPEC section 2: the 8-lane reduction -----------------------------------

function combine(s: Float64Array): number {
    return ((s[0]! + s[1]!) + (s[2]! + s[3]!)) + ((s[4]! + s[5]!) + (s[6]! + s[7]!));
}

function dotRaw(a: Vector, b: Vector): number {
    const s = new Float64Array(8);
    for (let i = 0; i < a.length; i++) s[i & 7] += a[i]! * b[i]!;
    return combine(s);
}

function sumSqRaw(a: Vector): number {
    const s = new Float64Array(8);
    for (let i = 0; i < a.length; i++) s[i & 7] += a[i]! * a[i]!;
    return combine(s);
}

function l2SqRaw(a: Vector, b: Vector): number {
    const s = new Float64Array(8);
    for (let i = 0; i < a.length; i++) {
        const d = a[i]! - b[i]!;
        s[i & 7] += d * d;
    }
    return combine(s);
}

function cosValue(dot: number, na: number, nb: number): number {
    const c = finite(dot / (na * nb));
    return c < -1 ? -1 : c > 1 ? 1 : c;
}

// --- SPEC section 3: operations ----------------------------------------------

/** Dot product. */
export function dotProduct(a: Vector, b: Vector): number {
    checkPair(a, b);
    return finite(dotRaw(a, b));
}

/** Same as {@link dotProduct} for real vectors. */
export function innerProduct(a: Vector, b: Vector): number {
    return dotProduct(a, b);
}

/** Euclidean norm. */
export function l2Norm(a: Vector): number {
    checkSingle(a);
    return finite(Math.sqrt(sumSqRaw(a)));
}

/** Euclidean distance. */
export function l2Distance(a: Vector, b: Vector): number {
    checkPair(a, b);
    return finite(Math.sqrt(l2SqRaw(a, b)));
}

/** Cosine similarity, clamped to [-1, 1]. Throws `ZERO_MAGNITUDE` when either norm is 0. */
export function cosineSimilarity(a: Vector, b: Vector): number {
    checkPair(a, b);
    const d = finite(dotRaw(a, b));
    const na = finite(Math.sqrt(sumSqRaw(a)));
    const nb = finite(Math.sqrt(sumSqRaw(b)));
    if (na === 0 || nb === 0) throw new VectorError('ZERO_MAGNITUDE', 'vector has magnitude zero');
    return cosValue(d, na, nb);
}

/** Unit-length copy: `a[i] * (1 / norm)`. */
export function normalize(a: Vector): Float64Array {
    checkSingle(a);
    const n = finite(Math.sqrt(sumSqRaw(a)));
    if (n === 0) throw new VectorError('ZERO_MAGNITUDE', 'vector has magnitude zero');
    const inv = finite(1 / n);
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = finite(a[i]! * inv);
    return out;
}

/** Element-wise sum. */
export function vecAdd(a: Vector, b: Vector): Float64Array {
    checkPair(a, b);
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = finite(a[i]! + b[i]!);
    return out;
}

/** Element-wise difference. */
export function vecSub(a: Vector, b: Vector): Float64Array {
    checkPair(a, b);
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = finite(a[i]! - b[i]!);
    return out;
}

/** Every element multiplied by `s`. */
export function vecMulScalar(a: Vector, s: number): Float64Array {
    checkSingle(a);
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = finite(a[i]! * s);
    return out;
}

// --- batch and matrix ---------------------------------------------------------

/** A candidate's position in the input and its score. */
export interface ScoredIndex {
    index: number;
    score: number;
}

function rank(scores: number[], descending: boolean): ScoredIndex[] {
    const out = scores.map((score, index) => ({ index, score }));
    // Array.prototype.sort is stable (ES2019), so equal scores keep index order
    return out.sort((x, y) => (descending ? y.score - x.score : x.score - y.score));
}

function checkCandidate(query: Vector, c: Vector): void {
    if (c.length !== query.length) {
        throw new VectorError('DIMENSION_MISMATCH', `expected ${query.length} elements, got ${c.length}`);
    }
}

/** Cosine similarity of `query` with each candidate, descending; a zero candidate scores 0. */
export function batchCosine(query: Vector, candidates: readonly Vector[]): ScoredIndex[] {
    checkSingle(query);
    const qn = finite(Math.sqrt(sumSqRaw(query)));
    if (qn === 0) throw new VectorError('ZERO_MAGNITUDE', 'query has magnitude zero');
    const scores = candidates.map(c => {
        checkCandidate(query, c);
        const d = finite(dotRaw(query, c));
        const cn = finite(Math.sqrt(sumSqRaw(c)));
        return cn === 0 ? 0 : cosValue(d, qn, cn);
    });
    return rank(scores, true);
}

/** Dot product of `query` with each candidate, descending. */
export function batchDot(query: Vector, candidates: readonly Vector[]): ScoredIndex[] {
    checkSingle(query);
    const scores = candidates.map(c => {
        checkCandidate(query, c);
        return finite(dotRaw(query, c));
    });
    return rank(scores, true);
}

/** Euclidean distance from `query` to each candidate, ascending. */
export function batchL2(query: Vector, candidates: readonly Vector[]): ScoredIndex[] {
    checkSingle(query);
    const scores = candidates.map(c => {
        checkCandidate(query, c);
        return finite(Math.sqrt(l2SqRaw(query, c)));
    });
    return rank(scores, false);
}

/** `out[i][j]` = cosine similarity of `a[i]` and `b[j]`; 0 when either has magnitude zero. */
export function distanceMatrixCosine(a: readonly Vector[], b: readonly Vector[]): number[][] {
    if (a.length === 0 || b.length === 0) throw new VectorError('EMPTY_VECTOR', 'vector set is empty');
    const first = a[0]!;
    checkSingle(first);
    for (const v of [...a, ...b]) checkCandidate(first, v);
    const na = a.map(v => finite(Math.sqrt(sumSqRaw(v))));
    const nb = b.map(v => finite(Math.sqrt(sumSqRaw(v))));
    return a.map((va, i) =>
        b.map((vb, j) => {
            const d = finite(dotRaw(va, vb));
            return na[i] === 0 || nb[j] === 0 ? 0 : cosValue(d, na[i]!, nb[j]!);
        }),
    );
}
