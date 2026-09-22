/**
 * LombokVector — TypeScript tests (matches lombokvector-vectors-v1.json)
 * Run: npx tsc && node --test dist/test.js
 */

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import {
  cosineSimilarity,
  dotProduct,
  l2Distance,
  l2Norm,
  normalize,
  vecAdd,
  vecSub,
  vecMulScalar,
  batchCosine,
  batchL2,
  distanceMatrixCosine,
  VectorError,
} from './index.js';

const TOL = 1e-6;
const TOL64 = 1e-14;

function approx(a: number, b: number, tol = TOL): void {
  assert.ok(Math.abs(a - b) < tol, `expected ≈${b}, got ${a}`);
}

describe('cosineSimilarity', () => {
  it('basic', () => approx(cosineSimilarity([1, 2, 3], [4, 5, 6]), 0.9746318));
  it('identical', () => approx(cosineSimilarity([1, 0, 0], [1, 0, 0]), 1.0));
  it('orthogonal', () => approx(cosineSimilarity([1, 0, 0], [0, 1, 0]), 0.0));
  it('opposite', () => approx(cosineSimilarity([1, 2, 3], [-1, -2, -3]), -1.0));
  it('negative mixed', () =>
    approx(cosineSimilarity([-0.5, 0.3, -0.8, 0.1], [0.2, -0.7, 0.4, 0.9]), -0.4431293675255979, TOL64));
  it('768d', () => {
    const a = Array.from({ length: 768 }, (_, i) => Math.sin(i * 0.1));
    const b = Array.from({ length: 768 }, (_, i) => Math.cos(i * 0.1));
    approx(cosineSimilarity(a, b), 0.012394344943011405, 1e-10);
  });
});

describe('dotProduct', () => {
  it('basic', () => approx(dotProduct([1, 2, 3], [4, 5, 6]), 32));
  it('zeros', () => approx(dotProduct([0, 0, 0], [1, 2, 3]), 0));
  it('negative', () => approx(dotProduct([1, -2, 3], [-4, 5, -6]), -32));
  it('single', () => approx(dotProduct([7], [3]), 21));
});

describe('l2Distance', () => {
  it('basic', () => approx(l2Distance([1, 2, 3], [4, 5, 6]), 5.196152));
  it('identical', () => approx(l2Distance([1, 2, 3], [1, 2, 3]), 0));
  it('unit axes', () => approx(l2Distance([1, 0, 0], [0, 1, 0]), 1.4142135));
});

describe('normalize', () => {
  it('basic', () => {
    const r = normalize([3, 4]);
    approx(r[0]!, 0.6);
    approx(r[1]!, 0.8);
  });
  it('3d', () => {
    const r = normalize([1, 2, 3]);
    approx(r[0]!, 0.2672612419124244, TOL64);
    approx(r[1]!, 0.5345224838248488, TOL64);
    approx(r[2]!, 0.8017837257372732, TOL64);
  });
});

describe('l2Norm', () => {
  it('basic', () => approx(l2Norm([3, 4]), 5));
  it('3d', () => approx(l2Norm([1, 2, 3]), 3.7416573867739413, TOL64));
});

describe('arithmetic', () => {
  it('add', () => assert.deepStrictEqual([...vecAdd([1, 2, 3], [4, 5, 6])], [5, 7, 9]));
  it('sub', () => assert.deepStrictEqual([...vecSub([4, 5, 6], [1, 2, 3])], [3, 3, 3]));
  it('mul_scalar', () => {
    const r = vecMulScalar([1, 2, 3], 2.5);
    approx(r[0]!, 2.5);
    approx(r[1]!, 5.0);
    approx(r[2]!, 7.5);
  });
});

describe('errors', () => {
  it('dimension mismatch', () =>
    assert.throws(() => cosineSimilarity([1, 2], [1, 2, 3]), VectorError));
  it('empty vector', () => assert.throws(() => cosineSimilarity([], []), VectorError));
  it('zero magnitude', () =>
    assert.throws(() => cosineSimilarity([0, 0, 0], [1, 2, 3]), VectorError));
  it('normalize zero', () => assert.throws(() => normalize([0, 0]), VectorError));
});

describe('batch', () => {
  it('batchCosine', () => {
    const r = batchCosine([1, 0, 0], [[1, 0, 0], [0, 1, 0], [0.707, 0.707, 0]]);
    assert.strictEqual(r[0]!.index, 0); // cos=1
    assert.strictEqual(r[2]!.index, 1); // cos=0
  });
  it('batchL2', () => {
    const r = batchL2([0, 0, 0], [[1, 0, 0], [3, 4, 0]]);
    assert.strictEqual(r[0]!.index, 0);
    approx(r[0]!.score, 1.0);
    approx(r[1]!.score, 5.0);
  });
  it('distanceMatrix', () => {
    const m = distanceMatrixCosine([[1, 0], [0, 1]], [[1, 0], [0.707, 0.707]]);
    approx(m[0]![0]!, 1.0);
    approx(m[1]![0]!, 0.0);
  });
});
