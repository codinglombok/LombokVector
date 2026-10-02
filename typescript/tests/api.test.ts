import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as lv from '../src/index.js';

const code = (c: lv.VectorErrorCode) => (e: unknown) => e instanceof lv.VectorError && e.code === c;

test('basic values', () => {
    assert.equal(lv.dotProduct([1, 2, 3], [4, 5, 6]), 32);
    assert.equal(lv.innerProduct([1, 2, 3], [4, 5, 6]), 32);
    assert.equal(lv.l2Norm([3, 4]), 5);
    assert.equal(lv.l2Distance([1, 2, 3], [4, 5, 6]), Math.sqrt(27));
    assert.equal(lv.cosineSimilarity([1, 0], [0, 1]), 0);
    assert.deepEqual(Array.from(lv.normalize([3, 4])), [3 * 0.2, 4 * 0.2]);
    assert.deepEqual(Array.from(lv.vecAdd([1, 2], [3, 4])), [4, 6]);
    assert.deepEqual(Array.from(lv.vecSub([1, 2], [3, 4])), [-2, -2]);
    assert.deepEqual(Array.from(lv.vecMulScalar([1, 2], 3)), [3, 6]);
});

test('typed arrays are accepted', () => {
    assert.equal(lv.dotProduct(new Float64Array([1, 2]), new Float32Array([3, 4])), 11);
});

test('lane order: (1e16 + 1) + (-1e16 + 1) = 0', () => {
    assert.equal(lv.dotProduct([1e16, 1, -1e16, 1], [1, 1, 1, 1]), 0);
});

test('errors', () => {
    assert.throws(() => lv.dotProduct([], []), code('EMPTY_VECTOR'));
    assert.throws(() => lv.dotProduct([1], [1, 2]), code('DIMENSION_MISMATCH'));
    assert.throws(() => lv.cosineSimilarity([0, 0], [1, 1]), code('ZERO_MAGNITUDE'));
    assert.throws(() => lv.normalize([0]), code('ZERO_MAGNITUDE'));
    assert.throws(() => lv.l2Norm([Number.NaN]), code('NON_FINITE'));
    assert.throws(() => lv.vecAdd([Infinity], [1]), code('NON_FINITE'));
    assert.throws(() => lv.vecMulScalar([], 1), code('EMPTY_VECTOR'));
    assert.throws(() => lv.batchCosine([0, 0], []), code('ZERO_MAGNITUDE'));
    assert.throws(() => lv.batchDot([1], [[1, 2]]), code('DIMENSION_MISMATCH'));
    assert.throws(() => lv.distanceMatrixCosine([], [[1]]), code('EMPTY_VECTOR'));
    assert.throws(() => lv.distanceMatrixCosine([[]], [[]]), code('EMPTY_VECTOR'));
    assert.throws(() => lv.distanceMatrixCosine([[1, 0]], [[1]]), code('DIMENSION_MISMATCH'));
    const e = new lv.VectorError('NON_FINITE', 'x');
    assert.equal(e.message, 'NON_FINITE: x');
    assert.equal(e.name, 'VectorError');
});

test('batch ranking keeps index order on ties', () => {
    const r = lv.batchDot([1, 1], [[1, 0], [0, 1], [2, -1], [-1, 2]]);
    assert.deepEqual(r.map(x => x.index), [0, 1, 2, 3]);
    const l = lv.batchL2([0, 0], [[3, 4], [1, 0], [0, 1]]);
    assert.deepEqual(l, [{ index: 1, score: 1 }, { index: 2, score: 1 }, { index: 0, score: 5 }]);
    const c = lv.batchCosine([1, 1], [[0, 0], [1, 1]]);
    assert.deepEqual(c.map(x => x.index), [1, 0]);
    assert.deepEqual(lv.distanceMatrixCosine([[1, 0], [0, 0]], [[1, 0]]), [[1], [0]]);
});
