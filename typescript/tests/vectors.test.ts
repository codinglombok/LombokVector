// Runs the binary64 cases of the shared vectors: every bit pattern must match.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as lv from '../src/index.js';

interface Case { id: string; op: string; precision: string; args: unknown[]; expected: unknown }

const path = fileURLToPath(new URL('../../../vectors/lombokvector-vectors-v1.json', import.meta.url));
const doc = JSON.parse(readFileSync(path, 'utf8')) as { cases: Case[] };

const view = new DataView(new ArrayBuffer(8));
function bits(x: number): string {
    view.setFloat64(0, x);
    return '0x' + view.getBigUint64(0).toString(16).padStart(16, '0');
}
const list = (v: ArrayLike<number>): string[] => Array.from(v, bits);
const ranked = (v: lv.ScoredIndex[]): [number, string][] => v.map(r => [r.index, bits(r.score)]);

function run(op: string, a: unknown[]): unknown {
    const v = a as number[][] & number[][][];
    switch (op) {
        case 'dot': return bits(lv.dotProduct(v[0]!, v[1]!));
        case 'norm': return bits(lv.l2Norm(v[0]!));
        case 'l2': return bits(lv.l2Distance(v[0]!, v[1]!));
        case 'cosine': return bits(lv.cosineSimilarity(v[0]!, v[1]!));
        case 'normalize': return list(lv.normalize(v[0]!));
        case 'add': return list(lv.vecAdd(v[0]!, v[1]!));
        case 'sub': return list(lv.vecSub(v[0]!, v[1]!));
        case 'scale': return list(lv.vecMulScalar(v[0]!, a[1] as number));
        case 'batch_cosine': return ranked(lv.batchCosine(a[0] as number[], a[1] as number[][]));
        case 'batch_dot': return ranked(lv.batchDot(a[0] as number[], a[1] as number[][]));
        case 'batch_l2': return ranked(lv.batchL2(a[0] as number[], a[1] as number[][]));
        case 'matrix_cosine': return lv.distanceMatrixCosine(a[0] as number[][], a[1] as number[][]).map(list);
        default: throw new Error(`unknown op ${op}`);
    }
}

const cases = doc.cases.filter(c => c.precision === 'f64');

test('vector file has at least 100 binary64 cases', () => {
    assert.ok(cases.length >= 100);
});

for (const c of cases) {
    test(`vector ${c.id}`, () => {
        let got: unknown;
        try {
            got = { value: run(c.op, c.args) };
        } catch (e) {
            assert.ok(e instanceof lv.VectorError, String(e));
            got = { error: e.code };
        }
        assert.deepStrictEqual(got, c.expected);
    });
}
