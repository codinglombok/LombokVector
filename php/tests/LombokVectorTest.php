<?php

declare(strict_types=1);

namespace CodingLombok\LombokVector\Tests;

use CodingLombok\LombokVector\LombokVector;
use CodingLombok\LombokVector\VectorError;
use PHPUnit\Framework\TestCase;

/**
 * LombokVector test suite — matches shared test vectors.
 * Tolerance: f64 ±1e-14 (PHP uses double precision).
 */
final class LombokVectorTest extends TestCase
{
    private const TOL = 1e-14;
    private const TOL_F32 = 1e-6;

    // ── Cosine similarity ──

    public function testCosineBasic(): void
    {
        $r = LombokVector::cosineSimilarity([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        $this->assertEqualsWithDelta(0.9746318461970762, $r, self::TOL);
    }

    public function testCosineIdentical(): void
    {
        $r = LombokVector::cosineSimilarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        $this->assertEqualsWithDelta(1.0, $r, self::TOL);
    }

    public function testCosineOrthogonal(): void
    {
        $r = LombokVector::cosineSimilarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        $this->assertEqualsWithDelta(0.0, $r, self::TOL);
    }

    public function testCosineOpposite(): void
    {
        $r = LombokVector::cosineSimilarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]);
        $this->assertEqualsWithDelta(-1.0, $r, self::TOL);
    }

    public function testCosineNegativeMixed(): void
    {
        $r = LombokVector::cosineSimilarity([-0.5, 0.3, -0.8, 0.1], [0.2, -0.7, 0.4, 0.9]);
        $this->assertEqualsWithDelta(-0.4431293675255979, $r, self::TOL);
    }

    public function testCosine768d(): void
    {
        $a = [];
        $b = [];
        for ($i = 0; $i < 768; $i++) {
            $a[] = sin($i * 0.1);
            $b[] = cos($i * 0.1);
        }
        $r = LombokVector::cosineSimilarity($a, $b);
        $this->assertEqualsWithDelta(0.012394344943011405, $r, 1e-10);
    }

    // ── Dot product ──

    public function testDotBasic(): void
    {
        $r = LombokVector::dotProduct([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        $this->assertEqualsWithDelta(32.0, $r, self::TOL);
    }

    public function testDotZeros(): void
    {
        $r = LombokVector::dotProduct([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        $this->assertEqualsWithDelta(0.0, $r, self::TOL);
    }

    public function testDotNegative(): void
    {
        $r = LombokVector::dotProduct([1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]);
        $this->assertEqualsWithDelta(-32.0, $r, self::TOL);
    }

    public function testDotSingle(): void
    {
        $r = LombokVector::dotProduct([7.0], [3.0]);
        $this->assertEqualsWithDelta(21.0, $r, self::TOL);
    }

    // ── L2 distance ──

    public function testL2Basic(): void
    {
        $r = LombokVector::l2Distance([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        $this->assertEqualsWithDelta(5.196152422706632, $r, self::TOL);
    }

    public function testL2Identical(): void
    {
        $r = LombokVector::l2Distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
        $this->assertEqualsWithDelta(0.0, $r, self::TOL);
    }

    public function testL2UnitAxes(): void
    {
        $r = LombokVector::l2Distance([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        $this->assertEqualsWithDelta(1.4142135623730951, $r, self::TOL);
    }

    // ── Inner product ──

    public function testInnerProduct(): void
    {
        $r = LombokVector::innerProduct([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        $this->assertEqualsWithDelta(32.0, $r, self::TOL);
    }

    // ── L2 norm ──

    public function testL2NormBasic(): void
    {
        $r = LombokVector::l2Norm([3.0, 4.0]);
        $this->assertEqualsWithDelta(5.0, $r, self::TOL);
    }

    public function testL2Norm3d(): void
    {
        $r = LombokVector::l2Norm([1.0, 2.0, 3.0]);
        $this->assertEqualsWithDelta(3.7416573867739413, $r, self::TOL);
    }

    // ── Normalize ──

    public function testNormalizeBasic(): void
    {
        $r = LombokVector::normalize([3.0, 4.0]);
        $this->assertEqualsWithDelta(0.6, $r[0], self::TOL);
        $this->assertEqualsWithDelta(0.8, $r[1], self::TOL);
    }

    public function testNormalize3d(): void
    {
        $expected = [0.2672612419124244, 0.5345224838248488, 0.8017837257372732];
        $r = LombokVector::normalize([1.0, 2.0, 3.0]);
        for ($i = 0; $i < 3; $i++) {
            $this->assertEqualsWithDelta($expected[$i], $r[$i], self::TOL);
        }
    }

    public function testNormalizeAlreadyUnit(): void
    {
        $r = LombokVector::normalize([1.0, 0.0, 0.0]);
        $this->assertEqualsWithDelta(1.0, $r[0], self::TOL);
        $this->assertEqualsWithDelta(0.0, $r[1], self::TOL);
        $this->assertEqualsWithDelta(0.0, $r[2], self::TOL);
    }

    // ── Arithmetic ──

    public function testVecAdd(): void
    {
        $r = LombokVector::vecAdd([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        $this->assertSame([5.0, 7.0, 9.0], $r);
    }

    public function testVecSub(): void
    {
        $r = LombokVector::vecSub([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]);
        $this->assertSame([3.0, 3.0, 3.0], $r);
    }

    public function testVecMulScalar(): void
    {
        $r = LombokVector::vecMulScalar([1.0, 2.0, 3.0], 2.5);
        $this->assertEqualsWithDelta(2.5, $r[0], self::TOL);
        $this->assertEqualsWithDelta(5.0, $r[1], self::TOL);
        $this->assertEqualsWithDelta(7.5, $r[2], self::TOL);
    }

    // ── Batch ──

    public function testBatchCosine(): void
    {
        $q = [1.0, 0.0, 0.0];
        $cands = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        $r = LombokVector::batchCosine($q, $cands);
        // Sorted descending: index 1 (1.0), index 2 (~0.707), index 0 (0.0)
        $this->assertSame(1, $r[0]['index']);
        $this->assertEqualsWithDelta(1.0, $r[0]['score'], self::TOL);
        $this->assertSame(2, $r[1]['index']);
        $this->assertSame(0, $r[2]['index']);
    }

    public function testBatchL2(): void
    {
        $q = [1.0, 0.0, 0.0];
        $cands = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        $r = LombokVector::batchL2($q, $cands);
        // Sorted ascending: index 1 (0.0), index 0 (√2), index 2 (1.0)
        $this->assertSame(1, $r[0]['index']);
        $this->assertEqualsWithDelta(0.0, $r[0]['score'], self::TOL);
    }

    // ── Error handling ──

    public function testEmptyVector(): void
    {
        $this->expectException(VectorError::class);
        LombokVector::cosineSimilarity([], []);
    }

    public function testDimensionMismatch(): void
    {
        $this->expectException(VectorError::class);
        LombokVector::dotProduct([1.0, 2.0], [1.0, 2.0, 3.0]);
    }

    public function testZeroMagnitude(): void
    {
        $this->expectException(VectorError::class);
        LombokVector::normalize([0.0, 0.0, 0.0]);
    }
}
