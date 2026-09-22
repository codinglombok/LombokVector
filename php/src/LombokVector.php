<?php

declare(strict_types=1);

/**
 * LombokVector — Pure PHP vector math for RAG systems.
 * Zero dependencies. Part of LombokRAGFrameworks (@codinglombok).
 * License: Apache-2.0
 */

namespace CodingLombok\LombokVector;

final class VectorError extends \RuntimeException
{
    public function __construct(
        public readonly string $code,
        string $message,
    ) {
        parent::__construct($message);
    }
}

final class LombokVector
{
    // ── Validation ──

    /** @param float[] $a @param float[] $b */
    private static function checkPair(array $a, array $b): void
    {
        if (count($a) === 0) {
            throw new VectorError('EMPTY_VECTOR', 'empty vector');
        }
        if (count($a) !== count($b)) {
            throw new VectorError(
                'DIMENSION_MISMATCH',
                'dimension mismatch: expected ' . count($a) . ', got ' . count($b),
            );
        }
    }

    /** @param float[] $a */
    private static function checkSingle(array $a): void
    {
        if (count($a) === 0) {
            throw new VectorError('EMPTY_VECTOR', 'empty vector');
        }
    }

    // ── Core kernels (4× unrolled) ──

    /** @param float[] $a @param float[] $b */
    private static function dotKernel(array $a, array $b): float
    {
        $n = count($a);
        $chunks = $n & ~3;
        $s = 0.0;
        $i = 0;
        while ($i < $chunks) {
            $s += $a[$i] * $b[$i] + $a[$i + 1] * $b[$i + 1]
                + $a[$i + 2] * $b[$i + 2] + $a[$i + 3] * $b[$i + 3];
            $i += 4;
        }
        while ($i < $n) {
            $s += $a[$i] * $b[$i];
            $i++;
        }
        return $s;
    }

    /** @param float[] $a */
    private static function sumSqKernel(array $a): float
    {
        $n = count($a);
        $chunks = $n & ~3;
        $s = 0.0;
        $i = 0;
        while ($i < $chunks) {
            $s += $a[$i] * $a[$i] + $a[$i + 1] * $a[$i + 1]
                + $a[$i + 2] * $a[$i + 2] + $a[$i + 3] * $a[$i + 3];
            $i += 4;
        }
        while ($i < $n) {
            $s += $a[$i] * $a[$i];
            $i++;
        }
        return $s;
    }

    /** @param float[] $a @param float[] $b */
    private static function l2SqKernel(array $a, array $b): float
    {
        $n = count($a);
        $chunks = $n & ~3;
        $s = 0.0;
        $i = 0;
        while ($i < $chunks) {
            $d0 = $a[$i] - $b[$i];
            $d1 = $a[$i + 1] - $b[$i + 1];
            $d2 = $a[$i + 2] - $b[$i + 2];
            $d3 = $a[$i + 3] - $b[$i + 3];
            $s += $d0 * $d0 + $d1 * $d1 + $d2 * $d2 + $d3 * $d3;
            $i += 4;
        }
        while ($i < $n) {
            $d = $a[$i] - $b[$i];
            $s += $d * $d;
            $i++;
        }
        return $s;
    }

    // ── Public API ──

    /** Cosine similarity in [-1, 1]. @param float[] $a @param float[] $b */
    public static function cosineSimilarity(array $a, array $b): float
    {
        self::checkPair($a, $b);
        $dot = self::dotKernel($a, $b);
        $na = sqrt(self::sumSqKernel($a));
        $nb = sqrt(self::sumSqKernel($b));
        if ($na == 0.0 || $nb == 0.0) {
            throw new VectorError('ZERO_MAGNITUDE', 'zero magnitude vector');
        }
        return $dot / ($na * $nb);
    }

    /** Dot product. @param float[] $a @param float[] $b */
    public static function dotProduct(array $a, array $b): float
    {
        self::checkPair($a, $b);
        return self::dotKernel($a, $b);
    }

    /** Euclidean (L2) distance. @param float[] $a @param float[] $b */
    public static function l2Distance(array $a, array $b): float
    {
        self::checkPair($a, $b);
        return sqrt(self::l2SqKernel($a, $b));
    }

    /** Inner product (alias). @param float[] $a @param float[] $b */
    public static function innerProduct(array $a, array $b): float
    {
        return self::dotProduct($a, $b);
    }

    /** L2 norm (magnitude). @param float[] $a */
    public static function l2Norm(array $a): float
    {
        self::checkSingle($a);
        return sqrt(self::sumSqKernel($a));
    }

    /** Normalize to unit length. @param float[] $a @return float[] */
    public static function normalize(array $a): array
    {
        self::checkSingle($a);
        $norm = sqrt(self::sumSqKernel($a));
        if ($norm == 0.0) {
            throw new VectorError('ZERO_MAGNITUDE', 'zero magnitude vector');
        }
        $inv = 1.0 / $norm;
        return array_map(static fn(float $x): float => $x * $inv, $a);
    }

    /** Element-wise addition. @param float[] $a @param float[] $b @return float[] */
    public static function vecAdd(array $a, array $b): array
    {
        self::checkPair($a, $b);
        $out = [];
        for ($i = 0, $n = count($a); $i < $n; $i++) {
            $out[] = $a[$i] + $b[$i];
        }
        return $out;
    }

    /** Element-wise subtraction. @param float[] $a @param float[] $b @return float[] */
    public static function vecSub(array $a, array $b): array
    {
        self::checkPair($a, $b);
        $out = [];
        for ($i = 0, $n = count($a); $i < $n; $i++) {
            $out[] = $a[$i] - $b[$i];
        }
        return $out;
    }

    /** Scalar multiplication. @param float[] $a @return float[] */
    public static function vecMulScalar(array $a, float $s): array
    {
        self::checkSingle($a);
        return array_map(static fn(float $x): float => $x * $s, $a);
    }

    /**
     * Batch cosine similarity: query vs candidates, sorted descending.
     * @param float[] $query
     * @param float[][] $candidates
     * @return array<int, array{index: int, score: float}>
     */
    public static function batchCosine(array $query, array $candidates): array
    {
        self::checkSingle($query);
        $qn = sqrt(self::sumSqKernel($query));
        if ($qn == 0.0) {
            throw new VectorError('ZERO_MAGNITUDE', 'zero magnitude query');
        }
        $results = [];
        foreach ($candidates as $i => $c) {
            if (count($c) !== count($query)) {
                throw new VectorError('DIMENSION_MISMATCH', "candidate $i: dimension mismatch");
            }
            $dot = self::dotKernel($query, $c);
            $cn = sqrt(self::sumSqKernel($c));
            $score = $cn != 0.0 ? $dot / ($qn * $cn) : 0.0;
            $results[] = ['index' => $i, 'score' => $score];
        }
        usort($results, static fn($a, $b) => $b['score'] <=> $a['score']);
        return $results;
    }

    /**
     * Batch L2 distance: query vs candidates, sorted ascending.
     * @param float[] $query
     * @param float[][] $candidates
     * @return array<int, array{index: int, score: float}>
     */
    public static function batchL2(array $query, array $candidates): array
    {
        self::checkSingle($query);
        $results = [];
        foreach ($candidates as $i => $c) {
            if (count($c) !== count($query)) {
                throw new VectorError('DIMENSION_MISMATCH', "candidate $i: dimension mismatch");
            }
            $results[] = ['index' => $i, 'score' => sqrt(self::l2SqKernel($query, $c))];
        }
        usort($results, static fn($a, $b) => $a['score'] <=> $b['score']);
        return $results;
    }
}
