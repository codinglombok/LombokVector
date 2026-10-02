<?php

declare(strict_types=1);

namespace CodingLombok\LombokVector;

/**
 * Binary64 vector math whose results are bit-identical to the Rust,
 * TypeScript, Python and Go ports. Every sum uses the 8-lane order of
 * docs/SPEC_LombokVector_v0.2.0.md section 2. Elements may be int or float;
 * they are converted to float before any arithmetic.
 */
final class LombokVector
{
    // --- SPEC section 2: the 8-lane reduction --------------------------------

    /** @param array<int, float> $s */
    private static function combine(array $s): float
    {
        return (($s[0] + $s[1]) + ($s[2] + $s[3])) + (($s[4] + $s[5]) + ($s[6] + $s[7]));
    }

    /** @param list<int|float> $a @param list<int|float> $b */
    private static function dotRaw(array $a, array $b): float
    {
        $s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        $n = count($a);
        for ($i = 0; $i < $n; $i++) {
            $s[$i & 7] += (float) $a[$i] * (float) $b[$i];
        }
        return self::combine($s);
    }

    /** @param list<int|float> $a */
    private static function sumSqRaw(array $a): float
    {
        $s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        $n = count($a);
        for ($i = 0; $i < $n; $i++) {
            $x = (float) $a[$i];
            $s[$i & 7] += $x * $x;
        }
        return self::combine($s);
    }

    /** @param list<int|float> $a @param list<int|float> $b */
    private static function l2SqRaw(array $a, array $b): float
    {
        $s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        $n = count($a);
        for ($i = 0; $i < $n; $i++) {
            $d = (float) $a[$i] - (float) $b[$i];
            $s[$i & 7] += $d * $d;
        }
        return self::combine($s);
    }

    private static function finite(float $x): float
    {
        if (!is_finite($x)) {
            throw new VectorError(VectorError::NON_FINITE, 'result is infinite or NaN');
        }
        return $x;
    }

    private static function checkPair(array $a, array $b): void
    {
        if (count($a) === 0) {
            throw new VectorError(VectorError::EMPTY_VECTOR, 'vector has no elements');
        }
        if (count($a) !== count($b)) {
            throw new VectorError(VectorError::DIMENSION_MISMATCH, sprintf('expected %d elements, got %d', count($a), count($b)));
        }
    }

    private static function checkSingle(array $a): void
    {
        if (count($a) === 0) {
            throw new VectorError(VectorError::EMPTY_VECTOR, 'vector has no elements');
        }
    }

    private static function cosValue(float $d, float $na, float $nb): float
    {
        // fdiv follows IEEE 754 (x / 0 is infinite or NaN) instead of throwing
        $c = self::finite(fdiv($d, $na * $nb));
        return $c < -1.0 ? -1.0 : ($c > 1.0 ? 1.0 : $c);
    }

    // --- SPEC section 3: operations --------------------------------------------

    /** Dot product. @param list<int|float> $a @param list<int|float> $b */
    public static function dotProduct(array $a, array $b): float
    {
        self::checkPair($a, $b);
        return self::finite(self::dotRaw(array_values($a), array_values($b)));
    }

    /** Same as dotProduct for real vectors. */
    public static function innerProduct(array $a, array $b): float
    {
        return self::dotProduct($a, $b);
    }

    /** Euclidean norm. */
    public static function l2Norm(array $a): float
    {
        self::checkSingle($a);
        return self::finite(sqrt(self::sumSqRaw(array_values($a))));
    }

    /** Euclidean distance. */
    public static function l2Distance(array $a, array $b): float
    {
        self::checkPair($a, $b);
        return self::finite(sqrt(self::l2SqRaw(array_values($a), array_values($b))));
    }

    /** Cosine similarity clamped to [-1, 1]; ZERO_MAGNITUDE when either norm is 0. */
    public static function cosineSimilarity(array $a, array $b): float
    {
        self::checkPair($a, $b);
        $a = array_values($a);
        $b = array_values($b);
        $d = self::finite(self::dotRaw($a, $b));
        $na = self::finite(sqrt(self::sumSqRaw($a)));
        $nb = self::finite(sqrt(self::sumSqRaw($b)));
        if ($na == 0.0 || $nb == 0.0) {
            throw new VectorError(VectorError::ZERO_MAGNITUDE, 'vector has magnitude zero');
        }
        return self::cosValue($d, $na, $nb);
    }

    /** Unit-length copy: a[i] * (1 / norm). @return list<float> */
    public static function normalize(array $a): array
    {
        self::checkSingle($a);
        $a = array_values($a);
        $n = self::finite(sqrt(self::sumSqRaw($a)));
        if ($n == 0.0) {
            throw new VectorError(VectorError::ZERO_MAGNITUDE, 'vector has magnitude zero');
        }
        $inv = self::finite(fdiv(1.0, $n));
        return array_map(static fn ($x): float => self::finite((float) $x * $inv), $a);
    }

    /** Element-wise sum. @return list<float> */
    public static function vecAdd(array $a, array $b): array
    {
        self::checkPair($a, $b);
        return array_map(static fn ($x, $y): float => self::finite((float) $x + (float) $y), array_values($a), array_values($b));
    }

    /** Element-wise difference. @return list<float> */
    public static function vecSub(array $a, array $b): array
    {
        self::checkPair($a, $b);
        return array_map(static fn ($x, $y): float => self::finite((float) $x - (float) $y), array_values($a), array_values($b));
    }

    /** Every element multiplied by $s. @return list<float> */
    public static function vecMulScalar(array $a, float $s): array
    {
        self::checkSingle($a);
        return array_map(static fn ($x): float => self::finite((float) $x * $s), array_values($a));
    }

    // --- batch and matrix --------------------------------------------------------

    /**
     * @param list<float> $scores
     * @return list<array{index: int, score: float}>
     */
    private static function rank(array $scores, bool $descending): array
    {
        $out = [];
        foreach ($scores as $i => $s) {
            $out[] = ['index' => $i, 'score' => $s];
        }
        // usort is stable since PHP 8.0, so equal scores keep index order
        usort($out, static fn ($x, $y): int => $descending ? $y['score'] <=> $x['score'] : $x['score'] <=> $y['score']);
        return $out;
    }

    private static function checkCandidate(array $query, array $c): void
    {
        if (count($c) !== count($query)) {
            throw new VectorError(VectorError::DIMENSION_MISMATCH, sprintf('expected %d elements, got %d', count($query), count($c)));
        }
    }

    /**
     * Cosine similarity of $query with each candidate, descending; a zero candidate scores 0.
     *
     * @param list<list<int|float>> $candidates
     * @return list<array{index: int, score: float}>
     */
    public static function batchCosine(array $query, array $candidates): array
    {
        self::checkSingle($query);
        $query = array_values($query);
        $qn = self::finite(sqrt(self::sumSqRaw($query)));
        if ($qn == 0.0) {
            throw new VectorError(VectorError::ZERO_MAGNITUDE, 'query has magnitude zero');
        }
        $scores = [];
        foreach (array_values($candidates) as $c) {
            self::checkCandidate($query, $c);
            $c = array_values($c);
            $d = self::finite(self::dotRaw($query, $c));
            $cn = self::finite(sqrt(self::sumSqRaw($c)));
            $scores[] = $cn == 0.0 ? 0.0 : self::cosValue($d, $qn, $cn);
        }
        return self::rank($scores, true);
    }

    /** Dot product of $query with each candidate, descending. @return list<array{index: int, score: float}> */
    public static function batchDot(array $query, array $candidates): array
    {
        self::checkSingle($query);
        $query = array_values($query);
        $scores = [];
        foreach (array_values($candidates) as $c) {
            self::checkCandidate($query, $c);
            $scores[] = self::finite(self::dotRaw($query, array_values($c)));
        }
        return self::rank($scores, true);
    }

    /** Euclidean distance from $query to each candidate, ascending. @return list<array{index: int, score: float}> */
    public static function batchL2(array $query, array $candidates): array
    {
        self::checkSingle($query);
        $query = array_values($query);
        $scores = [];
        foreach (array_values($candidates) as $c) {
            self::checkCandidate($query, $c);
            $scores[] = self::finite(sqrt(self::l2SqRaw($query, array_values($c))));
        }
        return self::rank($scores, false);
    }

    /**
     * out[i][j] = cosine similarity of a[i] and b[j]; 0 when either has magnitude zero.
     *
     * @param list<list<int|float>> $a
     * @param list<list<int|float>> $b
     * @return list<list<float>>
     */
    public static function distanceMatrixCosine(array $a, array $b): array
    {
        if (count($a) === 0 || count($b) === 0) {
            throw new VectorError(VectorError::EMPTY_VECTOR, 'vector set is empty');
        }
        $a = array_map('array_values', array_values($a));
        $b = array_map('array_values', array_values($b));
        self::checkSingle($a[0]);
        foreach (array_merge($a, $b) as $v) {
            self::checkCandidate($a[0], $v);
        }
        $na = array_map(static fn ($v): float => self::finite(sqrt(self::sumSqRaw($v))), $a);
        $nb = array_map(static fn ($v): float => self::finite(sqrt(self::sumSqRaw($v))), $b);
        $out = [];
        foreach ($a as $i => $va) {
            $row = [];
            foreach ($b as $j => $vb) {
                $d = self::finite(self::dotRaw($va, $vb));
                $row[] = ($na[$i] == 0.0 || $nb[$j] == 0.0) ? 0.0 : self::cosValue($d, $na[$i], $nb[$j]);
            }
            $out[] = $row;
        }
        return $out;
    }
}
