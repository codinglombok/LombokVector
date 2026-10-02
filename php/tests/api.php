<?php

declare(strict_types=1);

require_once __DIR__ . '/bootstrap.php';

use CodingLombok\LombokVector\LombokVector as LV;
use CodingLombok\LombokVector\VectorError;

check(LV::dotProduct([1, 2, 3], [4, 5, 6]) === 32.0, 'dot of ints gives a float');
check(LV::innerProduct([1, 2], [3, 4]) === 11.0, 'inner');
check(LV::l2Norm([3, 4]) === 5.0, 'norm');
check(LV::l2Distance([1, 2, 3], [4, 5, 6]) === sqrt(27), 'l2');
check(LV::cosineSimilarity([1, 0], [0, 1]) === 0.0, 'cosine');
check(LV::normalize([3, 4]) === [3 * 0.2, 4 * 0.2], 'normalize uses a[i] * (1 / norm)');
check(LV::vecAdd([1, 2], [3, 4]) === [4.0, 6.0], 'add');
check(LV::vecSub([1, 2], [3, 4]) === [-2.0, -2.0], 'sub');
check(LV::vecMulScalar([1, 2], 3) === [3.0, 6.0], 'scale');
$negZero = LV::vecMulScalar([-2], 0.0)[0];
check($negZero === 0.0 && fdiv(1.0, $negZero) === -INF, '-2 * 0.0 is -0.0');
check(LV::dotProduct([1e16, 1, -1e16, 1], [1, 1, 1, 1]) === 0.0, 'lane order');
check(LV::dotProduct(['a' => 1, 'b' => 2], [3, 4]) === 11.0, 'keys are ignored');

throwsCode(static fn () => LV::dotProduct([], []), VectorError::EMPTY_VECTOR, 'empty');
throwsCode(static fn () => LV::dotProduct([1], [1, 2]), VectorError::DIMENSION_MISMATCH, 'mismatch');
throwsCode(static fn () => LV::cosineSimilarity([0, 0], [1, 1]), VectorError::ZERO_MAGNITUDE, 'zero');
throwsCode(static fn () => LV::cosineSimilarity([1e200, 1e200], [1, 1]), VectorError::NON_FINITE, 'overflow');
throwsCode(static fn () => LV::normalize([0]), VectorError::ZERO_MAGNITUDE, 'normalize zero');
throwsCode(static fn () => LV::l2Norm([NAN]), VectorError::NON_FINITE, 'nan');
throwsCode(static fn () => LV::vecAdd([INF], [1]), VectorError::NON_FINITE, 'inf');
throwsCode(static fn () => LV::vecMulScalar([], 1), VectorError::EMPTY_VECTOR, 'scale empty');
throwsCode(static fn () => LV::batchCosine([0, 0], []), VectorError::ZERO_MAGNITUDE, 'batch zero query');
throwsCode(static fn () => LV::batchDot([1], [[1, 2]]), VectorError::DIMENSION_MISMATCH, 'batch mismatch');
throwsCode(static fn () => LV::batchL2([], []), VectorError::EMPTY_VECTOR, 'batch empty');
throwsCode(static fn () => LV::distanceMatrixCosine([], [[1]]), VectorError::EMPTY_VECTOR, 'matrix empty');
throwsCode(static fn () => LV::distanceMatrixCosine([[1, 0]], [[1]]), VectorError::DIMENSION_MISMATCH, 'matrix mismatch');
$e = new VectorError(VectorError::NON_FINITE, 'x');
check($e->getMessage() === 'NON_FINITE: x' && $e instanceof \InvalidArgumentException, 'error shape');

$r = LV::batchDot([1, 1], [[1, 0], [0, 1], [2, -1], [-1, 2]]);
check(array_column($r, 'index') === [0, 1, 2, 3], 'ties keep index order');
check(LV::batchL2([0, 0], [[3, 4], [1, 0], [0, 1]]) === [['index' => 1, 'score' => 1.0], ['index' => 2, 'score' => 1.0], ['index' => 0, 'score' => 5.0]], 'batch l2');
check(array_column(LV::batchCosine([1, 1], [[0, 0], [1, 1]]), 'index') === [1, 0], 'zero candidate last');
check(LV::distanceMatrixCosine([[1, 0], [0, 0]], [[1, 0]]) === [[1.0], [0.0]], 'matrix zero row');

finish('api');
