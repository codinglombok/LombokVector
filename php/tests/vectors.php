<?php

declare(strict_types=1);

// Runs the binary64 cases of the shared vectors: every bit pattern must match.

require_once __DIR__ . '/bootstrap.php';

use CodingLombok\LombokVector\LombokVector as LV;
use CodingLombok\LombokVector\VectorError;

$doc = json_decode((string) file_get_contents(__DIR__ . '/../../vectors/lombokvector-vectors-v1.json'), true, 512, JSON_THROW_ON_ERROR);

function bits(float $x): string
{
    return '0x' . bin2hex(pack('E', $x));
}

function vectorRun(string $op, array $a): mixed
{
    $list = static fn (array $v): array => array_map('bits', $v);
    $ranked = static fn (array $v): array => array_map(static fn ($r) => [$r['index'], bits($r['score'])], $v);
    return match ($op) {
        'dot' => bits(LV::dotProduct($a[0], $a[1])),
        'norm' => bits(LV::l2Norm($a[0])),
        'l2' => bits(LV::l2Distance($a[0], $a[1])),
        'cosine' => bits(LV::cosineSimilarity($a[0], $a[1])),
        'normalize' => $list(LV::normalize($a[0])),
        'add' => $list(LV::vecAdd($a[0], $a[1])),
        'sub' => $list(LV::vecSub($a[0], $a[1])),
        'scale' => $list(LV::vecMulScalar($a[0], (float) $a[1])),
        'batch_cosine' => $ranked(LV::batchCosine($a[0], $a[1])),
        'batch_dot' => $ranked(LV::batchDot($a[0], $a[1])),
        'batch_l2' => $ranked(LV::batchL2($a[0], $a[1])),
        'matrix_cosine' => array_map($list, LV::distanceMatrixCosine($a[0], $a[1])),
    };
}

$cases = array_values(array_filter($doc['cases'], static fn ($c) => $c['precision'] === 'f64'));
check(count($cases) >= 100, 'at least 100 binary64 cases');
foreach ($cases as $c) {
    try {
        $got = ['value' => vectorRun($c['op'], $c['args'])];
    } catch (VectorError $e) {
        $got = ['error' => $e->errorCode];
    }
    check($got === $c['expected'], $c['id'] . ': got ' . json_encode($got) . ' want ' . json_encode($c['expected']));
}
finish('vectors');
