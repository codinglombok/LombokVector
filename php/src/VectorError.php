<?php

declare(strict_types=1);

namespace CodingLombok\LombokVector;

/**
 * Error thrown by every operation; `errorCode` is stable across ports
 * (SPEC section 4).
 */
final class VectorError extends \InvalidArgumentException
{
    public const DIMENSION_MISMATCH = 'DIMENSION_MISMATCH';
    public const EMPTY_VECTOR = 'EMPTY_VECTOR';
    public const ZERO_MAGNITUDE = 'ZERO_MAGNITUDE';
    public const NON_FINITE = 'NON_FINITE';

    public function __construct(public readonly string $errorCode, string $message)
    {
        parent::__construct($errorCode . ': ' . $message);
    }
}
