"""Descriptive numerical audit helpers; never a serving admission gate."""
import numpy as np


def numerical_metrics(actual, reference):
    a, r = np.asarray(actual), np.asarray(reference)
    if (a.shape != r.shape or not a.size or a.dtype.kind != 'f' or r.dtype.kind != 'f'
            or not np.isfinite(a).all() or not np.isfinite(r).all()):
        raise ValueError('matching finite nonempty floating arrays required')
    d = a.astype(np.float64) - r.astype(np.float64)
    with np.errstate(over='ignore'):
        a16, r16 = a.astype(np.float16), r.astype(np.float16)
    if not np.isfinite(a16).all() or not np.isfinite(r16).all():
        raise ValueError('values exceed float16 range')
    return {'shape': list(a.shape), 'actual_dtype': str(a.dtype), 'reference_dtype': str(r.dtype),
            'max_absolute_error': float(np.abs(d).max()), 'mae': float(np.abs(d).mean()),
            'rmse': float(np.sqrt(np.square(d).mean())),
            'float16_differing_values': int(np.count_nonzero(a16 != r16)),
            'float16_array_equal': bool(np.array_equal(a16, r16))}


def float16_cell_analysis(actual, reference):
    """Measure distance outside the reference rounding cell, without a tolerance.

    Cell endpoints are midpoints of adjacent finite binary16 values. Ties are
    reported separately by exact roundtrip comparison (round-to-nearest-even).
    Both signs of zero have the same numerical cell. No allclose is used.
    """
    metrics = numerical_metrics(actual, reference)
    r = np.asarray(reference)
    if r.dtype != np.float16:
        raise ValueError('original float16 reference required')
    a = np.asarray(actual).astype(np.float64)
    with np.errstate(over='ignore'):
        lower_neighbor = np.nextafter(r, np.float16(-np.inf)).astype(np.float64)
        upper_neighbor = np.nextafter(r, np.float16(np.inf)).astype(np.float64)
    if not np.isfinite(lower_neighbor).all() or not np.isfinite(upper_neighbor).all():
        raise ValueError('extreme float16 endpoint needs a separate overflow policy')
    ref64 = r.astype(np.float64)
    lower = (lower_neighbor + ref64) / 2
    upper = (upper_neighbor + ref64) / 2
    excess = np.maximum(np.maximum(lower-a, a-upper), 0)
    mismatch = np.asarray(actual).astype(np.float16) != r
    coords = np.argwhere(mismatch)
    return {**metrics, 'max_rounding_cell_excess': float(excess.max()),
            'mismatch_coordinates': coords.tolist(),
            'mismatch_values': [{'index': index.tolist(), 'actual': float(a[tuple(index)]),
                'reference': float(ref64[tuple(index)]), 'cell_lower': float(lower[tuple(index)]),
                'cell_upper': float(upper[tuple(index)]), 'cell_excess': float(excess[tuple(index)])}
                for index in coords],
            'mismatches_by_clip': (mismatch.reshape(mismatch.shape[0], -1).sum(1).tolist()
                                   if mismatch.ndim >= 3 else None),
            'mismatches_by_second': (mismatch.reshape(-1, 10, mismatch.shape[-1]).sum((0, 2)).tolist()
                                     if mismatch.ndim >= 2 and mismatch.shape[-2] == 10 else None)}
