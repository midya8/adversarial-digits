"""Exact affine decisions along a supplied segment, not a floating execution proof."""
from fractions import Fraction
import math


def _rational(value):
    if type(value) not in (int, str, Fraction) or (type(value) is str and len(value) > 160):
        raise ValueError('endpoints require bounded integers, fractions or rational strings')
    try:
        x = Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise ValueError('invalid rational endpoint') from None
    if max(x.numerator.bit_length(), x.denominator.bit_length()) > 256 or not 0 <= x <= 1:
        raise ValueError('endpoint outside [0,1] or rational bit bound')
    return x


def segment_audit(weights, bias, start, end):
    """Partition t in [0,1] for x(t)=(1-t)*start+t*end.

    Up to 128 features and 20 classes. Python int/float coefficients are used
    as exact represented reals, not converted decimal approximations. Every
    pair crossing is retained, including crossings below the upper envelope.
    Open intervals and individual cut points separately honor first-index ties.
    """
    try:
        w, b = [list(row) for row in weights], list(bias)
        a, z = list(start), list(end)
    except TypeError:
        raise ValueError('expected coefficient and endpoint sequences') from None
    if not 1 <= len(w) <= 128 or not 2 <= len(b) <= 20 or len(a) != len(w) or len(z) != len(w) or any(len(row) != len(b) for row in w):
        raise ValueError('invalid segment dimensions')
    values = [v for row in w for v in row] + b
    if any(type(v) not in (int, float) or (type(v) is float and not math.isfinite(v)) for v in values):
        raise ValueError('coefficients must be finite Python integers or floats')
    if any(max(Fraction(v).numerator.bit_length(), Fraction(v).denominator.bit_length()) > 2048 for v in values):
        raise ValueError('coefficient bit bound exceeded')
    a, z = [_rational(v) for v in a], [_rational(v) for v in z]
    w, b = [[Fraction(v) for v in row] for row in w], [Fraction(v) for v in b]
    intercept = [b[k] + sum(row[k] * x for row, x in zip(w, a)) for k in range(len(b))]
    slope = [sum(row[k] * (y - x) for row, x, y in zip(w, a, z)) for k in range(len(b))]
    cuts = {Fraction(0), Fraction(1)}
    for i in range(len(b)):
        for j in range(i):
            d = slope[i] - slope[j]
            if d:
                t = (intercept[j] - intercept[i]) / d
                if 0 <= t <= 1:
                    cuts.add(t)
    cuts = sorted(cuts)

    def at(t):
        scores = [c + t * d for c, d in zip(intercept, slope)]
        winner = max(range(len(b)), key=lambda k: scores[k])
        return {'t': str(t), 'scores': [str(v) for v in scores], 'prediction': winner,
                'tied_maxima': [k for k, v in enumerate(scores) if v == scores[winner]]}

    return {'scope': 'exact real affine segment; not floating execution or nonlinear model',
            'cuts': [str(t) for t in cuts], 'points': [at(t) for t in cuts],
            'intervals': [{'left': str(lo), 'right': str(hi), 'probe': at((lo + hi) / 2)}
                          for lo, hi in zip(cuts, cuts[1:])],
            'intercepts': [str(v) for v in intercept], 'slopes': [str(v) for v in slope]}
