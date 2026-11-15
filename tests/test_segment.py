import itertools
import random
import unittest
from fractions import Fraction as F

from digitlab.segment import segment_audit


class SegmentTests(unittest.TestCase):
    def verify(self, w, b, a, z):
        result = segment_audit(w, b, a, z)
        cuts = [F(t) for t in result['cuts']]
        self.assertEqual((cuts[0], cuts[-1]), (0, 1))
        self.assertEqual(cuts, sorted(set(cuts)))
        def scores(t):
            x = [(1-t)*F(v)+t*F(u) for v, u in zip(a, z)]
            return [F(b[k])+sum(F(row[k])*v for row, v in zip(w, x)) for k in range(len(b))]
        # Independently recover every pair root from endpoint score differences.
        roots = {F(0), F(1)}
        for i, j in itertools.combinations(range(len(b)), 2):
            v0 = scores(0)[i]-scores(0)[j]
            v1 = scores(1)[i]-scores(1)[j]
            if v0 != v1:
                root = v0/(v0-v1)
                if 0 <= root <= 1:
                    roots.add(root)
        self.assertEqual(cuts, sorted(roots))
        for point in result['points'] + [i['probe'] for i in result['intervals']]:
            s = scores(F(point['t']))
            self.assertEqual(point['scores'], [str(v) for v in s])
            self.assertEqual(point['prediction'], s.index(max(s)))
            self.assertEqual(point['tied_maxima'], [k for k, v in enumerate(s) if v == max(s)])
        for interval, lo, hi in zip(result['intervals'], cuts, cuts[1:]):
            self.assertEqual((F(interval['left']), F(interval['right'])), (lo, hi))
            for n in (1, 2, 3, 7):
                t = (lo*n+hi)/(n+1)
                s = scores(t)
                self.assertEqual(s.index(max(s)), interval['probe']['prediction'])
        return result

    def test_crossings_ties_singleton_winner_and_dominated_cuts(self):
        r = self.verify([[1, 0, -1]], [0, 0, 1], [0], [1])
        self.assertEqual(r['cuts'], ['0', '1/2', '1'])
        self.assertEqual(r['points'][1]['prediction'], 0)
        # Class zero wins only at the three-way tie; not on either open side.
        r = self.verify([[0, -1, 1]], [0, .5, -.5], [0], [1])
        self.assertEqual([i['probe']['prediction'] for i in r['intervals']], [1, 2])
        self.assertEqual(r['points'][1]['prediction'], 0)
        self.verify([[0, 1, -1]], [10, 0, 1], [0], [1])

    def test_random_models_reversal_and_row_scaling(self):
        rng = random.Random(913)
        for _ in range(80):
            d, c = rng.randrange(1, 5), rng.randrange(2, 6)
            w = [[rng.randrange(-8, 9)/4 for _ in range(c)] for _ in range(d)]
            b = [rng.randrange(-8, 9)/4 for _ in range(c)]
            a, z = [[F(rng.randrange(9), 8) for _ in range(d)] for _ in range(2)]
            r = self.verify(w, b, a, z)
            rev = self.verify(w, b, z, a)
            self.assertEqual([F(t) for t in rev['cuts']], sorted(1-F(t) for t in r['cuts']))
            scaled = self.verify([[v*2 for v in row] for row in w], [v*2 for v in b], a, z)
            self.assertEqual(r['cuts'], scaled['cuts'])
            self.assertEqual([p['prediction'] for p in r['points']], [p['prediction'] for p in scaled['points']])

    def test_constant_segment_subnormal_large_integer_and_endpoint_ties(self):
        self.verify([[1, 1]], [0, 0], ['1/3'], ['1/3'])
        self.verify([[float.fromhex('0x0.0000000000001p-1022'), 0]], [0, 0], [0], [1])
        self.verify([[2**100+1, 2**100]], [0, 1], [0], [1])
        self.verify([[1, 0]], [0, 0], [0], [1])
        self.verify([[0, 1]], [0, 0], [0], [1])

    def test_invalid_inputs_and_no_mutation(self):
        w, b, a, z = [[1, -1]], [0, 0], ['1/4'], ['3/4']
        before = repr((w, b, a, z));self.verify(w, b, a, z)
        self.assertEqual(repr((w, b, a, z)), before)
        for args in [([], b, a, z), (w, [0], a, z), (w, b, [], z),
                     (w, b, [True], z), (w, b, [.5], z), (w, b, ['1/0'], z),
                     (w, b, [-1], z), (w, b, [F(1, 2**257)], z),
                     ([[float('inf'), 0]], b, a, z), ([[True, 0]], b, a, z),
                     ([[2**2049, 0]], b, a, z)]:
            with self.assertRaises(ValueError):
                segment_audit(*args)


if __name__ == '__main__':
    unittest.main()
