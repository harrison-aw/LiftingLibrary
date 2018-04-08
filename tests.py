import unittest
import numpy

from polytope import *
from toric import *


class TestPolytope(unittest.TestCase):
    def test_printing(self):
        vertices = [numpy.array(v) for v in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
        normals = [numpy.array(n) for n in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        poly = Polytope(vertices, normals)

        self.assertEqual(
            repr(poly),
            ('Polytope([array([1, 1]), array([-1,  1]), array([-1, -1]), array([ 1, -1])], '
             '[array([1, 0]), array([0, 1]), array([-1,  0]), array([ 0, -1])])')
        )

        self.assertEqual(str(poly), 'conv{(1, 1), (-1, 1), (-1, -1), (1, -1)}')

    def test_containment(self):
        vertices = [numpy.array(v) for v in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
        normals = [numpy.array(n) for n in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        poly = Polytope(vertices, normals)

        zero = numpy.array([0, 0])
        other = numpy.array([100, -100])

        self.assertTrue(zero in poly)
        self.assertFalse(other in poly)

    def test_dimension(self):
        vertices = [numpy.array(v) for v in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
        normals = [numpy.array(n) for n in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        poly = Polytope(vertices, normals)

        self.assertEqual(poly.ambient_dim, 2)

    def test_lattice_points(self):
        vertices = [numpy.array(v) for v in [(1, 1), (0, 1), (0, 0), (1, 0)]]
        normals = [numpy.array(n) for n in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        poly = Polytope(vertices, normals)

        self.assertEqual(list(poly.lattice_points), [(0, 0), (0, 1), (1, 0), (1, 1)])


class TestCompleteToricVariety(unittest.TestCase):
    def test_printing(self):
        vertices = [numpy.array([0, 0]), numpy.array([1, 0]), numpy.array([0, 1])]
        normals = [numpy.array([1, 0]), numpy.array([0, 1]), numpy.array([-1, -1])]
        poly = Polytope(vertices, normals)
        tvar = CompleteToricVariety(poly)

        self.assertEqual(repr(tvar), 'CompleteToricVariety(Polytope({}, {}))'.format(repr(vertices), repr(normals)))


class TestToricDivisor(unittest.TestCase):
    def test_printing(self):
        vertices = [numpy.array([0, 0]), numpy.array([1, 0]), numpy.array([0, 1])]
        normals = [numpy.array([1, 0]), numpy.array([0, 1]), numpy.array([-1, -1])]
        poly = Polytope(vertices, normals)
        tvar = CompleteToricVariety(poly)

        div = ToricDivisor(tvar, (1, 1, 1))

        self.assertEqual(str(div), '(1, 1, 1)')


if __name__ == '__main__':
    unittest.main()
