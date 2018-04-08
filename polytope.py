import numpy
from collections import namedtuple
from itertools import product


# Half space refers to the set {x : <normal, x> >= min}
Halfspace = namedtuple('Halfspace', ['normal', 'min'])


class Polytope:
    """The convex hull of finitely many vectors."""

    def __init__(self, vertices, normals):
        """Initializes a polytope object.

        :param vertices: These are the vertices of polytope. It should be an iterable that contains objects which are
                         array-like with shape (k,) (as in numpy)
        :param normals: These should be the normals of the facets of the polytope. It should be an iterable that
                        contains objects which are array-like with shape (k,) and implement the "dot" method
        """
        self.vertices = vertices
        self.normals = normals

        halfspaces = []
        for n in normals:
            new_halfspace = Halfspace(n, min(n.dot(v) for v in vertices))
            halfspaces.append(new_halfspace)
        self.halfspaces = halfspaces

    def __repr__(self):
        return 'Polytope({}, {})'.format(repr(self.vertices), repr(self.normals))

    def __str__(self):
        return 'conv{{{}}}'.format(', '.join(map(lambda v: str(tuple(v)), self.vertices)))

    def __contains__(self, point):
        """Determine if a point is in the polytope.

        :param point: an array-like object of shape (k,)
        :return: True if the point is in the polytope, false otherwise.
        """
        for h in self.halfspaces:
            if h.normal.dot(point) < h.min:
                return False
        return True

    @property
    def ambient_dim(self):
        """Determines the dimension of the ambient space of the polytope."""

        return len(self.vertices[0])

    @property
    def lattice_points(self):
        """Enumerate the lattice points inside the polytope.

        :return: a generator that returns tuples
        """

        basis = numpy.identity(self.ambient_dim, int)

        mins = [min(b.dot(v) for v in self.vertices) for b in basis]
        maxes = [max(b.dot(v) for v in self.vertices) for b in basis]

        box = product(*[range(a, b+1) for a, b in zip(mins, maxes)])

        for p in box:
            if p in self:
                yield p

    def facet_distances_from_point(self, point):
        """Calculate distances between a point and each facet of the polytope.

        :param point: an array-like of shape (k,)
        :return: List of distances
        """

        return tuple(h.normal.dot(point) - h.min for h in self.halfspaces)