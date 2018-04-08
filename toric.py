import numpy
import sympy
from functools import reduce
from operator import mul

class CompleteToricVariety:
    def __init__(self, polytope, **args):
        self.polytope = polytope

        if 'indeterminate_prefix' in args:
            self.indeterminate_prefix = args['indeterminate_prefix']
        else:
            self.indeterminate_prefix = 'X'

        if 'indeterminates' in args:
            self.indeterminates = args['indeterminates']
        else:
            num_facets = len(polytope.normals)
            symbols = ' '.join(self.indeterminate_prefix + str(i) for i in range(num_facets))
            self.indeterminates = sympy.symbols(symbols)

        if 'class_map' in args:
            self.class_map = args['class_map']
        else:
            self.class_map = None

    def __repr__(self):
        return 'CompleteToricVariety({})'.format(repr(self.polytope))

    def divisor_from_polytope(self, polytope):
        mins = [None] * len(self.polytope.normals)
        for m in polytope.lattice_points:
            for i, r in enumerate(self.polytope.normals):
                val = r.dot(m)
                if mins[i] is None or val < mins[i]:
                    mins[i] = val
        return ToricDivisor(self, tuple(-a for a in mins), polytope=polytope)

    def divisor_from_monomial(self, monomial):
        monomial = sympy.poly(monomial, *self.indeterminates)
        exponent_vector = list(monomial.as_dict().keys())[0]
        return ToricDivisor(self, exponent_vector)

    def degree_of(self, section):
        poly = sympy.poly(section, *self.indeterminates)
        classes = [self.class_map.dot(e) for e in poly.as_dict().keys()]

        candidate = classes[0]
        for cls in classes[1:]:
            if not (candidate == cls).all():
                raise ValueError('{} is not a homogeneous section'.format(repr(section)))
        return candidate

    def partition_matrix(self, sections, divisors, lattice_point_partitions):
        dim = len(sections)

        matrix = numpy.zeros((dim, dim), dtype=object)
        for i, s in enumerate(sections):
            coeffs = sympy.poly(s, *self.indeterminates).as_dict()
            lattice_points = list(divisors[i].polytope.lattice_points)
            for j in range(dim):
                f = divisors[i].polytope.facet_distances_from_point
                terms = [f(p) for p, label in zip(lattice_points, lattice_point_partitions[i]) if label == j]
                matrix[i][j] = sum(coeffs[t] * divisors[i].monomial_from_exponent_vector(t)
                                   for t in terms if t in coeffs)

        return sympy.Matrix(list(matrix))


class ToricDivisor:
    def __init__(self, variety, prime_divisor_coeffs, **args):
        self.variety = variety
        self.prime_divisor_coeffs = prime_divisor_coeffs

        if 'polytope' in args:
            self.polytope = args['polytope']
        else:
            self.polytope = None

    def __repr__(self):
        return 'ToricDivisor({}, {})'.format(repr(self.variety), repr(self.prime_divisor_coeffs))

    def __str__(self):
        return str(tuple(self.prime_divisor_coeffs))

    def __add__(self, other):
        return ToricDivisor(self.variety,
                            tuple(a + b for a, b in zip(self.prime_divisor_coeffs, other.prime_divisor_coeffs)))

    def __eq__(self, other):
        return (self.equiv_class == other.equiv_class).all()

    def monomial_from_lattice_point(self, lattice_point):
        distances = [n.dot(lattice_point) + a for n, a in zip(self.variety.polytope.normals, self.prime_divisor_coeffs)]
        return reduce(mul, (x**e for x, e in zip(self.variety.indeterminates, distances)))

    def monomial_from_exponent_vector(self, exponent_vector):
        return reduce(mul, (x**e for x, e in zip(self.variety.indeterminates, exponent_vector)))

    @property
    def section_basis(self):
        if self.polytope == None:
            raise ValueError('Divisor has no associated polytope.')

        return [self.monomial_from_lattice_point(m) for m in self.polytope.lattice_points]

    def section(self, *coeffs):
        return sum(c * b for c, b in zip(coeffs, self.section_basis))

    @property
    def equiv_class(self):
        try:
            return self.variety.class_map.dot(self.prime_divisor_coeffs)
        except AttributeError:
            raise ValueError('Associated variety has no class map defined.')

