import numpy
import sympy

from polytope import *
from toric import *


def symbols(prefix, count):
    return sympy.symbols(' '.join(prefix + str(i) for i in range(count)))


if __name__ == '__main__':
    # original variety
    vertices = [numpy.array(x) for x in [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
                                         (0, 0, 2), (2, 0, 2), (0, 2, 2),
                                         (2, 2, 1), (2, 1, 2), (1, 2, 2)]]
    normals = [numpy.array(x) for x in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0),
                                        (-1, -1, -1), (1, 0, 0), (-1, 0, 0)]]

    class_map = numpy.array([[1, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0],
                             [1, 0, 1, 0, 1, 1, 0],
                             [0, 0, 0, 0, 0, 1, 1]])

    poly = Polytope(vertices, normals)
    variety = CompleteToricVariety(poly, class_map=class_map)

    # restricted variety
    vertices_prime = [numpy.array(x) for x in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    normals_prime = [numpy.array(x) for x in [(0, 1), (0, -1), (1, 0), (-1, 0)]]

    class_map_prime = numpy.array([[1, 1, 0, 0],
                                   [0, 0, 1, 1]])

    poly_prime = Polytope(vertices_prime, normals_prime)
    variety_prime = CompleteToricVariety(poly_prime, class_map=class_map_prime)


    divisors = [
        variety.divisor_from_polytope(Polytope([
            numpy.array(x) for x in [(0, 0, 0), (2, 0, 0)]
        ], normals[:4] + normals[5:])),
        variety.divisor_from_polytope(Polytope([
            numpy.array(x) for x in [(0, -1, 0), (2, -1, 0), (2, 1, 0), (0, 1, 0)]
        ], normals[:4] + normals[5:])),
        variety.divisor_from_polytope(Polytope([
            numpy.array(x) for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        ], normals[:1] + normals[2:3] + normals[4:5] + normals[5:6])),
        variety.divisor_from_polytope(Polytope([
            numpy.array(x) for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        ], normals[:4] + normals[5:]))
    ]

    divisors_prime = [
        variety_prime.divisor_from_polytope(Polytope([
            numpy.array(x) for x in [(0, 0)]
        ], normals_prime)),
        variety_prime.divisor_from_polytope(Polytope([
            numpy.array(x) for x in [(-1, 0), (1, 0)]
        ], normals_prime)),
        variety_prime.divisor_from_polytope(Polytope([
            numpy.array(x) for x in [(0, 0), (1, 0), (0, 1)]
        ], normals_prime)),
        variety_prime.divisor_from_polytope(Polytope([
            numpy.array(x) for x in [(0, 0), (1, 0), (0, 1), (1, 1)]
        ], normals_prime))
    ]

    print('Divisor images:')
    for div, div_prime in zip(divisors, divisors_prime):
        print('{}  |-->  {}'.format(str(div), str(div_prime)))

    print()

    sum_of_divisors = sum(divisors[1:], ToricDivisor(variety, (0, 0, 0, 0, 0, 0, 0)))
    print(variety.class_map.dot(sum_of_divisors.prime_divisor_coeffs))

    print()

    a = symbols('a', len(divisors[0].section_basis))
    b = symbols('b', len(divisors[1].section_basis))
    c = symbols('c', len(divisors[2].section_basis))
    d = symbols('d', len(divisors[3].section_basis))
    sections = [d.section(*coeffs) for d, coeffs in zip(divisors, [a, b, c, d])]

    for i, s in enumerate(sections):
        print("Degree of F_{} = {}".format(str(i), str(variety.degree_of(s))))

    print()

    for i, s in enumerate(sections[1:]):
        print('Fbar_{} = {}'.format(str(i+1), str(s.as_poly(*variety_prime.indeterminates).as_expr())))

    print()

    lattice_points = [list(d.polytope.lattice_points) for d in divisors_prime[1:]]

    for i, l in enumerate(lattice_points):
        print('Lattice points in Pbar_{}: {}'.format(str(i+1), str(l)))

    print()

    partition = [[0, 1, 1], [0, 2, 1, 2], [0, 2, 1, 2]]

    matrix = variety_prime.partition_matrix(sections[1:], divisors_prime[1:], partition)

    print(matrix)

    print()

    elem_res_1 = sympy.poly(matrix.det(), *variety.indeterminates).as_expr()

    print(list(sympy.poly(matrix.det(), *variety_prime.indeterminates).as_dict().keys()))
    print()
    print(elem_res_1)
    print()
    print(variety.degree_of(elem_res_1))