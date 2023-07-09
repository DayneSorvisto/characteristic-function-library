from algebraic_data_analysis import compute_algebraic_variety, probability_to_polynomial, add_distributions, multiply_distributions
from algebraic_data_analysis import bayes_rule

from algebraic_data_analysis import proba, compute_char, char

# write fixtures and tests for the functions in algebraic_data_analysis.py
def test_compute_algebraic_variety():
    z = symbols('z', complex=True)
    polynomial = z**5 + 2*z - 1
    variety = compute_algebraic_variety(polynomial, z)
    assert variety == [-1/2 - sqrt(5)/2, -1/2 + sqrt(5)/2, 1]

def test_probability_to_polynomial():
    probabilities = [0.2, 0.3, 0.5]
    polynomial = probability_to_polynomial(probabilities)
    assert polynomial == Poly(0.2*x**0 + 0.3*x**1 + 0.5*x**2, x)

def test_add_distributions():
    x = symbols('x')
    polynomial1 = Poly(x**2 + 2*x + 1, x)
    polynomial2 = Poly(2*x + 3, x)
    sum_polynomial = add_distributions(polynomial1, polynomial2)
    assert sum_polynomial == Poly(3*x**2 + 4*x + 4, x)

def test_multiply_distributions():
    x = symbols('x')
    polynomial1 = Poly(x**2 + 2*x + 1, x)
    polynomial2 = Poly(2*x + 3, x)
    product_polynomial = multiply_distributions(polynomial1, polynomial2)
    assert product_polynomial == Poly(2*x**3 + 7*x**2 + 8*x + 3, x)

def test_proba():
    assert proba([1, 2, 3]) == [1/6, 1/3, 1/2]

def test_compute_char():
    assert compute_char([1, 2, 3]) == [1/6, 1/3, 1/2]

def test_char():
    assert char([1, 2, 3]) == [1/6, 1/3, 1/2]
