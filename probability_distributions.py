"""
Top level module for working with probability distributions including bayes rule
using polynomial representations of discrete probability distributions. 
"""

from sympy import symbols, Eq, solve
from sympy import Poly
from algebraic_data_analysis import compute_algebraic_variety, probability_to_polynomial, add_distributions, multiply_distributions
from algebraic_data_analysis import bayes_rule

if __name__ == "__main__":
    # Example usage
    x = symbols('x')
    polynomial1 = Poly(x**2 + 2*x + 1, x)
    polynomial2 = Poly(2*x + 3, x)

    # Add two distributions
    sum_polynomial = add_distributions(polynomial1, polynomial2)
    print("Sum:", sum_polynomial)

    # Multiply two distributions
    product_polynomial = multiply_distributions(polynomial1, polynomial2)
    print("Product:", product_polynomial)


    # Example usage
    z = symbols('z', complex=True)
    polynomial = z**5 + 2*z - 1
    variety = compute_algebraic_variety(polynomial, z)
    print("Algebraic Variety:", variety)

    # algebraic version of bayes rule 
    x = symbols('x')
    p_a = 0.7  # Probability of event A (float)
    p_b_given_a = Poly(3*x + 4, x)  # Polynomial representing P(B | A)
    # get coefficients as list of p_b_given_a
    p_b = 0.5  # Probability of event B (float)
    p_a_given_b = bayes_rule(p_a, p_b_given_a, p_b)
    print("P(A | B):", p_a_given_b)
