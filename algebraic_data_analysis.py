"""
this module contains functions for algebraic data analysis.
"""
import math 
import numpy as np
from sympy import symbols, Eq, solve
from sympy import Poly

def normalize_polynomial(polynomial):
    """
    normalizes a polynomial so that the sum of the coefficients is 1.
    """
    polynomial = Poly(polynomial)
    coeffs = polynomial.all_coeffs()
    coeff_sum = sum(coeffs)
    normalized_coeffs = [coeff / coeff_sum for coeff in coeffs]
    normalized_polynomial = Poly(normalized_coeffs, polynomial.gen)
    return normalized_polynomial

def bayes_rule(p_a, p_b_given_a, p_b):
    """
    Computes the conditional probability of event A given event B using Bayes rule.
    """
    p_a_given_b = p_b_given_a * p_a / p_b
    p_a_given_b_normalized = normalize_polynomial(p_a_given_b)
    return p_a_given_b_normalized
    
def compute_algebraic_variety(polynomial, z):
    """
    computes the algebraic variety of a polynomial
    """
    equation = Eq(polynomial, 0)
    variety = solve(equation, z)
    return variety

def probability_to_polynomial(probabilities):
    """
    Represents an array of probabilities as a polynomial.
    This can be used to compute for example product of polynomials
    and other algebraic operations corresponding to convolution.
    """
    x = symbols('x')
    n = len(probabilities)
    terms = [p * x**i for i, p in enumerate(probabilities)]
    polynomial = sum(terms)
    return Poly(polynomial, x)

def add_distributions(polynomial1, polynomial2):
    """
    Adds two distributions represented as polynomials.
    """
    x = polynomial1.gen
    result = Poly(polynomial1.as_expr() + polynomial2.as_expr(), x)
    return result

def multiply_distributions(polynomial1, polynomial2):
    """
    Multiplies two distributions represented as polynomials.
    """
    x = polynomial1.gen
    result = Poly(polynomial1.as_expr() * polynomial2.as_expr(), x)
    return result


def char(t):
 """
 Estimate of characteristic function for normally distributed data set
 """
 val = np.exp((1j)*t*mu-(1/2)*sigma**2 * t**2)
 return val


def estimate_pmf(X_train):
 """
 Estimate the probability mass function for training data X
 """
 _, freq = np.unique(X_train, return_counts=True)
 estimate = freq / X_train.size
 return estimate


def fit(vector):
  """
  This returns the characteristic
  function of a pmf represented
  as a vector.
  """
  def char(w):
      """
      Compute characteristic function at poitn x given PMF
      """
      pmf = estimate_pmf(vector)
      l = len(pmf)
      weights = pmf[:l]
      phi = [np.exp((1j)*(math.pi)*w*k) for k in range(l)]
      return np.dot(phi, pmf[:l])
  return char

def reverse(x, char, N):
 """
 x (float): value of random variable rv in range 0, 1,2, ... N-1
 char (func): estimation of characteristic function  for random variable
 N (int): total number of discrete outcomes in probability mass function
 returns: re-constructed probability of rv at x
 """
 if x < N:
   probability = 1/N * np.sum(
      np.array(
      [char(2*n*math.pi/N) * np.exp((-1j)*2*math.pi*x*n/N) for n in np.arange(N)]
      )
   )
   # probability should be a real number between 0 and 1
   if probability.real > 1:
     raise ValueError("Probability estimate out of bounds.")
   return probability.real
 else:
   raise ValueError("Point x is not in support of random variable.")


