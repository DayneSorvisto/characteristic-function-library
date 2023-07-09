# Polynomial Bayes' Rule

This Python code demonstrates how to represent and compute Bayes' rule using polynomials. Bayes' rule provides a way to calculate conditional probabilities between events. We also include functions for representing discrete distributons as polynomials and working with probabilities using basic algebra. There's more advanced functions for computing characteritic functions of probability distributions using the discrete-time Fourier transform.

## Usage

# Computing Characteristic Function for custom data represented as NumPy nd-array . 

This code example demonstrates how to compute the characteristic function of a discrete distribution and estimate the probability of heads using a provided module for algebraic data analysis. For custom data represented as an array and use kernel density methods
to estimate the probability mass function. 

To compute the characteristic function and estimate the probability of heads, follow these steps:

1. Install the required dependencies, such as NumPy.

2. Set the random seed for reproducibility, and define the standard deviation (`sigma`), mean (`mu`), and training data size (`N`).

3. Generate a training dataset `X_train` consisting of N coin flip outcomes ('H' for heads, 'T' for tails) using random selection.

4. Compute the characteristic function by calling the `compute_char` function on the training dataset `X_train`. Store the resulting characteristic function in the variable `char`.

5. Estimate the probability of heads and tails using the `proba` function for two possible outcomes (0 for heads, 1 for tails). Store the probabilities in variables `p` and `q`, respectively.

6. Print the estimated probability of heads, rounded to two decimal places.

Here's an example usage:

```python
import math
import random
import numpy as np
from algebraic_data_analysis import char, compute_char, proba

# Set random seed for reproducibility, define standard deviation and mean.
sigma = 1
mu = 0
N = 1000

random.seed(123)

# Training data size = N
X_train = np.array([random.choice(['H', 'T']) for _ in np.arange(N)])

char = compute_char(X_train)

p, q = [proba(x, char, 2) for x in range(2)]

print(f"The probability of heads is {round(p, 2)}")

# Bayes rule applied to polynomial representations 

To compute Bayes' rule using polynomials, follow these steps:

1. Install the required dependencies, such as SymPy, using `pip` or any preferred package manager.

2. Define the polynomial representations for the probabilities involved: P(A), P(B), and P(B | A).

3. Use the `bayes_rule` function, providing the polynomial representations as inputs.

4. The function will compute the polynomial representation for P(A | B).

5. Print or further analyze the resulting polynomial, which represents the conditional probability P(A | B).

Here's an example usage:

```python
from sympy import symbols, Poly

def bayes_rule(p_a, p_b, p_b_given_a):
    x = symbols('x')
    p_a_given_b = Poly((p_b_given_a * p_a) / p_b, x)
    return p_a_given_b

# Example usage
x = symbols('x')
p_a = Poly(x**2 + 2*x + 1, x)
p_b = Poly(2*x + 3, x)
p_b_given_a = Poly(3*x + 4, x)

p_a_given_b = bayes_rule(p_a, p_b, p_b_given_a)
print("P(A | B):", p_a_given_b)

## Philosophy of Algebraic Data Analysis

There are various ways to represent discrete probability distributions as polynomials. Multiplying polynomials corresponds to convolution of probability disttributions. We can also represent probability distributions uniquely using Fourier transform. you can perform powerful data analysis using basic algebraic operations corresponding to convolutions. 

