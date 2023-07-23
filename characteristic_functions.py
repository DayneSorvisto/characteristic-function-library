"""
Top level module for working with characteristic functions including
computing characteritsic functions and estimating probability mass functions
"""
import math 
import random 
import numpy as np 
from algebraic_data_analysis import char, compute_char, proba

# set random seed for reproducibility. define standard deviation and mean.
mu = 0
N = 1000

random.seed(123)

# training data size = N
X_train = np.array([random.choice(['H', 'T']) for _ in np.arange(N)])

char = compute_char(X_train)

p , q = [proba(x, char, 2) for x in range(2) ]

print(f"The probability of heads is {round(p,2)}")

