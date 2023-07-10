# Algebraic Data Analysis Library

This repository corresponds to a  paper on the discrete-time Fourier transform and its use in data analysis. You can use this special flavor of Fourier transform to estimate the probability distribution of some finite det of data points and then compute a discrete version of the characteristic function. For a continuous distribution, this function has many interesting statistical and mathematical properties (it is used for example in the proof of the central limit theorem). 

I've included a few extra functions for reversing the process (discrete characteristic function to discrete probability distribution) and some functions for representing discrete probability distributions as polynomials (multiplication corresponds to convolution and can be used for smoothing). 

# Contributing

If you use this code in a practical example of data analysis please create a PR and share the example.

## Usage

To compute the characteristic function of a custom data distribution represented as a NumPy nd-array, follow these steps:

1. Install the required dependencies, such as NumPy.

2. Set the random seed for reproducibility and define the standard deviation (`sigma`), mean (`mu`), and training data size (`N`).

3. Generate a training dataset `X_train` consisting of `N` data points using any appropriate method.

4. Estimate the probability mass function of the custom data using kernel density methods or other suitable techniques.

5. Compute the characteristic function by calling the `compute_char` function on the training dataset `X_train`. The resulting characteristic function is stored in the variable `char`.

6. Estimate the probability of desired events using the `proba` function. Provide the event index, the characteristic function, and the total number of possible outcomes as inputs. The probabilities are stored in variables `p`, `q`, etc., depending on the number of outcomes.

7. Print or further analyze the estimated probabilities.

Here's an example usage:

```python
import math
import random
import numpy as np
from algebraic_data_analysis import char, compute_char, proba

# Set random seed for reproducibility and define standard deviation and mean.
sigma = 1
mu = 0
N = 1000

random.seed(123)

# Generate a training dataset of size N (custom data generation method)
X_train = np.array(...)  # Replace with your custom data generation code

# Estimate probability mass function using appropriate methods
# For example, kernel density estimation or any other technique

char = compute_char(X_train)

# Estimate probability of events using characteristic function
p, q = [proba(x, char, 2) for x in range(2)]

print(f"The probability of event 0 is {round(p, 2)}")
print(f"The probability of event 1 is {round(q, 2)}")
```

In this example, the random seed is set for reproducibility, and the standard deviation (sigma), mean (mu), and training data size (N) are defined. A training dataset X_train is generated using an appropriate method for custom data generation. The probability mass function of the data is estimated using methods such as kernel density estimation. The compute_char function is then used to compute the characteristic function of the discrete distribution based on the training dataset. The proba function is employed to estimate the probabilities of specific events, by passing the event index, the computed characteristic function, and the total number of possible outcomes. The estimated probabilities are printed or further analyzed as needed.

## Philosophy of Algebraic Data Analysis

Algebraic data analysis offers various methods for representing and analyzing data using algebra. Currently, the starting point involves estimating a discrete probability distribution from the data and representing it using polynomials. For instance, multiplying polynomials corresponds to the convolution of probability distributions. Another algebraic structure used is the characteristic function, which utilizes the discrete-time Fourier transform to enable data analysis with complex numbers. By leveraging basic algebraic operations, such as convolutions, Fourier transforms, and smoothing operations, powerful and elegant data analysis and inference tasks can be performed on discrete probability distributions.


