# Algebraic Data Analysis Library

Can you do data analysis with complex numbers? Surprisingly, it turns out, yes you can and Python supports complex numbers natively which makes it easier. You can even represent discrete probability distributions as polynomials, multiply them together and interpet the result as a convolution of the two distributions (after normalizing the coefficients). 

This repository corresponds to a paper on the discrete-time Fourier transform ([papers with code](https://paperswithcode.com/search?q=author%3Adayne+sorvisto)] and its use and applications to data analysis. You can use this special flavor of Fourier transform (no restriction on the data having to be periodic, it just needs to represent a discerte probability distribution) and its inverse to estimate the probability distribution of some finite det of data points and compute a discrete version of the characteristic function. 

What is a characteristic function? For a continuous probability distribution (for example a Normal distribution), this function of a complex variable has many interesting statistical and mathematical properties including completely defining the distribution without loss of information (it is used for example in the proof of the central limit theorem). 

I've included a few extra functions in this library for reversing the process through the inverse transform (discrete characteristic function to discrete probability distribution) and some functions for representing discrete probability distributions as polynomials (multiplication in this case can be interpreted as convolution and can be used for smoothing). The idea is, you might be able to find your own applications by representing the histogram of your data as some kind of polynomial, multiply polynomials with the characteristic function or with other polynomials to create a purely algebraic data analysis. 

Can you find an interesting use case to real-world data (preferably data that isn't periodic)? 

# Contributing

This code is experimental. 

Despite the wide-spread use of Fourier transforms in signal processing and time series (especially seasonal data or data that is periodic in nature), applying the discrete-time Fourier transform to discrete probability distributions to do data analysis is a less well-known technique (the continuous version is widely applied in mathematicla statistics).

If you use this code in a practical example of data analysis please create a PR and share the example (I'm particularly interested in seeing different ways to interpret the discrete characteristic function with real world data or any surprising interpretations or applications to data science).

## Usage

To compute the characteristic function of a custom data distribution represented as a NumPy nd-array, follow these steps:

1. Install the required dependencies, such as NumPy.

2. Set the random seed for reproducibility and define the standard deviation (`sigma`), mean (`mu`), and training data size (`N`).

3. Generate a training dataset `X_train` consisting of `N` data points using any appropriate method.

4. Estimate the probability mass function of the custom data using kernel density methods or other suitable techniques. Technically this can be replaced witb a learning algorithm for more complex tabular data sets.

5. Compute the characteristic function by calling the `compute_char` function on the training dataset `X_train`. The resulting characteristic function is stored in the variable `char`.

6. Estimate the probability of desired events using the `proba` function. Provide the event index, the characteristic function, and the total number of possible outcomes as inputs. The probabilities are stored in variables `p`, `q`, etc., depending on the number of outcomes.

7. Print or further analyze the estimated probabilities.

Here's an example usage:

```python
import math
import random
import numpy as np
from algebraic_data_analysis import char, compute_char, proba

# Set random seed for reproducibility and probability of Heads and Tails for a coin toss experiment.
# Note: You can substitute coin toss experiment for any random process with parameters of your choice.
sigma = 1
pr_heads = 0.4
pr_tails = 0.6

random.seed(123)

# Generate a training dataset of size N (custom data generation method)
X_train = np.array(...)  # Replace with your custom data generation code

# Estimate probability mass function using appropriate methods
# For example, kernel density estimation or any other technique

char = compute_char(X_train)

# Reverse the process with the inverse transform. For example, estimate probability of events using discrete characteristic function
# This proves the process worked as expected 
p, q = [proba(x, char, 2) for x in range(2)]

print(f"The probability of event 0 is {round(p, 2)}")
print(f"The probability of event 1 is {round(q, 2)}")

```

In this example, the random seed is set for reproducibility, and parameters of a random process are defined (in this case coin toss experiment), and training data size (N) are defined. A training dataset X_train is generated using an appropriate method for custom data generation. The probability mass function of the data is estimated using methods such as kernel density estimation. The compute_char function is then used to compute the characteristic function of the discrete distribution based on the training dataset. The proba function is employed to estimate the probabilities of specific events, by passing the event index, the computed characteristic function, and the total number of possible outcomes. The estimated probabilities are printed or further analyzed as needed.

## Philosophy of Algebraic Data Analysis

The Fourier transform provides many equivalences between different areas and has unique applications to data analysis. Currently, the starting point involves estimating a discrete probability distribution from the data and representing it using polynomials. For instance, multiplying polynomials corresponds to the convolution of probability distributions. Another algebraic structure used is the characteristic function, which utilizes the discrete-time Fourier transform to enable data analysis with complex numbers. By leveraging basic algebraic operations, such as convolutions, Fourier transforms, and smoothing operations, powerful and elegant data analysis and inference tasks can be performed on discrete probability distributions.


