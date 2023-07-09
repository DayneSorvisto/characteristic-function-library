# Algebraic Data Analysis Library

This repository provides experimental Python code for symbolically working with real data using polynomials and other algebraic constructs. it provides mathematical transformations for computing these primitive representations such as Fourier transforms. 

Once you represent data in this way you can analyze it using algebraic methods and operations.

## Why use algebra for data analysis?

Representing real-world data on a computer using algebraic objects like polynomials has some interesting applications including a more elegant and rigorous way to analyze data. 

## Basic Theory

Polynomial operations such as multiplication naturally correspond to convolution of probability distributions and we can apply data analysis methods like Fourier transforms on these distributions to create algebraic objects like characteristic functions (these are alreqdy widely used in statistics for example in the proof of the central limit theorm) which have many useful properties. 

The code provides basic functions for performing algebraic data analysis. It demonstrates how to represent and compute Bayes' rule using polynomials, as well as how to estimate the characteristic function of a discrete distribution using the discrete-time Fourier transform. Additionally, it includes functions for representing discrete distributions as polynomials, working with discrete probability distributions using basic algebra.

Foundations are based on paper [Papers with code](https://paperswithcode.com/paper/algebraic-data-analysis)

## Data Analysis Workflow

A typical workflow would be to first choose a representation of your data for example an array.

You can use the library to estimate a discrete probaility distribution also as an array.

This array can be fed into mathematical transformations to represent the data as algebraic objects. The two main representations currently are polynomials and characteristic functions.

## Usage

To use the code for computing the characteristic function of a custom data distribution represented as a NumPy nd-array, follow these steps:

1. Install the required dependencies, such as NumPy.

2. Set the random seed for reproducibility and define the standard deviation (`sigma`), mean (`mu`), and training data size (`N`).

3. Generate a training dataset `X_train` consisting of N data points. You can use any method to generate your custom data.

4. Use kernel density methods or any other appropriate techniques to estimate the probability mass function of the custom data.

5. Compute the characteristic function by calling the `compute_char` function on the training dataset `X_train`. The resulting characteristic function is stored in the variable `char`.

6. Estimate the probability of events using the `proba` function for the desired outcomes. Pass the event index, the characteristic function, and the total number of possible outcomes as inputs. The probabilities are stored in variables `p`, `q`, and so on, depending on the number of outcomes.

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

## Explanation 

In this example, we first set the random seed for reproducibility and define the standard deviation (sigma), mean (mu), and training data size (N).

Next, we generate a training dataset X_train using any method suitable for your custom data (for example Kernel Density Estimation or a neural network). This could include data generation using random processes, real-world data, or any other appropriate technique.

After generating the training dataset, you can estimate the probability mass function of the data using kernel density estimation or any other appropriate methods.

Then, the compute_char function is used to compute the characteristic function of the discrete distribution based on the training dataset.

The proba function is used to estimate the probabilities of specific events. You can pass the event index, the computed characteristic function, and the total number of possible outcomes to the proba function.

The estimated probabilities are printed or further analyzed as needed.

## Philosophy of Algebraic Data Analysis

Algebraic data analysis provides various methods for representing and analyzing discrete probability distributions using polynomials. For example multiplying polynomials corresponds to convolution of probability distributions. The characteristic function is another more complex algebraic structure using the discrete-time Fourier transform that allows data analysis using complex numbers. By leveraging basic algebraic operations corresponding to convolutions, fourier transforms and smoothing operations you can perform powerful, elegant data analysis and inference tasks on discrete probability distributions.

## Contributions

Contribute to the project by creaitng a PR. The library is currently very basic and not easy to use. 

Improve the accuracy or ease of use of the library or add new functions based on current papers in algebraic data analysis. All implementations should have practical use and be able to work on real world data with appropriate documentation. 

Area that need work (first time contributions are welcome):

- Real world or interesting examples with actual data sets
- Links to other papers and work on algebraic data analysis (please add these to README)
- Algebraic implemention of Bayes Rule (see probaility distributions module)
- Creating PyPi package that can be used for algebraic data analysis in production environment
- Einsums for representing tensor operations 
- Connections to topological data analysis
- Support for algebraic geometry, commutative algebra, and combinatorics


