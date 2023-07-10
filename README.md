# Algebraic Data Analysis Library

This repository is a practical implementaton of ideas from an emerging field called algebraic statistics that can be applied to real world data.

- "Algebraic Statistics" by Peter Bühlmann and Sara van de Geer. [PDF]
- "Algebraic Statistics for Computational Biology" by Mathias Drton. [PDF]
- "Algebraic Methods for Statistical Models and Applications" by Sonja Petrović. [PDF]
- "Introduction to Algebraic Statistics" by Giovanni Pistone. [PDF]
- "Algebraic Methods in Statistics and Probability" by Bernd Sturmfels. [PDF]
- "Algebraic Statistics: A Computational Algebraic Geometry Approach" by Seth Sullivant. [PDF]

It aims to bring rigor and transparency to data analysis by applying algebraic concepts in a theorem, lemma, and corollary style, similar to mathematics. The provided experimental Python code allows symbolic manipulation of real data by mapping it to polynomials that can then be manipulated using algebraic operations like polynomial multiplication (interpreted as convolution of probability distrubutions) and other mathematical transformations like Fourier transforms to create new algebraic functions with specific properties. The library offers two key advantages:

1. Symbolic representation of data analysis: By representing data symbolically using algebraic objects, such as polynomials or functions of complex variables, data analysis becomes more expressive and interpretable.

2. Rigorous analysis with analytical methods: Once data is embedded in an algebraic object, analytical methods can be applied for more sophisticated and rigorous analysis.

## Representing Real Data as Polynomials

Many models are black boxes and although you could use for example, linear regression to represent data as a polynomial, it may not make sense to multiply 2 polynomial models together or otherwise apply any kind of algebraic operations to the model. Furthermore, you cannot compare or reuse different analyses because the model is emperical with its own unique preprocessing assumptions. 

There's no canonical way but several natural ways to represent data as polynomials and algebraic data structures that lead to interesting properties. In this way, real data can be manipulated symbolically by defining mathematical transformations that compute primitive representations. By representing data as polynomials or other advanced algebraic constructs (e.g., functions of complex variables), it becomes possible to apply algebraic methods and operations for analysis. The algebraic representation is "learned" from the data by estimating a discrete probability distribution. The choice of the learning phase workflow is flexible. Once the initial learning phase is complete, the transformations used are deterministic and strictly algebraic, existing within a vector space (with potential support for more general algebraic spaces like commutative rings in the future).

## Benefits of Algebra for Data Analysis

Using algebraic objects, such as polynomials, to represent real-world data brings several advantages, including a more elegant and rigorous approach to data analysis using mathematical notation. The ultimate goal is to represent all data analysis operations symbolically as algebraic operations within abstract algebraic structures like vector spaces or commutative rings, enabling the application of regular mathematical techniques.

## Basic Theory

Polynomial operations, like multiplication, naturally correspond to the convolution of probability distributions. Data analysis methods, such as Fourier transforms, can be applied to these distributions, yielding algebraic objects like characteristic functions, which possess useful properties. The code in this library provides basic functions for performing algebraic data analysis. It demonstrates how to represent and compute Bayes' rule using polynomials, estimate the characteristic function of a discrete distribution using the discrete-time Fourier transform, and work with discrete probability distributions using basic algebra. The foundations of this library are based on the paper [Papers with code](https://paperswithcode.com/paper/algebraic-data-analysis).

## Data Analysis Workflow

The typical workflow involves choosing a representation for the data, such as an array existing within an abstract vector space. The initial phase of the workflow is to learn a discrete probability distribution that represents the data, usually by estimating it from the data stored in an array. The array choice allows for additional tensor operations, which can be performed at any point during the analysis. Mathematical transformations between vector spaces are then applied to represent the data as algebraic objects, such as polynomials or functions of complex variables. The two primary representations currently supported are polynomials and characteristic functions (functions of a single complex variable). The computation of characteristic functions involves Fourier transforms, specifically the discrete-time Fourier transform, with special treatment for probability distributions.

*Note: A diagram illustrating the workflow and an example of symbolic analysis in LaTeX will be added.*

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

Algebraic data analysis offers various methods for representing and analyzing data. Currently, the starting point involves estimating a discrete probability distribution from the data and representing it using polynomials. For instance, multiplying polynomials corresponds to the convolution of probability distributions. Another algebraic structure used is the characteristic function, which utilizes the discrete-time Fourier transform to enable data analysis with complex numbers. By leveraging basic algebraic operations, such as convolutions, Fourier transforms, and smoothing operations, powerful and elegant data analysis and inference tasks can be performed on discrete probability distributions.

## Contributions

Contributions to the project are welcome via pull requests. The library is currently in its early stages and lacks ease of use. You can improve the accuracy or usability of the library or add new functions based on current papers in algebraic data analysis. All implementations should have practical use cases and be able to work with real-world data, accompanied by appropriate documentation.

Areas that require work (first-time contributions are welcome) include:

Providing real-world or interesting examples with actual datasets.
Adding links to other papers and work on algebraic data analysis in the README.
Implementing algebraic versions of Bayes' Rule (see probability distributions module).
Creating a PyPi package suitable for algebraic data analysis in production environments.
Incorporating einsums for representing tensor operations.
Exploring connections to topological data analysis.
Supporting algebraic geometry, commutative algebra, and combinatorics.
Feel free to contribute in any of these areas or suggest other improvements to enhance the library.
