# Algebraic Data Analysis Library 

What would a data analysis library based on polynomials look like?

The following library contains experimental code, methods and techniques for representing data as polynomials. 

There are many ways to do this including polynomial regression, graph polynomials, characteristic functions and more. Polynomial operations also have interesting interpretations such as convolution which lends itself to many different types of analyses. 

## Pragmatic Use Cases of of this Library

Visualizing Relationships: By representing a probability distribution as a graph, you can visualize the relationships or dependencies between different events or variables. The edges between vertices can indicate the existence of potential dependencies or interactions.

Graph Analytics: Once the probability distribution is converted into a graph, you can apply various graph analytics techniques to gain insights into the distribution. For example, you can calculate centrality measures to identify the most influential events, detect communities or clusters of related events, or analyze the overall connectivity and structure of the distribution. You can check if 2 polynomials are equal (in coefficients) to check graph isomorphism.

Simulation and Sampling: The graph representation of a probability distribution can be utilized in simulation or sampling tasks. You can perform random walks or simulations on the graph to generate samples that follow the underlying probability distribution. This can be useful for generating synthetic data or performing Monte Carlo simulations.

Identifying Key Events: The graph structure can help identify key events or variables within the probability distribution. Nodes with high degrees or high centrality values may represent crucial events that have a significant impact on the overall distribution.


Statistical Analysis: Characteristic functions play a crucial role in statistics, specifically in probability theory and distribution analysis. The characteristic function of a probability distribution uniquely defines the distribution, allowing us to calculate moments, derive limit theorems, and estimate parameters. It is used in the proof of the Central Limit Theorem and this library has as discrete version that will work on real-world data. 

Spectral and Time Series Analysis: Characteristic functions and the DTFT find applications in time series analysis, where the goal is to model and forecast data that evolves over time. You can use DTFT to compute characteristic function of a random variable or probability distribution represented as an array of numbers or arbitrary tensor.

Data Compression and Anomaly Detection (filtering): The DTFT is utilized in data compression techniques such as Fourier-based compression algorithms. 

The following functioality is included with the library:

#  Graph polynomials

Features: 
- Computing discrete probability distributions to weighted graphs
- Computing graph polynomials from weighted graphs
- Computing adjacecy list reprensetation of graph into graph polynomials

# Computing Characteristic Functions (based on Discrete-Time Fourier transforms:

## Convolution of probability distributions using Polynomials

Features:

- Computing the characteristic function for a discrete probability distribution (returns function of a complex variable that can be evaluated at different points in domain for exaomple a complex polynomial)
- Reverse process to compute original probability distribution from a characteristic function 
- Functions for estimating probability distribution from some array of data
- Neural networks with Fourier Transform layers for generalizing to arbitrary tensor data

## Introduction 

Convolution and Fourier transforms are mathematically connected through the Convolution Theorem, which states that convolution in the time domain is equivalent to multiplication in the frequency domain. This connection is a fundamental property of Fourier transforms and provides a powerful tool for analyzing signals and systems.
It can also be applied to probability distributons not just signals. In fact the characteristic function of a probability distribution (implemented in this library as a discrete-time Fourier transform) is widely used in mathematical statistics (such as proof of the central limit theorem).  

What would a data analysis library based on convolution look like? This library is a proof of concept based around explicit convolution of data stored as tensors without need for Convolutional Neural Networks (eventually I will add code for Fourier transform layers and graph polynomials as part of this library and symbolic support for Einsums for general tensor operations).

## Technical Motivation and Intuition 

What happens when you represent a probability distribution as a polynomial? It turns out, there's several interesting ways to do this and polynomial multiplication has a natural interpetation as convolution. This gives rise to a kind of "functional" data analysis workflow that can be generalized to data represented as arbitrary tensors.

The abstraction that makes it all work is a special flavour of Fourier transform called the discrete-time Fourier transform (with some special treatment for probability) which plays well under convolution operation and can be applied to any kind of tensor data (array, matrix or more general tensor). 

There are other ways to represent data as polynomials such as graph polynomials (after encoding your probability distribution as a graph).

Through this Fourier transform, you can represent discrete probability distributions as polynomials, multiply them and interpet the result as a convolution or some other mathematical operation and perform symbolic data analysis.

# Contributing

This code is experimental. 

Despite the wide-spread use of Fourier transforms in signal processing and time series (especially seasonal data or data that is periodic in nature), applying the discrete-time Fourier transform to discrete probability distributions to do data analysis is a less well-known technique (the continuous version is widely applied in mathematicla statistics).

If you use this code in a practical example of data analysis please create a PR and share the example (I'm particularly interested in seeing different ways to interpret the discrete characteristic function with real world data or any surprising interpretations or applications to data science).

## Example Usage

To compute the characteristic function of a custom data distribution represented as a NumPy nd-array, follow these steps:

1. Install the required dependencies, such as NumPy.

2. Set the random seed for reproducibility and define the parameters of your random process or distribution.

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

## Example Working with Graph Polynomials 

```
# Example probability distribution
prob_distribution = [0.2, 0.5, 0.1, 0.3]

# Convert probability distribution to graph
adjacency_matrix = convert_prob_distribution_to_graph(prob_distribution)

# Compute graph polynomial
polynomial = graph_polynomial(adjacency_matrix)
print(polynomial)
```
## Philosophy of Algebraic Data Analysis

The Fourier transform provides many equivalences between different areas and has unique applications to data analysis. Currently, the starting point involves estimating a discrete probability distribution from the data and representing it using polynomials. For instance, multiplying polynomials corresponds to the convolution of probability distributions. Other algebraic structure used include graph polynomials and the characteristic function, which utilizes the discrete-time Fourier transform to enable data analysis with complex numbers. By leveraging basic algebraic operations, such as convolutions, Fourier transforms, and smoothing operations, powerful and elegant data analysis and inference tasks can be performed on discrete probability distributions.


