# Practical Data Analysis with Characteristic Functions

This library is an implementation of my paper using characteristic functions for practical data analysis. If you're not familiar with the [Characteristic function](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)) you can read about it before using the library.

Please suggest improvements, contribute examples or star the repository.

## Basis of the repository

In probability theory and statistics, the characteristic function of a random variable completely defines its probability distribution (they contain the same information) but they have many interesting properties that are useful in data analysis instead of working with probability distributions directly. Although they are used in many mathematical applications like the proof of the Central Limit Theorem they are not yet widely used in practical data analysis. Fortunately Python supports complex numbers natively which makes implementation straightforward. You can use this library's fit function to estimate characteristic functions given an array of data values for example ['H', 'T', 'H', 'T',' 'H'] representing a coin toss experiment (this is only an example, you can encode your data any way you want as long as it's a tensor).

The fit function is actually a higher order function (it returns another function, representing the characteristic function). It's called fit because it has to first fit a probability distribution to your data before applying a Fourier transform to compute the characteristic function. The exact calculation requires some data preprocessing to adapt Fourier transforms to probability distributions that is outlined in several technical papers. 

The library includes another function called reverse that is able to reverse the process (it takes a characteristc function as input) and uses the inverse Fourier transform to recover a probability distribution. 

If a random variable has a probability density function then the characteristic function is the Fourier transform of the probability density function. For a real world application based on sample data, we can estimate or learn the probability mass function then the discrete version of the characteristic function is the discrete-time Fourier transform of the probability mass function. A visualization of a characteristic function (the real component) is given below.

![image description](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Sinc_simple.svg/280px-Sinc_simple.svg.png)

Interestingly, characteristc functions can be generalized to tensors as well so library could be extended for data analysis with arbitrary tensor data. 

## Use in Data Analysis 

Characteristic functions can be used as part of procedures for fitting probability distributions to samples of data. Cases where this provides a practicable option compared to other possibilities include fitting the stable distribution since closed form expressions for the density are not available which makes implementation of maximum likelihood estimation difficult. Estimation procedures are available which match the theoretical characteristic function to the empirical characteristic function, calculated from the data. Paulson et al. (1975)[19] and Heathcote (1977)[20] provide some theoretical background for such an estimation procedure. In addition, Yu (2004)[21] describes applications of empirical characteristic functions to fit time series models where likelihood procedures are impractical. Empirical characteristic functions have also been used by Ansari et al. (2020)[22] and Li et al. (2020)[23] for training generative adversarial networks.

Since the characteristic function is the Fourier transform of the probability density function. Thus it provides an alternative route to analytical results compared with working directly with probability density functions or cumulative distribution functions. There are particularly simple results for the characteristic functions of distributions defined by the weighted sums of random variables. 

# Contributing

This code is experimental but is a library for a functional (lisp-like) syntax for data analysis using pure functions. 

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
from algebraic_data_analysis import fit, reverse

# Set random seed for reproducibility and probability of Heads and Tails for a coin toss experiment.
# Note: You can substitute coin toss experiment for any random process with parameters of your choice.
pr_heads = 0.4
pr_tails = 0.6

random.seed(123)

# Generate a training dataset of size N (custom data generation method). This can be any tensor.
X_train = np.array(...)  # Replace with your custom data generation code

# Estimate probability mass function using appropriate methods
# For example, kernel density estimation or any other technique

char = fit(X_train)

# This process is reversible, we can recover the original distribution. Reverse the process with the inverse transform.
# This proves the process worked as expected 
p, q = [reverse(x, char, 2) for x in range(2)]

print(f"The probability of event 0 is {round(p, 2)}")
print(f"The probability of event 1 is {round(q, 2)}")

```

In this example, the random seed is set for reproducibility, and parameters of a random process are defined (in this case coin toss experiment), and training data size (N) are defined. A training dataset X_train is generated using an appropriate method for custom data generation. The probability mass function of the data is estimated using methods such as kernel density estimation. The compute_char function is then used to compute the characteristic function of the discrete distribution based on the training dataset. The proba function is employed to estimate the probabilities of specific events, by passing the event index, the computed characteristic function, and the total number of possible outcomes. The estimated probabilities are printed or further analyzed as needed.

## Philosophy of Library

Library currently provides two methods fit and reverse and the idea is to have a kind of functional data analysis workflow using pure functions. Fit will take a tensor and estiamate a discrete version of the characteristic function while reverse takes a characteristic function and reverses process. You can use reverse to recover the original probability distribution.

The Fourier transform provides many equivalences between different areas and has unique applications to data analysis. By leveraging basic algebraic operations, such as convolutions, Fourier transforms, and smoothing operations, powerful and elegant data analysis and inference tasks can be performed on discrete probability distributions by using characteristic functions.


