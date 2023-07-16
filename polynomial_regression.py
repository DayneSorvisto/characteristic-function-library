import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(X, Y, degree):
    """
    Fits a polynomial regression model to the X-Y relationship.

    Parameters:
        X (list or numpy array): The input X values.
        Y (list or numpy array): The input Y values.
        degree (int): The degree of the polynomial.

    Returns:
        numpy array: The coefficients of the polynomial regression model.
    """
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    # Generate polynomial features
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly, Y)

    return model.coef_
