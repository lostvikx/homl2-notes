# Notes: Training Models
Two different ways to train a linear model:
1. Closed-form: Normal Function or SVD
2. Iterative optimization: Gradient Descent (GD)

## Linear Regression
`y = theta_0 + (theta_1 * x_1) + (theta_2 * x_2) + ... + (theta_n * x_n)`
* y: predicted value
* n: number of features
* x_i: ith feature
* theta_j: jth model parameter (theta_0: bias term and rest are feature weights)

We can represent them as column vectors: `y = theta.T * x`.

## Normal Equation
`theta_best = inv(X.T * X) * X.T * y`, theta_best: is the value of theta that minimizes the cost function

## Singular Value Decomposition
`theta_best = X^+ * y`, X^+ is the pseudoinverse of X (Moore-Penrose inverse)

* Both the Normal Equation and the SVD approach get very slow as the number of features grow.
* But, they can handle large training sets efficiently (number of instances).

