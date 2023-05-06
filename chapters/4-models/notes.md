# Notes: Training Models
Two different ways to train a linear model:
1. Closed-form: Normal Function or SVD
2. Iterative optimization: Gradient Descent (GD)

# Linear Regression
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

* Both the Normal Equation and the SVD approach get very slow as the number of features (n) grow.
* But, they can handle large training sets (m) efficiently (number of instances).

# Graditent Descent
It is a generic optimization algorithm capable of finding optimal solutions. It iteratively tweaks params to minimize the cost function.

* GD scales well with the number of features.

Size of step, or the learning rate is one hyperparameter of GD.
* If learning rate is too small, it will take many iterations to converge.
* If learning rate is too large, it can possibly make the algorithm diverge, with larger values.

> When using GD, one should ensure that all features have a similar scale.

* Set a large number of iteration, but interrupt when the gradient vector becomes tiny, this value is the tolerance.

## Batch GD
Gradient vector of the cost function is the partial derivative of the MSE function. `2/m * X^T (X.dot(theta) - y)`

* Batch GD uses the whole batch of training data at every step.

## Stochastic GD
Picks a random instance in the training set at every step and compute the gradient only on that single instance, making the algorithm much faster.

* The final parameter values are good, but not optimal.
* SGD has a better chance of finding the global minimum than BGD.
* The randomness helps to escape the local optima, but never settle om the minimum. So we gradually reduce the learning rate.
* The function that determines the learning rate at each iteration is called the learning schedule.

## Mini-batch GD
Instead of computing the gradients based on the full training set or based on a single instance, Mini-batch GD computes the gradients on a small random set of instances called mini-batches.

## Polynomial Regression
Use a linear model to fit nonlinear data by adding powers of each of each features as new features, then train the linear model on this extended set of features.

* Note: When there are multiple features Polynomial Regression is capable of finding relationships between features.
* PolynomialFeatures(degree=d) transforms an array with n features into an array containing (n + d)! / (d! * n!) features.

## Learning Curves
We use learning curves to understand if the model is overfitting or underfitting the training data.

## Regularized Linear Models
There are three main types:
1. Ridge Regression
2. Lasso Regression
3. Elastic Net

The goal of these regularized regression is to find a line (or a hyperplane in higher dimensions) that best fits the data by minimizing the sum of the squared differences between the predicted values and the actual values, while also adding a penalty/regularization term to the regression equation that prevents the model from overfitting to the training data.

### Ridge Regression
Ridge regression adds a penalty term that is proportional to the square of the magnitude of the coefficients. This penalty term shrinks the coefficients towards zero, but it does not set any of them exactly to zero. As a result, ridge regression is particularly useful when there are many variables in the data that are potentially related to the outcome variable and you want to identify the most important variables.

### Lasso Regression
Lasso regression, on the other hand, adds a penalty term that is proportional to the absolute value of the coefficients. This penalty term can set some of the coefficients to exactly zero, which makes Lasso regression particularly useful for feature selection when there are many variables in the data and you want to identify a smaller subset of the most important variables.

### Elastic Net
Elastic net is the combination of Ridge and Lasso regression. The regularization term is a simple mix of both terms, we decide on the l1 ratio.

So, which one should you pick: plain linear, ridge, lasso, or perhaps elastic net? 

It is almost always better to have some regularization, so avoid using plain linear regression. Ridge is a good default, but if you suspect there are only a few features that are useful, then use Lasso or Elastic Net, so that the weights of the useless features become zero. Also, Elastic Net is preferred over Lasso regression.

## Early Stopping
