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

* Set a large number of iteration, but interrupt when the gradient vector becomes tiny, this value is the tolerance (tol).

## Batch GD
* Gradient vector of the cost function is the partial derivative of the MSE function. `2/m * X^T (X.dot(theta) - y)`
* Batch GD uses the whole batch of training data at every step.

## Stochastic GD
Picks a random instance in the training set at every step and compute the gradient only on that single instance, making the algorithm much faster.

* The final parameter values are good, but not optimal.
* SGD has a better chance of finding the global minimum than BGD.
* The randomness helps to escape the local optima, but never settle on the minimum. So we gradually reduce the learning rate.
* The function that determines the learning rate at each iteration is called the learning schedule.

## Mini-batch GD
Instead of computing the gradients based on the full training set or based on a single instance, Mini-batch GD computes the gradients on a small random set of instances called mini-batches.

# Comparing Algorithms for Linear Regression
| Algorithm       | Large m | Large n | Hyperparams | Scalling | Scikit-learn     |
|-----------------|---------|---------|-------------|----------|------------------|
| Normal Equation | Fast    | Slow    | 0           | No       | N/A              |
| SVD             | Fast    | Slow    | 0           | No       | LinearRegression |
| Batch GD        | Slow    | Fast    | 2           | Yes      | SGDRegressor     |
| Stochastic GD   | Fast    | Fast    | >= 2        | Yes      | SGDRegressor     |
| Mini-batch GD   | Fast    | Fast    | >= 2        | Yes      | SGDRegressor     |

# Polynomial Regression
Use a linear model to fit nonlinear data by adding powers of each of each features as new features, then train the linear model on this extended set of features.

* Note: When there are multiple features Polynomial Regression is capable of finding relationships between features.
* PolynomialFeatures(degree=d) transforms an array with n features into an array containing (n + d)! / (d! * n!) features.

## Learning Curves
We use learning curves to understand if the model is overfitting or underfitting the training data.

* Plot training set and validation set errors at a given training set size.
* Typically, training error begins from 0 and increases and validation error decreases as m (# instances) increases.
* When both curves (training and validation error) have reached a plateau, are close and high, it means that the model is underfitting. Soluiton: Use a more complex model or create better features.
* When the error on the training data is much lower than validation error, and there is gap between the curves (model performs much better on the training set than validation set), it means that the model is overfitting. Solution: Use a less complex model or increase the number of instances (m).

### Bias-Variance Tradeoff
* Bias: Generalization error, might be due to wrong assumptions. High-bias model is more likely to underfit the training data.
* Variance: Model's excessive sensitivity to small variations in training data. Model with high degrees of freedom is likely to have high variance and thus overfit the training data.
* Increasing a model's complexity will typically increase the variance and reduce its bias. And reducing the model's complexity will increase its bias and reduce its variance.

# Regularized Linear Models
There are three main types:
1. Ridge Regression
2. Lasso Regression
3. Elastic Net

The goal of these regularized regression is to find a line (or a hyperplane in higher dimensions) that best fits the data by minimizing the sum of the squared differences between the predicted values and the actual values, while also adding a penalty/regularization term to the regression equation that prevents the model from overfitting to the training data.

## Ridge Regression
Ridge regression adds a penalty term that is proportional to the square of the magnitude of the coefficients. This penalty term shrinks the coefficients towards zero, but it does not set any of them exactly to zero. As a result, ridge regression is particularly useful when there are many variables in the data that are potentially related to the outcome variable and you want to identify the most important variables.

## Lasso Regression
Lasso regression, on the other hand, adds a penalty term that is proportional to the absolute value of the coefficients. This penalty term can set some of the coefficients to exactly zero, which makes Lasso regression particularly useful for feature selection when there are many variables in the data and you want to identify a smaller subset of the most important variables.

## Elastic Net
Elastic net is the combination of Ridge and Lasso regression. The regularization term is a simple mix of both terms, we decide on the l1 ratio.

So, which one should you pick: plain linear, ridge, lasso, or perhaps elastic net? 

It is almost always better to have some regularization, so avoid using plain linear regression. Ridge is a good default, but if you suspect there are only a few features that are useful, then use Lasso or Elastic Net, so that the weights of the useless features become zero. Also, Elastic Net is preferred over Lasso regression.

## Early Stopping
Stop training as soon as the validation error reaches a minimum. As we increase the epoch, after a while the validation error stops decreasing and starts to go up. This indicates that the model has started to overfit the training data.

# Regularization Guide
While Ridge regression can be effective at reducing the impact of multicollinearity in linear regression, it may not be the best choice in situations where the dataset has a large number of predictors, and only a small subset of them are truly important for the response variable. In such cases, Ridge regression may end up including many irrelevant predictors in the model, resulting in a higher variance and reduced prediction performance.

On the other hand, Lasso regression is known for its feature selection capabilities, as it tends to shrink the coefficients of irrelevant predictors to zero, effectively removing them from the model. However, Lasso regression may not be ideal in cases where there are groups of highly correlated predictors, as it tends to select only one predictor from each group, which may not capture the full information contained in the group.

Elastic Net regression combines the strengths of Ridge and Lasso regression by adding both L1 and L2 penalties to the loss function, which allows it to perform both regularization and feature selection. The L1 penalty promotes sparsity in the coefficient estimates and can effectively remove irrelevant predictors from the model, while the L2 penalty can mitigate the impact of multicollinearity and improve the stability of the coefficient estimates.

Therefore, while Ridge regression can be useful in reducing the impact of multicollinearity in linear regression, Elastic Net regression may be preferred over Ridge regression when dealing with datasets that have a large number of predictors, and only a small subset of them are truly important for the response variable. Elastic Net regression provides a more flexible and balanced approach to regularization and feature selection, and can result in a more accurate and stable model.

# Exercise
1. We should use a Gradient Descent algorithm to train a dataset with a million features.
2. If the features of our training set has different scales, then Gradient Descent algorithm will suffer as it requires that all the features are scaled. Scalling can be done by preproccessing the dataset using StandardScaler with scikit-learn.
3. Yes, gradient descent algorithm can get stuck in a local minimum when training a logistic regression model. We can fix this by increasing the number of iterations (max_iter) and setting a tolerance (tol).
4. No, some gradient descent algorithms might not converge (diverge) because of a high learning rate (eta0), while some may converge given a high number of iterations.
5. As the number of epoch increases (epoch: number of times the training set is passed through the model), the validation error starts to decrease. This happens upto a point, after which the validation error starts to increase consistantly. Meaning that the model is now overfitting to the training data, and to fix this the best solution is early stopping. We can stop training the model as the validation error increases beyond a threshold from the minimum validation error.
6. No, because the minimum found out by the cost function of a mini-batch gradient descent might be a local optima and not the global optima. Also, because we are only taking a random batch from the training set to figure out the minima, we must let it run for the entire training set. Therefore, we should not immediately stop the mini-batch gradient descent as the validation error goes up.
7. Mini-batch gradient descent will reach the global optima the quickest as it uses random instances (mini-batches) to compute the optimal weights. Batch gradient descent takes the longest time among the 3 gradient descent methods discussed, while stocastic gradient descent is erratic, jumps around the global optima a lot before converging. All methods can actually converge given the right learning rate (eta0) and learning schedule for mini-batch and stocastic.
8. When ploting the learning curves, if there is a large gap between the training error and validation error, it means that the model is overfitting. We can solve this problem by (1) decreasing the complexity of the model, change the degree param when using Polynomial (2) increasing the number of instances (3) can use a regularized regression model like Ridge.
9. The training and validation errors are fairly high and almost equal, this means that our Ridge model is underfitting the training data. Therefore, it is said that the model suffers from high bias and low variance. To fix this we must reduce the regularization hyperparameter alpha, so that the model reduces it bias.
10. (a) We may want to use Ridge regression instead of plain Linear regression to give us better control over the bias/variance trade-off. The regularization term helps allows the model to better fit the training data and generalize to new unseen data. (b) We may want to use Lasso regression instead of Ridge regression, if we suspect that there are only a few features that useful to predict the target variable. Lasso removes the useless features by setting their weights to zero. (c) We may want to use Elastic Net instead of Lasso regression if the dataset has a high degree of colinearity (many useful features). Elastic Net combines the strengths of both Ridge and Lasso, allowing it to do both regularization and feature selection. Lasso will be to harsh, it may remove even useful features during training.
11. To classify pictures as outdoor/indoor and daytime/nighttime, we can implement two Logistic regression classifiers to identify the two classes. We use two logistic regression classifiers instead of a softmax regression classifier because they are essentially two different classes and not a multi-class. A picture can be classified as indoor and daytime.