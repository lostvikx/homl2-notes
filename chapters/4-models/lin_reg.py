# %%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# %%
sns.set_theme(context="notebook", palette="muted")
# %% [markdown]
# * y = h(theta), y = theta^T * x
# * theta_best = (X^T * X)^-1 * X^T * y, theta_best: is the value of theta that minimizes the cost function
# %%
X = 2 * np.random.rand(100, 1)  # domain: [0, 2)
y = 4 + (3 * X) + np.random.randn(100, 1)  # y = 4 + 3x1
# %%
plt.plot(X, y, "b.")
plt.xlabel("X1")
plt.ylabel("y")
plt.axis([0, 2, 0, 15])
plt.show()
# %%
X_b = np.c_[np.ones((100, 1)), X]  # x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
# %% [markdown]
# The theta_best results are pretty close to the original params of the linear func.
# %%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # x0 = 1 to each instance
y_pred = X_new_b.dot(theta_best)
y_pred
# %%
plt.plot(X_new, y_pred, "r-", label="Predictions")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.legend()
plt.xlabel("X1")
plt.ylabel("y")
plt.show()
# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)  # bias term: intercept_ & weights: coef_
# %%
lin_reg.predict(X_new)
# %% [markdown]
# Another way to compute the theta_best is X^+ * y (X^+: pseudoinverse)
# %%
np.linalg.lstsq(X_b, y, rcond=1e-6)[0]
# %%
np.linalg.pinv(X_b).dot(y)
# %%
