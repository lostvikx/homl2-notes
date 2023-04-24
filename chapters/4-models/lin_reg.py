# %%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# %%
sns.set_theme(context="notebook", palette="muted")
# %% [markdown]
# * y = h(theta), y = theta.T * x
# * theta_best = inv(X.T * X) * X.T * y, theta_best: is the value of theta that minimizes the cost function
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
eta = 0.1  # learning rate
n_iter = 1000
m = 100  # instances

theta = np.random.randn(2, 1)

for i in range(n_iter):
  gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
  theta = theta - eta * gradients
print(theta)
# %%
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

f.suptitle("Comparing Different Learning Rates")
f.tight_layout()
ax1.set_ylabel("y")

theta = np.random.randn(2, 1)
for ax in (ax1, ax2, ax3):
  ax.plot(X_new, X_new_b.dot(theta), "r--", label="Initial")

def plot_gd(eta, ax, theta=theta):
  n_iter = 100
  m = len(X)  # no. of instances or 100
  ax.set_title(f"eta = {eta}")

  for i in range(n_iter):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    if i < 10:
      ax.plot(X_new, X_new_b.dot(theta), "b-")

plot_gd(0.02, ax1)
plot_gd(0.1, ax2)
plot_gd(0.5, ax3)

for ax in (ax1, ax2, ax3):
  ax.plot(X, y, "b.")
  ax.axis([0, 2, -5, 15])
  ax.set_xlabel("X1")
  ax.legend()

plt.show()
# %%
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters
m = len(X)

def learning_schedule(t, t0=t0, t1=t1):
  return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
  for i in range(m):
    rand_idx = np.random.randint(m)
    xi = X_b[rand_idx:rand_idx+1]
    yi = y[rand_idx:rand_idx+1]
    # xi is only 1 instance
    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
    eta = learning_schedule(epoch * m + i)
    theta = theta - eta * gradients
print(theta)
# %%
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

f.suptitle("Stocastic Gradient Descent")
f.tight_layout()
ax1.set_ylabel("y")

theta = np.random.randn(2, 1)
for ax in (ax1, ax2):
  ax.plot(X_new, X_new_b.dot(theta), "r--", label="Initial")

def plot_sgd(get_eta, ax, title, theta=theta):
  n_epochs = 50
  t0, t1 = 5, 50  # learning schedule hyperparameters
  m = len(X)
  ax.set_title(title)

  for epoch in range(n_epochs):
    for i in range(m):
      rand_idx = np.random.randint(m)
      xi = X_b[rand_idx:rand_idx+1]
      yi = y[rand_idx:rand_idx+1]
      # xi is only 1 instance
      gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
      eta = get_eta(epoch * m + i)
      theta = theta - eta * gradients
    if (epoch < 20):
      ax.plot(X_new, X_new_b.dot(theta), "b-")

plot_sgd(learning_schedule, ax1, title="Learning Schedule")
plot_sgd(lambda x: 0.1, ax2, title="Constant Learning Rate")

for ax in (ax1, ax2):
  ax.plot(X, y, "b.")
  ax.axis([0, 2, -5, 15])
  ax.set_xlabel("X1")
  ax.legend()

plt.show()