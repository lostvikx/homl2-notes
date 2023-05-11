# %%
import numpy as np
from matplotlib import pyplot as plt
# %%
theme_name = "fast"
try:
  plt.style.use(theme_name)
except:
  print(f"Cannot use Pyplot theme: {theme_name}!")
# %% [markdown]
# * y = h(theta), y = theta.T * x
# * theta_best = inv(X.T * X) * X.T * y, theta_best: is the value of theta that minimizes the cost function
# %%
X = 2 * np.random.rand(100, 1)  # domain: [0, 2)
y = 4 + (3 * X) + np.random.randn(100, 1)  # y = 4 + 3x1
# %%
plt.scatter(X, y, s=10)
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
plt.scatter(X, y, s=10)
plt.plot(X_new, y_pred, "r-", label="Predictions")
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

f.suptitle("GD Different Learning Rates")
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
  ax.scatter(X, y, s=10)
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

f.suptitle("Stocastic GD")
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
  ax.scatter(X, y, s=10)
  ax.axis([0, 2, -5, 15])
  ax.set_xlabel("X1")
  ax.legend()

plt.show()
# %%
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
# %%
print(sgd_reg.intercept_, sgd_reg.coef_)
# %%
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
# %%
plt.scatter(X, y, s=10)
plt.xlabel("X1")
plt.ylabel("y")
plt.axis([-3, 3, 0, 10])
plt.show()
# %%
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print(X[0], X_poly[0])
# %%
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print(lin_reg.intercept_, lin_reg.coef_)
# %%
y_poly_pred = lin_reg.predict(X_poly)

# Remember to sort values to plot a smooth curve.
sorted_zip = sorted(zip(X, y_poly_pred), key=lambda x: x[0])
X_new, y_poly_pred = zip(*sorted_zip)

plt.title("Polynomial Regression")
plt.scatter(X, y, s=10)
plt.plot(X_new, y_poly_pred, "r-", label="Predictions")
plt.xlabel("X1")
plt.ylabel("y")
plt.axis([-3, 3, 0, 10])
plt.legend()
plt.show()
# %%
def plot_poly_pred(degree, line_style="r-", X=X, y=y):
  poly_features = PolynomialFeatures(degree=degree, include_bias=False)
  X_poly = poly_features.fit_transform(X)
  lin_reg = LinearRegression()
  lin_reg.fit(X_poly, y)
  y_poly_pred = lin_reg.predict(X_poly)
  sorted_zip = sorted(zip(X, y_poly_pred), key=lambda x: x[0])
  X_new, y_poly_pred = zip(*sorted_zip)
  plt.plot(X_new, y_poly_pred, line_style, label=f"{degree}")

plt.title("High Degree Polynomial Regression")
plt.scatter(X, y, s=10)
plot_poly_pred(degree=1, line_style="C1-")
plot_poly_pred(degree=10, line_style="C2-")
plot_poly_pred(degree=30, line_style="C3-")
plt.xlabel("X1")
plt.ylabel("y")
plt.axis([-3, 3, 0, 10])
plt.legend()
plt.show()
# %%
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = (1.6 * X**3) + (0.8 * X**2) + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y):
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  train_errors, val_errors = [], []

  for m in range(1, len(X_train)):
    model.fit(X_train[:m], y_train[:m])
    y_train_pred = model.predict(X_train[:m])
    y_val_pred = model.predict(X_val)
    train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))

  plt.plot(np.sqrt(train_errors), "r-", label="Train")
  plt.plot(np.sqrt(val_errors), "b-", label="Validation")
  plt.title("Learning Curves")
  plt.xlabel("Training Set Size")
  plt.ylabel("RMSE")
  plt.legend()
# %%
plot_learning_curves(LinearRegression(), X, y)
plt.axis([0, 80, 0, 20])
plt.show()
# %% [markdown]
# ## Explaination
# First, there are only a few training data (instances), hences the RMSE is 0. But, as new instances are added, it becomes difficult for the model to fit the data perfectly, and the error goes up until it reaches a plateau. As for the validation set, because in the beginning only a few instances were used to train the model, it was bad at generalizing, hence the high error. But, as more training instances were provided to train the model, the validation error slowly goes down to a plateau.
# * Both curves have reach plateau, and are fairly high.
# * This is a clear case of model underfitting the training data.
# %%
from sklearn.pipeline import Pipeline

poly_reg = Pipeline([
  ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
  ("lin_reg", LinearRegression())
])

plot_learning_curves(poly_reg, X, y)
plt.axis([0, 80, 0, 3])
plt.show()
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

m = 40
X = 6 * np.random.rand(m, 1) - 3
noise = np.random.randn(m, 1)
y = (0.5 * X) + 5 + noise

def plot_ridge(X, y, ax, degree, alpha, line_style):
  """
  Plot the ridge regularization line of fit
  """
  model = Pipeline([
    ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
    ("std_scaler", StandardScaler()),
    ("ridge", Ridge(alpha=alpha, solver="cholesky", random_state=42))
  ])

  model.fit(X, y)
  y_pred = model.predict(X)
  
  sorted_zip = sorted(zip(X, y_pred), key=lambda x: x[0])
  X, y_pred = zip(*sorted_zip)

  ax.plot(X, y_pred, line_style, label=f"a={alpha}")

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

f.suptitle("Ridge Regression")
f.tight_layout()
ax1.set_ylabel("y")

plot_ridge(X, y, ax=ax1, degree=1, alpha=0, line_style="C1--")
plot_ridge(X, y, ax=ax1, degree=1, alpha=10, line_style="C2:")
plot_ridge(X, y, ax=ax1, degree=1, alpha=100, line_style="C3-")

plot_ridge(X, y, ax=ax2, degree=10, alpha=0, line_style="C1--")
plot_ridge(X, y, ax=ax2, degree=10, alpha=0.1, line_style="C2:")
plot_ridge(X, y, ax=ax2, degree=10, alpha=1, line_style="C3-")

for ax in (ax1, ax2):
  ax.scatter(X, y, s=10)
  ax.set_xlabel("X1")
  ax.legend()

plt.show()
# %%
sgd_reg = SGDRegressor(loss="squared_error", penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
# %% [markdown]
# * Note: As alpha increases, the predictions becomes less extreme or more reasonable. The model's variance (fit) decreases, but bias (error) increases.
# * Note: The right plot shows a polynomial of degree 10, visualize how Ridge generalizes the fit.
# %%
from sklearn.linear_model import Lasso

def plot_lasso(X, y, ax, degree, alpha, line_style):
  """
  Plot the lasso regularization line of fit
  """
  model = Pipeline([
    ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
    ("std_scaler", StandardScaler()),
    ("lasso", Lasso(alpha=alpha, random_state=42, tol=0.1, max_iter=10000))
  ])

  model.fit(X, y)
  y_pred = model.predict(X)
  
  sorted_zip = sorted(zip(X, y_pred), key=lambda x: x[0])
  X, y_pred = zip(*sorted_zip)

  ax.plot(X, y_pred, line_style, label=f"a={alpha}")

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

f.suptitle("Lasso Regression")
f.tight_layout()
ax1.set_ylabel("y")

plot_lasso(X, y, ax=ax1, degree=1, alpha=0.001, line_style="C1--")
plot_lasso(X, y, ax=ax1, degree=1, alpha=0.1, line_style="C2:")
plot_lasso(X, y, ax=ax1, degree=1, alpha=0.5, line_style="C3-")

plot_lasso(X, y, ax=ax2, degree=10, alpha=0.001, line_style="C1--")
plot_lasso(X, y, ax=ax2, degree=10, alpha=0.1, line_style="C2:")
plot_lasso(X, y, ax=ax2, degree=10, alpha=0.5, line_style="C3-")

for ax in (ax1, ax2):
  ax.scatter(X, y, s=10)
  ax.set_xlabel("X1")
  ax.legend()

plt.show()
# %%
m = 50
X = 4 * np.random.rand(m, 1) - 2
noise = np.random.randn(m, 1)
y = (0.5 * X**2) + (1.2 * X) + 5 + noise
# %%
from sklearn.linear_model import ElasticNet

def plot_elastic_net(X, y, ax, degree, alpha, l1_ratio, line_style):
  model = Pipeline([
    ("poly_features", PolynomialFeatures(degree=degree)),
    ("std_scaler", StandardScaler()),
    ("elastic_net", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
  ])

  model.fit(X, y)
  y_pred = model.predict(X)

  sorted_zip = sorted(zip(X, y_pred), key=lambda x: x[0])
  X, y_pred = zip(*sorted_zip)

  ax.plot(X, y_pred, line_style, label=f"a={alpha}")

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

f.suptitle("Elastic Net")
f.tight_layout()
ax1.set_ylabel("y")

plot_elastic_net(X, y, ax=ax1, degree=3, alpha=0.001, l1_ratio=0.75, line_style="C1--")
plot_elastic_net(X, y, ax=ax1, degree=3, alpha=0.1, l1_ratio=0.75, line_style="C2:")
plot_elastic_net(X, y, ax=ax1, degree=3, alpha=0.5, l1_ratio=0.75, line_style="C3-")

plot_elastic_net(X, y, ax=ax2, degree=10, alpha=0.001, l1_ratio=0.25, line_style="C1--")
plot_elastic_net(X, y, ax=ax2, degree=10, alpha=0.1, l1_ratio=0.25, line_style="C2:")
plot_elastic_net(X, y, ax=ax2, degree=10, alpha=0.5, l1_ratio=0.25, line_style="C3-")

for ax in (ax1, ax2):
  ax.scatter(X, y, s=10)
  ax.set_xlabel("X1")
  ax.legend()

plt.show()
# %%
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
noise = np.random.randn(m, 1)
y = (0.5 * X**2) + (X) + 2 + noise
# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

preprocessor = Pipeline([
  ("poly_features", PolynomialFeatures(degree=60, include_bias=False)),
  ("std_scaler", StandardScaler())
])

X_train_tr = preprocessor.fit_transform(X_train)
X_val_tr = preprocessor.transform(X_val)

print(X_train_tr.shape, X_val_tr.shape)
# %%
# warm_start: continues to train the same model, instead of restarting from scratch
sgd_reg = SGDRegressor(penalty="l2", eta0=0.003, random_state=42)

n_epochs = 500
min_val_error = float("inf")
best_epoch = None
train_errors, val_errors = [], []

for epoch in range(n_epochs):
  sgd_reg.partial_fit(X_train_tr, y_train.ravel())

  y_train_pred = sgd_reg.predict(X_train_tr)
  y_val_pred = sgd_reg.predict(X_val_tr)

  train_error = mean_squared_error(y_train, y_train_pred)
  val_error = mean_squared_error(y_val, y_val_pred)

  train_errors.append(train_error)
  val_errors.append(val_error)

  if (val_error < min_val_error):
    min_val_error = val_error
    best_epoch = epoch

print(best_epoch)
# %%
plt.plot([0, n_epochs], [np.sqrt(min_val_error), np.sqrt(min_val_error)], "k:")  # best RMSE
plt.plot(np.sqrt(train_errors), "C1--", label="Training Set")
plt.plot(np.sqrt(val_errors), "C2-", label="Validation Set")
plt.plot(best_epoch, np.sqrt(val_errors[best_epoch]), "C3o", markersize=5, label="Best Model")

plt.title("Early Stopping Regularization")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.tight_layout()
plt.show()