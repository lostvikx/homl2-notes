# %%
import numpy as np
from matplotlib import pyplot as plt
# %%
from sklearn import datasets

iris = datasets.load_iris()
print(list(iris.keys()))
# %%
print(iris["feature_names"])
print(iris["target_names"])
# %%
iris["data"][:5]
# %%
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(int)  # 1: virginica, else 0
# %%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)
# %%
X_test = np.linspace(0, 3, 500).reshape(500, 1)
y_proba = log_reg.predict_proba(X_test)

print(y_proba[:3])
# %%
decision_boundary = X_test[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(10, 5))

plt.scatter(X[y == 0], y[y == 0], s=10)
plt.scatter(X[y == 1], y[y == 1], s=10)

plt.plot(X_test, y_proba[:, 0], "C0--", label="Not Virginica")
plt.plot(X_test, y_proba[:, 1], "C1-", label="Virginica")

plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", label="Decision Boundary", linewidth=2)

plt.title("Decision Boundary")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.show()
# %%
print(decision_boundary)
# %%
log_reg.predict([[1.5], [1.7]])
# %% [markdown]
# Note: Any value under the decision_boudary is classified as Not Virginica, while a value above it is classified as a Virginica.
# %%
X = iris["data"][:, (2, 3)]  # (petal length, petal width)
y = (iris["target"] == 2).astype(int)  # 1: virginica, else 0
# %%
log_reg = LogisticRegression(random_state=42, C=10**10)  # C: alpha inverse (higher value of C, less regularized model)
log_reg.fit(X, y)
# %%
print(log_reg.coef_)
# %%
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.show()
# %% [markdown]
# Let's ignore the flowers with petal length of less than 3 in creating X_test.
# %%
petal_lens = np.linspace(2.5, 7, 500).reshape(500, 1)
petal_wids = np.linspace(0.8, 2.7, 200).reshape(200, 1)

x0, x1 = np.meshgrid(petal_lens, petal_wids)
# %%
plt.figure(figsize=(10, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Not Virginica")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Virginica")

left_right = np.array([2.5, 7])
boundary = - ((log_reg.coef_[0][0] * left_right) + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.plot(left_right, boundary, "k--", label="Decision Boundary")

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.axis([2.5, 7, 0.8, 2.7])
plt.legend()
plt.show()
# %%
plt.figure(figsize=(10, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Not Virginica")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Virginica")

left_right = np.array([2.5, 7])
boundary = - ((log_reg.coef_[0][0] * left_right) + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.plot(left_right, boundary, "k--", label="Decision Boundary")

X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_reg.predict_proba(X_new)

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz)
plt.clabel(contour)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.axis([2.5, 7, 0.8, 2.7])
plt.legend()
plt.show()
# %%
X = iris["data"][:, (2, 3)]  # (petal length, petal width)
y = iris["target"]
# %%
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
# %%
softmax_reg.predict([[5, 2]])  # Iris with petal length 5cm and petal width 2cm.
# %%
softmax_reg.predict_proba([[5, 2]])
# %%
plt.figure(figsize=(10, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Iris Setosa")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Iris Versicolor")
plt.scatter(X[y == 2, 0], X[y == 2, 1], label="Iris Virginica")

petal_lens = np.linspace(0, 8, 500).reshape(500, 1)
petal_wids = np.linspace(0, 3.5, 200).reshape(200, 1)

x0, x1 = np.meshgrid(petal_lens, petal_wids)

X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = softmax_reg.predict_proba(X_new)
y_pred = softmax_reg.predict(X_new)

zz = y_proba[:, 1].reshape(x0.shape)

contour = plt.contour(x0, x1, zz)
plt.clabel(contour)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.axis([0, 7, 0, 3.5])
plt.legend()
plt.show()
# %% [markdown]
# # Batch Gradient Descent with Early Stopping
# %%
X = iris["data"][:, (2, 3)]
y = iris["target"]
# %%
X_with_bias = np.c_[np.zeros((len(X), 1)), X]
# %%
test_ratio, val_ratio = 0.2, 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
val_size = int(total_size * val_ratio)
train_size = total_size - test_size - val_size

np.random.seed(42)
rnd_idx = np.random.permutation(total_size)

train_idx = rnd_idx[:train_size]
test_idx = rnd_idx[train_size:-test_size]
val_idx = rnd_idx[-test_size:]

X_train = X_with_bias[train_idx]
y_train = y[train_idx]

X_test = X_with_bias[test_idx]
y_test = y[test_idx]

X_val = X_with_bias[val_idx]
y_val = y[val_idx]

print(X_train.shape, X_test.shape, X_val.shape)
# %%
def one_hot_encoder(y):
  """
  Returns a One Hot Encoded version of y
  """
  n_classes = y.max() + 1
  m = len(y)
  y_one_hot = np.zeros((m, n_classes))
  y_one_hot[np.arange(m), y] = 1
  return y_one_hot
# %%
y_train[:5]
# %%
one_hot_encoder(y_train[:5])
# %%
y_train_ohe = one_hot_encoder(y_train)
y_test_ohe = one_hot_encoder(y_test)
y_val_ohe = one_hot_encoder(y_val)
# %%
def softmax(logits):
  exp = np.exp(logits)
  exp_sum = np.sum(exp, axis=1, keepdims=True)
  return exp / exp_sum
# %%
n_inputs = X_train.shape[1]  # n_features + bias (3)
n_outputs = len(np.unique(y_train))  # n_classes (3)

print(n_inputs, n_outputs)
# %%
eta = 0.03
n_iter = 5000
m = len(X_train)
epsilon = 1e-7
best_loss = float("inf")
alpha = 0.01  # regularization hyperparameter

np.random.seed(42)
theta = np.random.randn(n_inputs, n_outputs)

for i in range(n_iter):
  logits = X_train.dot(theta)
  y_proba = softmax(logits)

  xentropy_loss = -np.mean(np.sum(y_train_ohe * np.log(y_proba + epsilon), axis=1))
  l2_loss = 1/2 * np.sum(np.square(theta[1:]))
  loss = xentropy_loss + (alpha * l2_loss)

  if ((i+1) % 500 == 0) or (i == 0):
    print(i+1, loss)

  error = y_proba - y_train_ohe
  gradient = 1/m * X_train.T.dot(error)
  theta = theta - eta * gradient

  if loss < best_loss:
    best_loss = loss
  else:
    print(i - 1, best_loss)
    print(i, loss, "Early Stopping!")
    break
# %%
theta
# %%
logits = X_val.dot(theta)
y_proba = softmax(logits)
y_pred = np.argmax(y_proba, axis=1)

accuracy_score = np.mean(y_pred == y_val)
print(f"Accuracy: {accuracy_score * 100:.2f}%")
# %%
logits = X_test.dot(theta)
y_proba = softmax(logits)
y_pred = np.argmax(y_proba, axis=1)

accuracy_score = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy_score * 100:.2f}%")