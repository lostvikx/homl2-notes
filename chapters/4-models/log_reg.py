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
