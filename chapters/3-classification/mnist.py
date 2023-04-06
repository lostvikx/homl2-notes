# %% [markdown]
# # MNIST Classification
# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# %%
sns.set_theme(context="notebook",style="darkgrid",palette="muted")
# %%
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784",version=1,parser="auto")
mnist.keys()
# %% [markdown]
# Datasets loaded by sklearn are like pandas, with keys such as DESCR: description of the dataset, data: 2D array, and target: array with labels.
# %%
X,y = mnist["data"],mnist["target"]
print(X.shape,y.shape)
# %% [markdown]
# Note: 70000 images with 784 features: 28x28 pixels image.
# %%
some_digit = X.iloc[0].to_numpy()
some_digit_image = some_digit.reshape((28,28))

plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")
plt.show()
# %%
y.iloc[0]  # label of index = 0
# %%
y = y.astype(np.uint8)
# %%
X = X.to_numpy()
y = y.to_numpy()
# %%
# Split the data into train and test set!
X_train,X_test = X[:60000],X[60000:]
y_train,y_test = y[:60000],y[60000:]
# %%
import random

f, ax = plt.subplots(nrows=10,ncols=10)

def fetch_random_image(X):
  rand_index = random.randint(0,10000)
  some_img = X[rand_index]
  return some_img.reshape((28,28))

for row in ax:
  for sub in row:
    sub.imshow(fetch_random_image(X),cmap="binary")
    sub.axis("off")

plt.show()
# %% [markdown]
# # Binary Classifier
# Capable of distinguishing between 5 and not-5.
# %%
y_train_5 = (y_train == 5)  # True for 5, False for rest.
y_test_5 = (y_test == 5)
# %%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
# %%
sgd_clf.predict([X_train[0]])
# %% [markdown]
# # Implementing Cross-Validation
# %%
# This is how cross_val_score function works under the hood.
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3,random_state=42,shuffle=True)

for train_idx, test_idx in skfolds.split(X_train,y_train_5):
  clone_clf = clone(sgd_clf)
  X_train_folds,y_train_folds = X_train[train_idx],y_train_5[train_idx]
  X_test_fold,y_test_fold = X_train[test_idx],y_train_5[test_idx]

  clone_clf.fit(X_train_folds,y_train_folds)
  y_pred = clone_clf.predict(X_test_fold)
  n_correct = sum(y_pred == y_test_fold)
  print(n_correct/len(y_pred))
# %%
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring="accuracy")
