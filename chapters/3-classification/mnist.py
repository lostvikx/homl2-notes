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
# %%
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
# %%
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5,y_train_pred)
# %% [markdown]
# * Row: actual class (neg,pos)
# * Column: predicted class (neg,pos)
# * [[TN, FP], [FN, TP]]
# %%
confusion_matrix(y_train_5,y_train_5) # pretend perfection
# %% [markdown]
# ## Important Metrics 
# * Precision: accuracy of positive decisions.
# * Recall: sensitivity or true positive rate.
# %%
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5,y_train_pred)
# %%
recall_score(y_train_5,y_train_pred)
# %% [markdown]
# Use f1_score (the harmonic mean of precision_score and recall_score) is used to compare different models (ML algorithms).
# %%
from sklearn.metrics import f1_score

f1_score(y_train_5,y_train_pred)
# %% [markdown]
# Note: Increasing precision reduces recall, this is called the precision/recall trade-off.
# %%
# Specific to our estimator: SGDClassifier
y_scores = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
y_scores
# %%
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
# %%
def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
  plt.plot(thresholds,precisions[:-1],label="Precision")
  plt.plot(thresholds,recalls[:-1],label="Recall")
  plt.legend()
  plt.axis([-40000,40000,0,1])
  plt.title("Precision/Recall Trade-off")
  plt.xlabel("Threshold")

plt.figure(figsize=(8,4))
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()
# %% [markdown]
# Sometimes increasing the threshold (moving it to the right) may result in the decrease in precision (mostly increases). While recall can only decrease as threshold is increased.
# %%
# Threshold is 0
(y_train_pred == (y_scores > 0)).all()
# %%

# %%
def plot_precision_vs_recall(precisions,recalls,label=None):
  plt.plot(recalls,precisions,label=label)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  # plt.axis([0,1,0,1])

# plt.figure(figsize=(8,6))
plot_precision_vs_recall(precisions,recalls)
plt.show()
# %%
# Suppose we want a 90% precision.
# np.argmax: returns first index of max value (True value)
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
threshold_90_precision
# %%
# Prediction: All y_scores >= threshold_90_precision should be True.
y_train_pred_90 = (y_scores >= threshold_90_precision)
# %%
precision_score(y_train_5,y_train_pred_90)
# %%
recall_score(y_train_5,y_train_pred_90)
# %% [markdown]
# We get a classifier that has 90% precision, but with a recall of about 48%.
# %% [markdown]
# ## ROC Curve
# * Plots Recall vs (1 - Specificity)
# * Recall: True positive rate
# * Specificity: True negative rate
# %%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5,y_scores)
# %%
def plot_roc_curve(fpr,tpr,label=None):
  plt.plot(fpr,tpr,label=label)
  plt.plot([0,1],[0,1], "--")
  plt.xlabel("False Positive Rate (Fall-Out)")
  plt.ylabel("True Positive Rate (Recall)")
  # plt.axis([0,1,0,1])

plot_roc_curve(fpr,tpr)
plt.show()
# %%
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5,y_scores)
# %%
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100,random_state=42)
y_probas_forest = cross_val_predict(forest_clf,X_train,y_train_5,cv=3,method="predict_proba")
# %%
y_probas_forest[0]
# %%
y_scores_forest = y_probas_forest[:,1]
fpr_forest,tpr_forest,thresholds_forest = roc_curve(y_train_5,y_scores_forest)
# %%
plt.plot(fpr,tpr,label="SGD")
plot_roc_curve(fpr_forest,tpr_forest,"RandomForest")
plt.legend()
plt.show()
# %%
roc_auc_score(y_train_5,y_scores_forest)
# %% [markdown]
# Trying to measure the precision and recall using RandomForest.
# %%
y_pred_forest = cross_val_predict(forest_clf,X_train,y_train_5,cv=3)
# %%
confusion_matrix(y_train_5,y_pred_forest)
# %%
precision_score(y_train_5,y_pred_forest)
# %%
recall_score(y_train_5,y_pred_forest)
# %%
f1_score(y_train_5,y_pred_forest)
# %%
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5,y_scores_forest)
# %%
plt.plot(recalls,precisions,label="SGD")
plot_precision_vs_recall(precisions_forest,recalls_forest,label="RandomForest")
plt.legend()
plt.show()
# %% [markdown]
# # Multiclass Classification
# They can distinguish between more than two classes. Not all classification predictive models support multi-class classification.
# * Binary Classifiers: Logistic Regression, Perceptron, and SVM.
# * OvR (One vs Rest): It involves splitting the multi-class dataset into multiple binary classification problems. A possible downside of this approach is that it requires one model to be created for each class. (Example: 0-detector, 1-detector, ...)
# * OvO (One vs One): Unlike one-vs-rest that splits it into one binary dataset for each class, the one-vs-one approach splits the dataset into one dataset for each class versus every other class. (Example: 0s vs 1s, 0s, vs 2s, ...) Note: n_estimators = (n_classes * (n_classes – 1)) / 2
# %%
from sklearn.svm import SVC

svm_clf = SVC()  # Uses OvO by default (One vs One strategy)
svm_clf.fit(X_train,y_train)
svm_clf.predict([some_digit])
# %%
some_digit_score = svm_clf.decision_function([some_digit])
some_digit_score
# %%
svm_clf.classes_[np.argmax(some_digit_score)]
# %%
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train,y_train)
# %%
ovr_clf.predict([some_digit])
# %%
len(ovr_clf.estimators_)
# %%
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train)
sgd_clf.predict([some_digit])
# %%
sgd_clf.decision_function([some_digit])
# %% [markdown]
# Unfortunately, we get the wrong result here.
# %%
cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring="accuracy")
# %%
# Try to improve the accuracy of the model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# %%
cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring="accuracy")

# %%
