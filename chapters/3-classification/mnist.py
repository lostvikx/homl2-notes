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
  rand_index = random.randint(0,len(X)-1)
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
# %% [markdown]
# ## Error Analysis
# %%
y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
# %%
conf_mx = confusion_matrix(y_train,y_train_pred)
conf_mx
# %%
plt.figure(figsize=(8,6))
sns.heatmap(conf_mx)
plt.show()
# %%
# Sum of actual classes
# row_sums will be used to calculate percent of correctly classified.
row_sums = conf_mx.sum(axis=1,keepdims=True)
row_sums
# %%
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx,0)
sns.heatmap(norm_conf_mx)
plt.show()
# %% [markdown]
# * Columns: Prediction, Rows: Actual
# * We can clearly see that column 8 is bright, meaning that some numbers are misclassified as 8. 3s and 5s often get misclassified as 8s.
# %%
cl_5,cl_8 = 5,8
X_55 = X_train[(y_train == cl_5) & (y_train_pred == cl_5)]
X_58 = X_train[(y_train == cl_5) & (y_train_pred == cl_8)]
X_85 = X_train[(y_train == cl_8) & (y_train_pred == cl_5)]
X_88 = X_train[(y_train == cl_8) & (y_train_pred == cl_8)]
# %%
# f,ax = plt.subplots(nrows=2,ncols=2)

# sub.imshow(fetch_random_image(X_55),cmap="binary")
# sub.axis("off")
# %%
import math

def plot_digits(instances,images_per_row=5):
  size = 28
  n_rows = math.ceil(len(instances) / images_per_row)
  image_grid = instances.reshape((n_rows, images_per_row, size, size))
  big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,images_per_row*size)
  plt.imshow(big_image,cmap="binary")
  plt.axis("off")

def rand_sample_indices(X,n=25):
  return np.random.randint(0,len(X)-1,n)

plt.figure(figsize=(10,10))

plt.subplot(221).set_title("Correctly Predicted as 5")
plot_digits(X_55[rand_sample_indices(X_55)])

plt.subplot(222).set_title("Wrongly Predicted as 8")
plot_digits(X_58[rand_sample_indices(X_58)])

plt.subplot(223).set_title("Wrongly Predicted as 5")
plot_digits(X_85[rand_sample_indices(X_85)])

plt.subplot(224).set_title("Correctly Predicted as 8")
plot_digits(X_88[rand_sample_indices(X_88)])

plt.show()
# %% [markdown]
# # Multilabel Classification
# In some cases we may want our classifier to output multiple classes for each instance. A classification system that outputs multiple binary tags is called a multilabel classification system.
# %%
from sklearn.neighbors import KNeighborsClassifier

y_train_big = (y_train >= 7)  # Identified class is greater than or equal to 7.
y_train_odd = (y_train % 2 == 1)  # Identified class is an odd number.
y_train_multilabel = np.c_[y_train_big,y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_train_multilabel)
# %%
knn_clf.predict([some_digit])
# %% [markdown]
# The above output suggest that the output is less than 7 and is an odd number.
# %%
y_train_pred_knn = cross_val_predict(knn_clf,X_train,y_train_multilabel,cv=3)
# %%
f1_score(y_train_multilabel,y_train_pred_knn,average="macro")
# %% [markdown]
# # Multioutput Classification
# Each label can be multiclass. We can build a system to remove random noise from an image and hopefully output a clean digit. Output is going to be a 28x28 size array, therefore a multilayer output (one label per pixel). And each label can have multiple outputs (pixel intensity: 0 to 255).
# %%
# Only adding pixel intensity ranging from [0,100)
noise = np.random.randint(0,100,size=(len(X_train),28*28))
X_train_mod = X_train + noise
# %%
noise = np.random.randint(0,100,size=(len(X_test),28*28))
X_test_mod = X_test + noise
# %%
y_train_mod = X_train
y_test_mod = X_test
# %%
some_index = random.randint(0,len(X_test_mod)-1)
def plot_digit(instance):
  img = instance.reshape((28,28))
  plt.imshow(img,cmap="binary")
  plt.axis("off")

plt.figure(figsize=(10,10))

plt.subplot(121).set_title("Input")
plot_digit(X_test_mod[some_index])

plt.subplot(122).set_title("Expected Output")
plot_digit(y_test_mod[some_index])

print(f"Number: {y_test[some_index]}")
# %%
knn_clf.fit(X_train_mod,y_train_mod)
# %%
clean_digit = knn_clf.predict([X_test_mod[some_index]])
# %%
plt.figure(figsize=(5,5))
plot_digit(clean_digit)
plt.title("Actual Output")
plt.show()