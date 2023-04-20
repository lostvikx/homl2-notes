# %% [markdown]
# # Predicting Survival on the Titanic
# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# %%
sns.set_theme(context="notebook",style="darkgrid",palette="muted")
# %% [markdown]
# * Survived: 0 or 1 (label)
# * Pclass: passenger class (ordinal)
# * Name: name of the passenger (most have titles)
# * Sex: male or female
# * Age: age of the passenger
# * SibSp: # of siblings / spouses aboard
# * Parch: # of parents / children aboard
# * Ticket: ticket number
# * Fare: passenger fare
# * Cabin: cabin number
# * Embarked: port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) (Categorical)
# %%
try:
  path = "data/titanic"
  in_kaggle = False
  train_set = pd.read_csv(f"{path}/train.csv")
  test_set = pd.read_csv(f"{path}/test.csv")
except:
  path = "../input/titanic"
  in_kaggle = True
  train_set = pd.read_csv(f"{path}/train.csv")
  test_set = pd.read_csv(f"{path}/test.csv")

train_ids = train_set["PassengerId"]
test_ids = test_set["PassengerId"]

train_set = train_set.drop(["PassengerId","Ticket"],axis=1)
test_set = test_set.drop(["PassengerId","Ticket"],axis=1)

train_set.head()
# %%
train_set.info()
# %%
test_set.info()
# %% [markdown]
# ## Train Set
# * Cabin feature only has 204 out of the total 891 instances, hence we drop it.
# * Age feature has a few missing values, we will fill with mean or median.
# * Embarked: missing 2 instances
# ## Test Set
# * Cabin: only 91 of 418, we may drop it.
# * Age: 332 of 418, will fill.
# * Fare: 1 instance is missing.
# %%
train_set.describe()
# %%
def get_cabin_floors(passenger_class,data=train_set):
  """
  params: passenger_class: int (1,2,3)
  returns: list of cabin floors
  """
  cabins = list(data.loc[data["Pclass"] == passenger_class]["Cabin"].unique())
  cabins.remove(np.nan)
  return sorted(list(set([cab[0] for cab in cabins])))

print("1st class cabins:",get_cabin_floors(1))
print("2nd class cabins:",get_cabin_floors(2))
print("3rd class cabins:",get_cabin_floors(3))
# %%
def update_cabin_class(row):
  top = ["B", "C"]
  mid = ["A", "D", "E", "T"]
  low = ["F", "G"]
  cabin_class = None

  if (row["Cabin"] in top) or (row["Fare"] > 50):
    cabin_class = 3
  elif (row["Cabin"] in mid) or (25 <= row["Fare"] <= 50):
    cabin_class = 2
  elif (row["Cabin"] in low) or (row["Fare"] < 25):
    cabin_class = 1
  else:
    print("update_cabin_class has an error!")
  return cabin_class

def fix_cabin(df):
  """
  Cabin class rules:
    Top: B,C or ($50+)
    Mid: A,D,E,T or ($25 to $50)
    Low: F,G or (upto $25)
  """
  df = df.copy()
  df["Cabin"] = df["Cabin"].str.extract("(\w{1})\w*")
  df["Cabin"] = df.apply(update_cabin_class,axis=1)
  return df

class FixCabin(BaseEstimator, TransformerMixin):
  def __init__(self,fix=True):
    self.fix = fix

  def fit(self,X,y=None):
    return self

  def transform(self,X):
    if self.fix:
      X_tr = fix_cabin(X)
    else:
      X_tr = X.drop("Cabin",axis=1)
    return X_tr
# %%
test_pipe = Pipeline([
  ("fix",FixCabin())
])
# %%
test_pipe.fit_transform(train_set).info()
# %%
train_set_tr = train_set.copy()
train_set_tr["Cabin"] = train_set_tr["Cabin"].str.extract("(\w{1})")
# %%
train_set_tr[train_set_tr["Cabin"].notna()].head()
# %%
train_set_tr[train_set_tr["Cabin"].notna()][["Cabin","Fare"]].groupby("Cabin",as_index=False).mean()
# %%
train_set_tr["Cabin"] = train_set_tr.apply(update_cabin_class,axis=1)
train_set_tr["Cabin"].value_counts()
# %%
train_set_tr["Name"].str.extract("(\w+)\.").value_counts()
# %%
def title_feature(dataset):
  """
  Adds Title feature and removes Name feature.
  """
  dataset = dataset.copy()
  dataset["Title"] = dataset["Name"].str.extract("(\w+)\.")

  common_titles = ["Mr","Mrs","Miss","Master"]
  dataset["Title"] = dataset["Title"].apply(lambda title: title if title in common_titles else "Rare")

  return dataset.drop("Name",axis=1)
# %%
train_set_tr = title_feature(train_set)
train_set_tr["Title"].value_counts()
train_set_tr = test_pipe.fit_transform(train_set_tr)
# %%
titanic = train_set_tr.copy()
titanic.head()
# %% [markdown]
# # Explore the data
# %%
titanic[["Survived","Sex"]].groupby("Sex",as_index=False).mean()
# %%
sns.countplot(data=titanic,x="Sex",hue="Survived")
plt.show()
# %%
sns.countplot(data=titanic,x="Cabin",hue="Survived")
plt.show()
# %%
titanic[["Survived","Cabin"]].groupby("Cabin",as_index=False).mean()
# %%
sns.countplot(data=titanic,x="Survived")
plt.show()
# %%
titanic[["Survived","Pclass"]].groupby("Pclass",as_index=False).mean()
# %%
sns.countplot(data=titanic,x="Pclass",hue="Survived")
plt.show()
# %%
titanic[["Survived","Embarked"]].groupby("Embarked",as_index=False).mean()
# %%
sns.countplot(data=titanic,x="Embarked",hue="Survived")
plt.show()
# %%
sns.countplot(data=titanic,x="Pclass",hue="Sex")
plt.show()
# %%
sns.displot(data=titanic,x="Age",kde=True)
plt.show()
# %%
sns.displot(data=titanic,x="Fare",kde=True)
plt.show()
# %%
sns.displot(data=titanic[titanic["Fare"] <= 300],x="Fare",kde=True)
plt.show()
# %%
sns.displot(data=titanic[titanic["Fare"] <= 100],x="Fare",kde=True,hue="Pclass",multiple="stack")
plt.show()
# %%
sns.displot(data=titanic,x="Age",hue="Survived",multiple="stack")
plt.show()
# %%
try:
  corr_mx = titanic.corr(numeric_only=True)
except:
  corr_mx = titanic.corr()
corr_mx
# %%
corr_mx["Survived"].sort_values(ascending=False)
# %%
titanic[["Survived","Fare"]].groupby("Survived",as_index=False).mean()
# %%
sns.catplot(data=titanic,x="Sex",y="Survived",hue="Pclass",kind="bar")
plt.show()
# %%
sns.countplot(data=titanic,x="SibSp",hue="Survived")
plt.show()
# %%
def alone_feature(dataset):
  """
  Adds FamilyMem & Alone feature.
  """
  dataset = dataset.copy()
  dataset["FamilyMem"] = dataset["SibSp"] + dataset["Parch"]
  dataset["Alone"] = np.where(dataset["FamilyMem"] > 0, "No", "Yes")
  return dataset
# %%
titanic["FamilyMem"] = titanic["SibSp"] + titanic["Parch"]
titanic.head()
# %%
sns.countplot(data=titanic,x="FamilyMem",hue="Survived")
plt.show()
# %%
# Check if passenger is alone!
titanic["Alone"] = np.where(titanic["FamilyMem"] > 0, 0, 1)
titanic.head()
# %%
sns.countplot(data=titanic,x="Alone",hue="Survived")
plt.show()
# %%
try:
  corr_mx = titanic.corr(numeric_only=True)
except:
  corr_mx = titanic.corr()
corr_mx
# %%
corr_mx["Survived"].sort_values(ascending=False)
# %%
# titanic = titanic.drop(["SibSp","Parch","FamilyMem"],axis=1)
# titanic.head()
# %%
titanic["FamilyMem"].value_counts()
# %%
titanic.describe()
# %%
sns.scatterplot(data=titanic,x="Age",y="Fare",hue="Survived")
plt.show()
# %%
sns.scatterplot(data=titanic,x="Age",y="Fare",hue="Cabin")
plt.show()
# %%
# Just found this out!
titanic[titanic["Embarked"].isna()]
# Both are of Pclass 1
# %%
sns.countplot(data=titanic,x="Pclass",hue="Embarked")
plt.show()
# %%
sns.countplot(data=titanic[titanic["Pclass"] == 1],x="Embarked")
plt.show()
# %%
fill_val = titanic[titanic["Pclass"] == 1]["Embarked"].value_counts().index[0]
titanic = titanic.fillna({"Embarked": fill_val})
titanic["Embarked"].value_counts()
# %% [markdown]
# # Prepare the data
# %%
class AdditionalFeatures(BaseEstimator, TransformerMixin):
  def __init__(self):
    return None

  def fit(self,X,y=None):
    return self

  def transform(self,X):
    X_tr = title_feature(X)
    X_tr = alone_feature(X_tr)
    return X_tr

# Testing
trans_pipeline = Pipeline([
  ("trans", AdditionalFeatures())
])

titanic = trans_pipeline.fit_transform(train_set)
titanic.head()
# %%
y_train = titanic["Survived"].to_numpy()
titanic = titanic.drop("Survived",axis=1)
titanic.info()
# %%
titanic = test_pipe.fit_transform(titanic)
titanic.head()
# %%
titanic_num = titanic[["Age","Fare","SibSp","Parch"]]

num_pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy="mean")),
  ("std_scaler", StandardScaler())
])

num_pipeline.fit_transform(titanic_num).shape
# %%
titanic_cat = titanic[["Sex","Embarked","Title","Alone"]]

cat_pipeline = Pipeline([
  ("ohe", OneHotEncoder())
])
# Don't worry, we get an extra, because we didn't fillna.
cat_pipeline.fit_transform(titanic_cat).shape
# %%
pd.get_dummies(titanic_cat).head()
# %%
num_attribs = list(titanic_num.columns)
cat_attribs = list(titanic_cat.columns)

col_trans = ColumnTransformer([
  ("num",num_pipeline,num_attribs),
  ("cat",cat_pipeline,cat_attribs)
], remainder="passthrough")

col_trans.fit_transform(titanic).shape
# %%
def fill_values(df, col_name):
  """
  Fills the missing values in Embarked, Age, and Fare (test_set) features.
  """
  df = df.copy()
  if col_name == "Embarked":
    # Both missing instances belong to Pclass == 1
    val = df[df["Pclass"] == 1]["Embarked"].value_counts().index[0]
    df = df.fillna({"Embarked": val})
  elif col_name == "Age":
    # Age according to the Pclass.
    # Age according to Fare price.
    for n in range(1,4):
      mean_age = df[df["Pclass"] == n]["Age"].mean()
      df.loc[df["Pclass"] == n, "Age"] = df.loc[df["Pclass"] == n, "Age"].fillna(mean_age)
  elif col_name == "Fare":
    # Only 1 missing instance of Pclass == 3
    mean_fare = df[df["Pclass"] == 3]["Fare"].mean()
    df = df.fillna({"Fare": mean_fare})
  return df

class FillMissingFeatures(BaseEstimator, TransformerMixin):
  def __init__(self, fill_missing_by_pclass=True):
    self.fill_missing_by_pclass = fill_missing_by_pclass

  def fit(self,X,y=None):
    return self

  def transform(self,X):
    if self.fill_missing_by_pclass:
      X_tr = fill_values(X, "Embarked")
      X_tr = fill_values(X_tr, "Age")
      X_tr = fill_values(X_tr, "Fare")
    else:
      # Test set has a missing Fare feature vector.
      X_tr = fill_values(X, "Fare")
    return X_tr
# %%
test_pipe = Pipeline([
  ("trans", FillMissingFeatures())
])

test_pipe.fit_transform(train_set).info()
# %%
test_pipe.fit_transform(test_set).info()
# %%
preprocessor = Pipeline([
  ("additional_features", AdditionalFeatures()),
  ("fill_missing", FillMissingFeatures()),
  ("fix_cabin", FixCabin()),
  ("prepare", col_trans)
])
# %%
train_set = train_set.drop("Survived",axis=1)
X_train = preprocessor.fit_transform(train_set)
X_test = preprocessor.fit_transform(test_set)
print(X_train.shape,X_test.shape)
# %% [markdown]
# # Predictions using ML: Classification
# * SGDClassifier
# * LogisticRegression
# * RandomForestClassifier
# * KNeighborsClassifier
# %%
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train)
# %%
y_train[:3]
# %%
sgd_clf.predict(X_train[:3])
# %%
cross_val_score(sgd_clf,X_train,y_train,cv=5,scoring="accuracy").mean()
# %%
y_pred_sgd = cross_val_predict(sgd_clf,X_train,y_train,cv=5)
# %%
confusion_matrix(y_train,y_pred_sgd)
# %%
print("Precision:",precision_score(y_train,y_pred_sgd))
print("Recall:",recall_score(y_train,y_pred_sgd))
# %%
def model_scores(model,X_train,y_true,cv=5):
  """
  Print a model's accuracy, precision, recall, and f1 scores.
  """
  y_pred = cross_val_predict(model,X_train,y_true,cv=cv)
  print(confusion_matrix(y_true,y_pred))
  print("Accuracy:",accuracy_score(y_true,y_pred))
  print("Precision:",precision_score(y_true,y_pred))
  print("Recall:",recall_score(y_true,y_pred))
  print("F1:",f1_score(y_true,y_pred))
# %%
model_scores(sgd_clf,X_train,y_train)
# %%
y_scores_sgd = cross_val_predict(sgd_clf,X_train,y_train,cv=5,method="decision_function")

prec_sgd, recall_sgd, thres_sgd = precision_recall_curve(y_train,y_scores_sgd)

def plot_precision_recall_threshold(precisions,recalls,thresholds):
  plt.plot(thresholds,precisions[:-1],label="Precision")
  plt.plot(thresholds,recalls[:-1],label="Recall")
  plt.title("Precision/Recall Trade-off")
  plt.legend()

plot_precision_recall_threshold(prec_sgd,recall_sgd,thres_sgd)
plt.show()
# %%
def plot_precision_recall(precisions,recalls,label=None):
  plt.plot(recalls,precisions,label=label)
  plt.xlabel("Recall")
  plt.ylabel("Precision")

plot_precision_recall(prec_sgd,recall_sgd)
plt.show()
# %%
fpr_sgd,tpr_sgd,thres_sgd = roc_curve(y_train,y_scores_sgd)

def plot_roc_curve(fpr,tpr,label=None):
  plt.plot(fpr,tpr,label=label)
  plt.plot([0,1],[0,1], "--")
  plt.xlabel("False Positive Rate (Fall-Out)")
  plt.ylabel("True Positive Rate (Recall)")

plot_roc_curve(fpr_sgd,tpr_sgd)
plt.show()
# %%
roc_auc_score(y_train,y_scores_sgd)
# %%
log_clf = LogisticRegression(random_state=42)
log_clf.fit(X_train,y_train)
# %%
log_clf.predict(X_train[:3])
# %%
cross_val_score(log_clf,X_train,y_train,cv=5,scoring="accuracy").mean()
# %%
y_scores_log = cross_val_predict(log_clf,X_train,y_train,cv=5,method="decision_function")

prec_log, recall_log, thres_log = precision_recall_curve(y_train,y_scores_log)

plot_precision_recall(prec_sgd,recall_sgd,label="SGD")
plt.plot(recall_log,prec_log,label="Logistic")
plt.legend()
plt.show()
# %%
fpr_log,tpr_log,thres_log = roc_curve(y_train,y_scores_log)

plot_roc_curve(fpr_log,tpr_log,label="Logistic")
plt.plot(fpr_sgd,tpr_sgd,label="SGD")
plt.legend()
plt.show()
# %%
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train,y_train)
# %%
forest_clf.predict(X_train[:3])
# %%
cross_val_score(forest_clf,X_train,y_train,cv=5,scoring="accuracy").mean()
# %%
y_scores_forest = cross_val_predict(forest_clf,X_train,y_train,cv=5,method="predict_proba")[:,1]

prec_for,recall_for,thres_for = precision_recall_curve(y_train,y_scores_forest)

plot_precision_recall(prec_for,recall_for,label="RandomForest")
plt.plot(recall_sgd,prec_sgd,label="SGD")
plt.plot(recall_log,prec_log,label="Logistic")
plt.legend()
plt.show()
# %%
fpr_for,tpr_for,thres_for = roc_curve(y_train,y_scores_forest)

plot_roc_curve(fpr_for,tpr_for,label="RandomForest")
plt.plot(fpr_sgd,tpr_sgd,label="SGD")
plt.plot(fpr_log,tpr_log,label="Logistic")
plt.legend()
plt.show()
# %%
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_train)
# %%
knn_clf.predict(X_train[:3])
# %%
cross_val_score(knn_clf,X_train,y_train,cv=5,scoring="accuracy").mean()
# %%
y_scores_knn = cross_val_predict(knn_clf,X_train,y_train,cv=5,method="predict_proba")[:,1]

prec_knn,recall_knn,thres_knn = precision_recall_curve(y_train,y_scores_knn)

plot_precision_recall(prec_knn,recall_knn,label="KNN")
plt.plot(recall_sgd,prec_sgd,label="SGD")
plt.plot(recall_log,prec_log,label="Logistic")
plt.plot(recall_for,prec_for,label="RandomForest")
plt.legend()
plt.show()
# %%
fpr_knn,tpr_knn,thres_knn = roc_curve(y_train,y_scores_knn)

plot_roc_curve(fpr_knn,tpr_knn,label="KNN")
plt.plot(fpr_sgd,tpr_sgd,label="SGD")
plt.plot(fpr_log,tpr_log,label="Logistic")
plt.plot(fpr_for,tpr_for,label="RandomForest")
plt.legend()
plt.show()
# %%
svc_clf = SVC(random_state=42)
svc_clf.fit(X_train,y_train)
# %%
svc_clf.predict(X_train[:3])
# %%
cross_val_score(svc_clf,X_train,y_train,cv=5,scoring="accuracy").mean()
# %%
y_scores_svc = cross_val_predict(svc_clf,X_train,y_train,cv=5,method="decision_function")

prec_svc,recall_svc,thres_svc = precision_recall_curve(y_train,y_scores_svc)

plot_precision_recall(prec_svc,recall_svc,label="SVC")
plt.plot(recall_knn,prec_knn,label="KNN")
plt.plot(recall_sgd,prec_sgd,label="SGD")
plt.plot(recall_log,prec_log,label="Logistic")
plt.plot(recall_for,prec_for,label="RandomForest")
plt.legend()
plt.show()
# %%
fpr_svc,tpr_svc,thres_svc = roc_curve(y_train,y_scores_svc)

plot_roc_curve(fpr_svc,tpr_svc,label="SVC")
plt.plot(fpr_knn,tpr_knn,label="KNN")
plt.plot(fpr_sgd,tpr_sgd,label="SGD")
plt.plot(fpr_log,tpr_log,label="Logistic")
plt.plot(fpr_for,tpr_for,label="RandomForest")
plt.legend()
plt.show()
# %%
print("---ROC AUC Score---")
print("SGD:",roc_auc_score(y_train,y_scores_sgd))
print("Logistic:",roc_auc_score(y_train,y_scores_log))
print("RandomForest:",roc_auc_score(y_train,y_scores_forest))
print("KNN:",roc_auc_score(y_train,y_scores_knn))
print("SVC:",roc_auc_score(y_train,y_scores_svc))
# %%
print("SGD Scores")
model_scores(sgd_clf,X_train,y_train)
# %%
print("Logistic Scores")
model_scores(log_clf,X_train,y_train)
# %%
print("RandomForest Scores")
model_scores(forest_clf,X_train,y_train)
# %%
print("KNN Scores")
model_scores(knn_clf,X_train,y_train)
# %%
print("SVC Scores")
model_scores(svc_clf,X_train,y_train)
# %% [markdown]
# # Tuning Hyperparameters

# We saw that `LogisticRegression`, `RandomForestClassifier`, and `SVC` perform well on our dataset. Hence, we will try to tune their hyperparameters to find an optimal model that generalizes well on the test set.
# %%
train_set.head()
# %%
preprocessor_params = {
  "preprocessor__fill_missing__fill_missing_by_pclass": [True,False],
  "preprocessor__prepare__num__imputer__strategy": ["mean","median"],
  "preprocessor__fix_cabin__fix": [True,False]
}
# %%
def get_best_params(grid_search):
  return {key.replace("classifier__", ""): val for (key, val) in grid_search.best_params_.items() if key.startswith("classifier__")}

verbose_lvl = 2
if in_kaggle:
  verbose_lvl = 1
# %%
log_pipe = Pipeline([
  ("preprocessor",preprocessor),
  ("classifier",LogisticRegression(random_state=42))
])

param_grid = {
  **preprocessor_params,
  "classifier__solver": ["liblinear","lbfgs"],
  "classifier__max_iter": [1000],
  "classifier__penalty": ["l2"],
  "classifier__C": [100, 10, 1.0, 0.1, 0.01]
}

log_grid_search = GridSearchCV(log_pipe,param_grid,scoring="accuracy",cv=5,verbose=verbose_lvl)
# %%
log_grid_search.fit(train_set,y_train)
# %%
print(log_grid_search.best_params_)
print("Params:",get_best_params(log_grid_search))
print("Score:",log_grid_search.best_score_)
# %%
forest_pipe = Pipeline([
  ("preprocessor", preprocessor),
  ("classifier",RandomForestClassifier(random_state=42))
])

param_grid = {
  **preprocessor_params,
  "classifier__n_estimators": [200,225,250],
  "classifier__max_features": ["sqrt","log2"],
}

forest_grid_search = GridSearchCV(forest_pipe,param_grid,scoring="accuracy",cv=5,verbose=verbose_lvl)
# %%
forest_grid_search.fit(train_set,y_train)
# %%
print(forest_grid_search.best_params_)
print("Params:",get_best_params(forest_grid_search))
print("Score:",forest_grid_search.best_score_)
# %%
knn_pipe = Pipeline([
  ("preprocessor", preprocessor),
  ("classifier", KNeighborsClassifier())
])

param_grid = {
  **preprocessor_params,
  "classifier__weights": ["uniform","distance"],
  "classifier__n_neighbors": [7,9,11],
  "classifier__metric": ["minkowski","manhattan","euclidean"]
}

knn_grid_search = GridSearchCV(knn_pipe,param_grid,scoring="accuracy",cv=5,verbose=verbose_lvl)
# %%
knn_grid_search.fit(train_set,y_train)
# %%
print(knn_grid_search.best_params_)
print("Params:",get_best_params(knn_grid_search))
print("Score:",knn_grid_search.best_score_)
# %%
svc_pipe = Pipeline([
  ("preprocessor", preprocessor),
  ("classifier",SVC(random_state=42))
])

param_grid = {
  **preprocessor_params,
  "classifier__kernel": ["poly","rbf"],
  "classifier__C": [0.1, 1.0, 10.0],
  "classifier__gamma": ["scale","auto"]
}

svc_grid_search = GridSearchCV(svc_pipe,param_grid,scoring="accuracy",cv=5,verbose=verbose_lvl)
# %%
svc_grid_search.fit(train_set,y_train)
# %%
print(svc_grid_search.best_params_)
print("Params:",get_best_params(svc_grid_search))
print("Score:",svc_grid_search.best_score_)
# %%
model_names = ["Logistic","RandomForest","KNeighbors","SVC"]
grids = [log_grid_search,forest_grid_search,knn_grid_search,svc_grid_search]
scores = []

for (model,grid) in zip(model_names,grids):
  score = grid.best_score_
  print(f"{model} Score: {score}")
  scores.append(score)

(val, idx) = max((val,idx) for (idx,val) in enumerate(scores))
best_model = grids[idx]
best_model
# %% [markdown]
# Wow! SVC came out on top with the best accuracy score.
# %%
y_pred = best_model.predict(test_set)
y_pred.shape
# %%
final_pred = pd.DataFrame({
  "PassengerId": test_ids,
  "Survived": y_pred
})
final_pred.head()
# %%
def save_preds(predictions,name="submission"):
  if in_kaggle:
    predictions.to_csv(f"{name}.csv",index=False)
  else:
    predictions.to_csv(f"{path}/{name}.csv",index=False)
# %%
save_preds(final_pred)