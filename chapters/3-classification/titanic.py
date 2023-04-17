# %% [markdown]
# # Predicting Survival on the Titan
# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
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

train_set = train_set.drop("PassengerId",axis=1)
test_set = test_set.drop("PassengerId",axis=1)

train_set.head()
# %%
train_set.info()
# %% [markdown]
# * Cabin feature only has 204 out of the total 891 instances, hence we drop it.
# * Age feature has a few missing values, we will fill with mean or median.
# %%
train_set.describe()
# %%
def show_cabin_floor(passenger_class,data=train_set):
  """
  params: passenger_class: int (1,2,3)
  returns: list of cabin floors
  """
  cabins = list(data.loc[data["Pclass"] == passenger_class]["Cabin"].unique())
  cabins.remove(np.nan)
  return sorted(list(set([cab[0] for cab in cabins])))

print("1st class cabins:",show_cabin_floor(1))
print("2nd class cabins:",show_cabin_floor(2))
print("3rd class cabins:",show_cabin_floor(3))
# %%
train_set["Name"].str.extract("(\w+)\.").value_counts()
# %%
def title_feature(dataset):
  dataset = dataset.copy()
  dataset["Title"] = dataset["Name"].str.extract("(\w+)\.")

  common_titles = ["Mr","Mrs","Miss","Master"]
  dataset["Title"] = dataset["Title"].apply(lambda title: title if title in common_titles else "Rare")

  return dataset.drop("Name",axis=1)
# %%
train_set_tr = title_feature(train_set)
test_set_tr = title_feature(test_set)
train_set_tr["Title"].value_counts()
# %%
# ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title"]
# %%
def drop_useless_features(dataset):
  return dataset.drop(["Ticket","Cabin"],axis=1)
# %%
titanic = train_set_tr.drop(["Ticket","Cabin"],axis=1)
titanic.head()
# %% [markdown]
# # Explore the data
# %%
titanic[["Survived","Sex"]].groupby("Sex",as_index=False).mean()
# %%
sns.countplot(data=titanic,x="Sex",hue="Survived")
plt.show()
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
def alone_feature(dataset, keep_details=False):
  dataset = dataset.copy()
  dataset["FamilyMem"] = dataset["SibSp"] + dataset["Parch"]
  dataset["Alone"] = np.where(dataset["FamilyMem"] > 0, "No", "Yes")
  if keep_details:
    return dataset
  else:
    return dataset.drop(["FamilyMem"],axis=1)
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
titanic = titanic.drop(["SibSp","Parch","FamilyMem"],axis=1)
titanic.head()
# %%
sns.scatterplot(data=titanic,x="Age",y="Fare",hue="Survived")
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
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DataTransformation(BaseEstimator, TransformerMixin):
  def __init__(self,create_title=True,keep_details=False):
    self.create_title = create_title
    self.keep_details = keep_details

  def fit(self,X,y=None):
    return self

  def transform(self,X):
    X = X.copy()
    
    if self.create_title:
      X = title_feature(X)
    X = X.drop(["Ticket","Cabin"],axis=1)
    X = alone_feature(X,keep_details=self.keep_details)

    fill_val = X[X["Pclass"] == 1]["Embarked"].value_counts().index[0]
    X = X.fillna({"Embarked": fill_val})

    return X

trans_pipeline = Pipeline([
  ("trans", DataTransformation())
])

titanic = trans_pipeline.fit_transform(train_set)
titanic.head()
# %%
y_train = titanic["Survived"].to_numpy()
titanic = titanic.drop("Survived",axis=1)
titanic.info()
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

titanic_num = titanic[["Age","Fare","SibSp","Parch"]]

num_pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy="mean")),
  ("std_scaler", StandardScaler())
])

num_pipeline.fit_transform(titanic_num).shape
# %%
from sklearn.preprocessing import OneHotEncoder

titanic_cat = titanic[["Sex","Embarked","Title","Alone"]]

cat_pipeline = Pipeline([
  ("ohe", OneHotEncoder())
])

cat_pipeline.fit_transform(titanic_cat).shape
# %%
pd.get_dummies(titanic_cat).head()
# %%
from sklearn.compose import ColumnTransformer

num_attribs = list(titanic_num.columns)
cat_attribs = list(titanic_cat.columns)

col_trans = ColumnTransformer([
  ("num",num_pipeline,num_attribs),
  ("cat",cat_pipeline,cat_attribs),
  ("ord","passthrough",["Pclass"])
])

col_trans.fit_transform(titanic).shape
# %%
preprocessor = Pipeline([
  ("relevant_features", DataTransformation()),
  ("prepare", col_trans)
])
# %%
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
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train)
# %%
sgd_clf.predict(X_train[:3])
# %%
y_train[:3]
# %%
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf,X_train,y_train,cv=5,scoring="accuracy").mean()
# %%
from sklearn.model_selection import cross_val_predict

y_pred_sgd = cross_val_predict(sgd_clf,X_train,y_train,cv=5)
# %%
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train,y_pred_sgd)
# %%
from sklearn.metrics import precision_score,recall_score

print("Precision:",precision_score(y_train,y_pred_sgd))
print("Recall:",recall_score(y_train,y_pred_sgd))
# %%
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def model_scores(model,X_train,y_true,cv=5):
  y_pred = cross_val_predict(model,X_train,y_true,cv=cv)
  print(confusion_matrix(y_true,y_pred))
  print("Accuracy:",accuracy_score(y_true,y_pred))
  print("Precision:",precision_score(y_true,y_pred))
  print("Recall:",recall_score(y_true,y_pred))
  print("F1:",f1_score(y_true,y_pred))
# %%
model_scores(sgd_clf,X_train,y_train)
# %%
from sklearn.metrics import precision_recall_curve

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
from sklearn.metrics import roc_curve, roc_auc_score

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
from sklearn.linear_model import LogisticRegression

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
from sklearn.ensemble import RandomForestClassifier

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
from sklearn.neighbors import KNeighborsClassifier

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
from sklearn.svm import SVC

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
from sklearn.model_selection import GridSearchCV

log_pipe = Pipeline([
  ("preprocessor",preprocessor),
  ("classifier",LogisticRegression(random_state=42,max_iter=200))
])

param_grid = {
  "preprocessor__relevant_features__keep_details": [False,True],
  "preprocessor__prepare__num__imputer__strategy": ["mean","median"],
  "classifier__solver": ["liblinear","lbfgs"],
  "classifier__penalty": ["l2"],
  "classifier__C": [100, 10, 1.0, 0.1, 0.01]
}

log_grid_search = GridSearchCV(log_pipe,param_grid,scoring="accuracy",cv=5,verbose=3)
# %%
log_grid_search.fit(train_set,y_train)
# %%
print("Params:",log_grid_search.best_params_)
print("Score:",log_grid_search.best_score_)
# %%
forest_pipe = Pipeline([
  ("preprocessor", preprocessor),
  ("classifier",RandomForestClassifier(random_state=42))
])

param_grid = {
  "preprocessor__relevant_features__keep_details": [False,True],
  "preprocessor__prepare__num__imputer__strategy": ["mean","median"],
  "classifier__n_estimators": [10,100,200,1000],
  "classifier__max_features": ["sqrt","log2"],
}

forest_grid_search = GridSearchCV(forest_pipe,param_grid,scoring="accuracy",cv=5,verbose=3)
# %%
forest_grid_search.fit(train_set,y_train)
# %%
print("Params:",forest_grid_search.best_params_)
print("Score:",forest_grid_search.best_score_)
# %%
svc_pipe = Pipeline([
  ("preprocessor", preprocessor),
  ("classifier",SVC(random_state=42))
])

param_grid = {
  "preprocessor__relevant_features__keep_details": [False,True],
  "preprocessor__prepare__num__imputer__strategy": ["mean","median"],
  "classifier__kernel": ["linear","poly","rbf"],
  "classifier__C": [100, 10, 1.0, 0.1, 0.001],
}

svc_grid_search = GridSearchCV(svc_pipe,param_grid,scoring="accuracy",cv=5,verbose=3)
# %%
svc_grid_search.fit(train_set,y_train)
# %%
print("Params:",svc_grid_search.best_params_)
print("Score:",svc_grid_search.best_score_)
# %% [markdown]
# Wow! SVC came out on top with the best accuracy score.
# %%
y_pred = svc_grid_search.predict(test_set)
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
save_preds(final_pred,"svc_preds")
# %%
# TODO: Do something with Cabin feature
# TODO: Impute Age based on the other columns like Pclass
# %% [markdown]
# # Ensemble Learning
# %%
def get_best_params(grid_search):
  return {key.replace("classifier__",""): val for key,val in grid_search.best_params_.items() if key.startswith("classifier__")}
# %%
from sklearn.ensemble import VotingClassifier

lr_clf = LogisticRegression(**get_best_params(log_grid_search))
svc_clf = SVC(**get_best_params(svc_grid_search))

models = [
  ("lr",lr_clf),
  ("svc",svc_clf)
]

voting_clf = VotingClassifier(estimators=models, voting="hard")
# %%
voting_clf.fit(X_train,y_train)
# %%
y_pred = voting_clf.predict(X_train)
accuracy_score(y_train,y_pred)
# %%
y_pred = voting_clf.predict(X_test)
y_pred.shape
# %%
final_pred = pd.DataFrame({
  "PassengerId": test_ids,
  "Survived": y_pred
})
final_pred.head()
# %%
save_preds(final_pred,"voting_preds")
# %%
