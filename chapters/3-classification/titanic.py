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
path = "data/titanic"

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
    return dataset.drop(["SibSp","Parch","FamilyMem"],axis=1)
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
    self.keep_family_details = keep_details

  def fit(self,X,y=None):
    return self

  def transform(self,X):
    X = X.copy()
    
    if self.create_title:
      X = title_feature(X)
    X = X.drop(["Ticket","Cabin"],axis=1)
    X = alone_feature(X,keep_details=self.keep_family_details)

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

titanic_num = titanic[["Age","Fare"]]

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
  ("cat",cat_pipeline,cat_attribs)
])

titanic_prepared = np.c_[titanic["Pclass"].to_numpy(),col_trans.fit_transform(titanic)]
titanic_prepared.shape
# %%
full_pipeline = Pipeline([
  ("relevant_features", trans_pipeline),
  ("prepare", col_trans)
])
# %%
def prepare_data(dataset):
  return np.c_[dataset["Pclass"].to_numpy(),full_pipeline.fit_transform(dataset)]
# %%
X_train = prepare_data(train_set)
X_train.shape