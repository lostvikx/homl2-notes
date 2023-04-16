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
def add_title(dataset):
  dataset = dataset.copy()
  dataset["Title"] = dataset["Name"].str.extract("(\w+)\.")

  common_titles = ["Mr","Mrs","Miss","Master"]

  dataset["Title"] = dataset["Title"].apply(lambda title: title if title in common_titles else "Rare")

  return dataset
# %%
train_set = add_title(train_set)
test_set = add_title(test_set)
train_set["Title"].value_counts()
# %%
# ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title"]
# %%
titanic = train_set.drop(["Name","Ticket","Cabin"],axis=1)
X_test = test_set.drop(["Name","Ticket","Cabin"],axis=1)
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
def add_family_members(dataset,keep_all=False):
  dataset = dataset.copy()
  dataset["FamilyMem"] = dataset["SibSp"] + dataset["Parch"]
  dataset["Alone"] = np.where(dataset["FamilyMem"] > 0, 0, 1)
  if keep_all:
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
