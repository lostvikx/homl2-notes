# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import seaborn as sns
#%%
sns.set_theme(context="notebook",style="darkgrid",palette="muted")
#%%
housing_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing_path = "data/housing.csv"
def fetch_housing_data(url=housing_url,path=housing_path):
  res = requests.get(url,stream=True)
  with open(path,"wb") as f:
    for line in res:
      f.write(line)
    f.close()
  return pd.read_csv(path)
# %%
housing = fetch_housing_data()
housing.head()
# %% [markdown]
# Each instance is a block group or district in the state of California. A district has a population of 600 to 3000 people.
#
# Our model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.
#
# Frame the problem:
# * Supervised learning task, since we have been given labled training examples. (Each district's median housing price)
# * Multiple regression task, since the system will use multiple features. It is also a univariate regression problem, since we are only trying to predict a single value for each district.
# * No continuous flow of data and the data is small enough to fit on memory, so batch learning.
# %%
housing.info()
# %%
housing["ocean_proximity"].value_counts()
# %%
housing.describe()
# %%
housing.hist(bins=50,figsize=(20,15))
plt.show()
# %% [markdown]
# Information provided:
# Median income doesn't look right at first, it has been scaled and capped at 15. The numbers represent tens of thousands of dollars (3.0 == $30000).
#
# Note: Working with preprocessed attributes is common in ML.
#
# Housing median age and the median house values are also capped. The latter is a serious problem, since it is our target attribute. We have two options: (1) Collect proper labels for instance (districts) whose labels were capped. (2) Remove those districts from the training set (also from the test set).
# %%
# Custom fn to split dataset
def split_train_test(data,test_ratio=0.2):
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(test_ratio * len(data))
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]

train_set,test_set = split_train_test(housing,0.2)
print(len(train_set),len(test_set))
# %%
# A simpler way!
from sklearn.model_selection import train_test_split

train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
# %% [markdown]
# Note: Stratified sampling: the population is divided into homogeneous subgroups called stratas, the right proportion of instances are sampled from each stratum to guarantee that the test test is representative of the overall population.
#
# Suppose, we think that median_income is an important attribute to predict median housing prices. We may want to ensure that the test set is representative of the various categories of incomes in the whole dataset.
# %%
sns.histplot(housing,x="median_income",bins=10)
plt.show()
# %%
housing["income_cat"] = pd.cut(x=housing["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
# %%
sns.histplot(housing,x="income_cat")
plt.show()
# %%
from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for (train_index,test_index) in strat_split.split(housing,housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]
# %%
strat_test_set["income_cat"].value_counts(normalize=True)
# %%
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
test_set["income_cat"].value_counts(normalize=True)
# %%
def get_income_cat_proportions(data):
  return data["income_cat"].value_counts(normalize=True)
# Sampling Bias
compare_sampling = pd.DataFrame({
  "Overall": get_income_cat_proportions(housing),
  "Stratified": get_income_cat_proportions(strat_test_set),
  "Random": get_income_cat_proportions(test_set),
}).sort_index()

compare_sampling["Rand. %error"] = ((compare_sampling["Random"] / compare_sampling["Overall"]) - 1) * 100
compare_sampling["Strat. %error"] = ((compare_sampling["Stratified"] / compare_sampling["Overall"]) - 1) * 100

compare_sampling
# %%
for set_ in (strat_train_set,strat_test_set,housing):
  set_.drop("income_cat",axis=1,inplace=True)
# %%
housing = strat_train_set.copy()
# %%
housing.plot(kind="scatter",x="longitude",y="latitude")
plt.title("California Housing Data")
plt.show()
# %%
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
plt.title("High Density Points in California")
plt.show()
# %%
sns.scatterplot(data=housing,x="longitude",y="latitude",hue="ocean_proximity",alpha=0.7)
plt.show()
# %%
housing.plot(kind="scatter",x="longitude",y="latitude",
  alpha=0.4,s=housing["population"]/100,label="population",
  c="median_house_value",cmap=plt.get_cmap("jet"),
  colorbar=True,figsize=(10,7)
)
plt.show()
# %%
corr_matrix = housing.corr(numeric_only=True)
# %%
corr_matrix["median_house_value"].sort_values(ascending=False)
# %% [markdown]
# Median house value shows a strong positive correlation with median income of a district.
# Median house value shows a weak negative correlation with latitude of a district.
#
# Note: Correlation coefficient only measures linear correlations, it may completely miss out on nonlinear relationship. Eg: if x is near 0, y generally goes up.
# %%
sns.scatterplot(data=housing,x="median_income",
  y="median_house_value",alpha=0.5
)
plt.show()
# %%
attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
pd.plotting.scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()
# %%
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
housing.head()
# %%
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
# %%
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing.head()
# %% [markdown]
# Attribute Combination: rooms_per_household, bedrooms_per_room, and population_per_household.
# %%
# Data Cleaning (Should have done this earlier)
# We know that total_bedrooms attribute has some missing values, we need to fix this.
# housing.dropna(subset=["total_bedrooms"]) # Option 1: drop instances with na values
# housing.drop("total_bedrooms",axis=1) # Option 2: drop the entire attribute
# housing["total_bedrooms"].fillna(housing["total_bedrooms"].median(),inplace=True) # Option 3: fill na values with the median of that attribute.
# %%
from sklearn.impute import SimpleImputer

# def fill_missing_values(data,strategy="median"):
#   imputer = SimpleImputer(strategy=strategy)
#   data_num = data.select_dtypes(include=np.number)
#   data_non_num = housing.select_dtypes(exclude=np.number)
#   imputer.fit(data_num)
#   print(imputer.statistics_)
#   print(data_num.median(numeric_only=True).values)
#   X = imputer.transform(data_num)
#   data_tf = pd.DataFrame(X,columns=data_num.columns,index=data_num.index)
#   return pd.merge(data_tf,data_non_num,left_index=True,right_index=True)

# housing = fill_missing_values(housing)
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
housing_tr.head()
# %%
# Handling categorical data
housing_cat = housing[["ocean_proximity"]]
housing_cat.head()
# %%
housing_cat.value_counts().plot(kind="barh")
plt.show()
# %%
# If we had ordinal data:
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoder = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoder[:5]
# %%
ordinal_encoder.categories_
# %%
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
# %%
housing_cat_1hot.toarray()
# %%
cat_encoder.categories_
# %% [markdown]
# Using a Custom Transformer to add new attributes (done previously) like rooms_per_household and etc.
# %%
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix,bedrooms_ix,population_ix,household_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
  def __init__(self,add_bedrooms_per_room=True):
    self.add_bedrooms_per_room = add_bedrooms_per_room
  def fit(self,X,y=None):
    return self
  def transform(self,X,y=None):
    rooms_per_household = X[:,rooms_ix] / X[:,household_ix]
    population_per_household = X[:,population_ix] / X[:,household_ix]
    if self.add_bedrooms_per_room:
      bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
      return np.c_[X,rooms_per_household,bedrooms_per_room,population_per_household]
    else:
      return np.c_[X,rooms_per_household,population_per_household]
# %%
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
attr_adder.transform(housing.values)
# %% [markdown]
# We can use feature scaling to get all attributes to have the same scale.
# %%
housing_num = housing.drop("ocean_proximity",axis=1)
# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# imputer: fills the na values with median of that attribute
# adder: adds new attributes defined earlier
# standard_scaler: transforms the values and standardizes it
num_pipeline = Pipeline([
  ("imputer",SimpleImputer(strategy="median")),
  ("attributes_adder",CombinedAttributesAdder()),
  ("standard_scaler",StandardScaler())
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr.shape
# %%
from sklearn.compose import ColumnTransformer

num_attributes = list(housing_num.columns)
cat_attributes = ["ocean_proximity"]

# num: apply the transformation pipeline defined in the previous cell
# cat: apply onehotencoding to categorical attributes
full_pipeline = ColumnTransformer([
  ("num",num_pipeline,num_attributes),
  ("cat",OneHotEncoder(),cat_attributes)
])

housing_prepared = full_pipeline.fit_transform(housing)
# %%
# attributes: 9 + 3 (adder transform) + 4 (onehotencoder) = 16
housing_prepared.shape
# %%
# https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb