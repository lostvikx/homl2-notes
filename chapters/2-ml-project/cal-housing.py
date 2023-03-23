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
# housing = strat_train_set.copy()
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
# %% [markdown]
# Attribute Combination: rooms_per_household, bedrooms_per_room, and population_per_household.
# %%
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
housing.head()
# %%
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
# %%
