# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
sns.set_theme(context="notebook",style="darkgrid",palette="muted")
# %%
kaggle_path = "../input/diamonds/diamonds.csv"
try:
  diamonds = pd.read_csv("data/diamonds.csv").iloc[:,1:]
except:
  print("Kaggle Notebook")
  diamonds = pd.read_csv(kaggle_path).iloc[:,1:]
diamonds.head()
# %% [markdown]
# # Understanding the features:
# * carat: weight of the diamond
# * cut: (Fair, Good, Very Good, Premium, Ideal) ordinal
# * color: J (worst) to D (best) ordinal
# * clarity: [I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)]
# * depth: z / (mean(x,y)) = 2*z / (x+y) in percentage
# * table: percentage of width on the top relative to width
# * price: given in US Dollars
# * x: length in mm
# * y: width in mm
# * z: depth in mm
# %% [markdown]
# ![Diamond Diagram](https://lh6.googleusercontent.com/SOSfDhRFlincUXUkLr_EbLxkRELE105iYQZlggeeRWCBuk15wJXkHxcNwiWpQaL3vwUCwSqrKmgSwSEWS3Gn9KpAFrgaFJ6G1hby8VJ2bb2xrdDSBkSi2tgYQZMQ5IKgeHFVGBUE)
# %%
diamonds.info()
# %%
diamonds["depth"] = 2 * diamonds["z"] / (diamonds["x"] + diamonds["y"])
diamonds["table"] = diamonds["table"] / 100
diamonds.head()
# %%
diamonds.shape
# %%
diamonds.describe()
# %%
diamonds = diamonds.dropna(axis=0)
# %% [markdown]
# Featrues such as depth, x, y, and z have a min of zero, they require cleaning.
# %%
sns.scatterplot(data=diamonds,x="x",y="y")
plt.show()
# %% [markdown]
# We can clearly see a few outliers in our dataset. We are going to use Tukey's method to get rid of outliers in our data.
# %%
def remove_outliers(data,outlier_cols,k=1.5):
  df = data.copy()
  for col in outlier_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1 # inter quartile range
    # ~ is used to inversing boolean (bitwise not)
    df = df.loc[~((df[col] < (q1 - (k * iqr))) | (df[col] > (q3 + (k * iqr))))]
  return df

has_outliers = ["depth","x","y","z"]
diamonds_cleaned = remove_outliers(diamonds,has_outliers)
diamonds_cleaned.head()
# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(diamonds_cleaned,test_size=0.2,random_state=42)
# %%
diamonds = train_set.copy()
print(diamonds.shape)
# %% [markdown]
# # Explore the data
# %%
diamonds["cut"].value_counts().plot(kind="bar")
plt.show()
# %%
try:
  corr_matrix = diamonds.corr(numeric_only=True)
except:
  corr_matrix = diamonds.corr()
corr_matrix
# %%
corr_matrix["price"].sort_values(ascending=False)
# %%
sns.scatterplot(data=diamonds,x="price",y="carat")
plt.show()
# %%
attribs = ["price","carat","x","y","z","table","depth"]
pd.plotting.scatter_matrix(diamonds[attribs],figsize=(12,8),diagonal="hist")
plt.show()
# %%
diamonds = train_set.drop("price",axis=1,inplace=False)
diamonds_labels = train_set["price"].copy().to_numpy()
diamonds.head()
# %%
diamonds_num = diamonds.loc[:,["carat","depth","table","x","y","z"]]
diamonds_cat = diamonds.loc[:,["cut","color","clarity"]]
# %%
from sklearn.preprocessing import OrdinalEncoder

ordinal_categories = [
  ["Fair","Good","Very Good","Premium","Ideal"],
  ["J","I","H","G","F","E","D"],
  ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
]

ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
diamonds_cat_encoded = ordinal_encoder.fit_transform(diamonds_cat)
diamonds_cat_encoded[:3]
# %%
ordinal_encoder.categories_
# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add a custom pipeline here:
num_pipeline = Pipeline([
  ("std_scaler",StandardScaler())
])
# We can add a cat_pipeline later!
num_pipeline.fit_transform(diamonds_num).shape
# %%
from sklearn.compose import ColumnTransformer

num_attribs = list(diamonds_num.columns)
cat_attribs = list(diamonds_cat.columns)

full_pipeline = ColumnTransformer([
  ("num",num_pipeline,num_attribs),
  ("cat",ordinal_encoder,cat_attribs)
])

diamonds_prepared = full_pipeline.fit_transform(diamonds)
# %%
diamonds_prepared.shape
# %% [markdown]
# # Ready to apply ML models
# Now we are ready to apply ML algorithms, specifically regression models.
# Models: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR
# %%
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(diamonds_prepared,diamonds_labels)
# %%
def print_rmse(model,X_train,y_train):
  y_pred = model.predict(X_train)
  rmse = np.sqrt(mean_squared_error(y_train,y_pred))
  print(f"RMSE: {round(rmse,2)}")
# %%
print_rmse(lin_reg,diamonds_prepared,diamonds_labels)
# %%
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(diamonds_prepared,diamonds_labels)
# %%
print_rmse(tree_reg,diamonds_prepared,diamonds_labels)
# %%
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(diamonds_prepared,diamonds_labels)
# %%
print_rmse(forest_reg,diamonds_prepared,diamonds_labels)
# %%
from sklearn.svm import SVR

svr_reg = SVR()
svr_reg.fit(diamonds_prepared,diamonds_labels)
# %%
print_rmse(svr_reg,diamonds_prepared,diamonds_labels)
# %% [markdown]
# # Cross Validation
# So far we have only seen how well does our models bias, linear regression shows a high bias, while the decision tree regressor shows a low bias indicating a high possibility of overfitting the training data.
# Now let us do a cross validation to get an idea of overfitting and see how good are models are the generalize to new data.
# %%
from sklearn.model_selection import GridSearchCV

# Can be a list of dictionaries.
param_grid = {
  "n_estimators": [100,150,200], 
  "max_features": [2,4,6,8]
}

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg,param_grid,scoring="neg_mean_squared_error",return_train_score=True,cv=5)
grid_search.fit(diamonds_prepared,diamonds_labels)
# %%
forest_best_estimator = grid_search.best_estimator_
forest_best_estimator
# %%
forest_best_params = grid_search.best_params_
# %%
# TODO: Add a way to sort for minimum rmse score.
def print_cv_scores(grid_search):
  cv_scores = zip(grid_search.cv_results_["params"], grid_search.cv_results_["mean_train_score"])
  for param, score in cv_scores:
    print(param, np.sqrt(-score))
# %%
print_cv_scores(grid_search)
# %%
# Only in Ensemble Learning Methods
feature_importances = forest_best_estimator.feature_importances_
feature_importances
# %%
# Function to print the feature importances:
attributes = num_attribs + cat_attribs
sorted(zip(feature_importances,attributes),reverse=True)
# %%
param_grid = {
  "kernel": ["linear","poly"], 
  "gamma": ["auto"]
}

svr = SVR()
grid_search = GridSearchCV(svr,param_grid,scoring="neg_mean_squared_error",return_train_score=True,cv=5)
grid_search.fit(diamonds_prepared,diamonds_labels)
# %%
svr_best_estimator = grid_search.best_estimator_
svr_best_estimator
# %%
svr_best_params = grid_search.best_params_
# %%
print_cv_scores(grid_search)
# %%
def indices_of_top_k(features,k):
  return np.sort(np.argpartition(features,-k)[-k:])

from sklearn.base import BaseEstimator,TransformerMixin

class TopFeatureSelector(BaseEstimator,TransformerMixin):
  def __init__(self,feature_importances,k):
    self.feature_importances = feature_importances
    self.k = k
  def fit(self,X,y=None):
    self.top_feature_indices = indices_of_top_k(self.feature_importances,self.k)
    return self
  def transform(self,X):
    return X[:,self.top_feature_indices]
# %%
k = 3
top_features = indices_of_top_k(feature_importances,k)
np.array(attributes)[top_features]
# %%
preparation_and_top_features = Pipeline([
  ("preparation",full_pipeline),
  ("feature_selection",TopFeatureSelector(feature_importances,k))
])
diamonds_prepared_with_top_features = preparation_and_top_features.fit_transform(diamonds)
# %%
diamonds_prepared_with_top_features[:2]
# %%
diamonds_prepared[:2,top_features]
# %%
X_test = test_set.drop("price",axis=1)
y_test = test_set["price"]
# %%
X_test_prepared = full_pipeline.transform(X_test)
# %%
forest_pred = forest_best_estimator.predict(X_test_prepared)
forest_mse = mean_squared_error(y_test,forest_pred)
np.sqrt(forest_mse)
# %%
svr_pred = svr_best_estimator.predict(X_test_prepared)
svr_mse = mean_squared_error(y_test,svr_pred)
np.sqrt(svr_mse)
# %% [markdown]
# # Complete Pipeline
# %%
forest_reg = RandomForestRegressor(random_state=42)

forest_pipeline = Pipeline([
  ("preparation", preparation_and_top_features),
  ("prediction",forest_reg)
])

# Regression problem: max_features: n_features/3
# n_features = 9, max_features = 3 (ISLR)

param_grid = {
  "preparation__feature_selection__k": [4,6,8],
  "prediction__n_estimators": [200,250,300],
  "prediction__max_features": [2,4,6]
}

forest_grid = GridSearchCV(forest_pipeline,param_grid,scoring="neg_mean_squared_error",return_train_score=True,cv=5)
# %%
forest_grid.fit(diamonds,diamonds_labels)
# %%
forest_grid.best_params_
# %%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

lin_reg_pipe = Pipeline([
  ("preparation", preparation_and_top_features),
  ("preprocessing", PolynomialFeatures(include_bias=False)),
  ("prediction", LinearRegression())
])

param_grid = {
  "preparation__feature_selection__k": [4,6,8],
  "preprocessing__degree": [1,2,3,4],
}

lin_grid = GridSearchCV(lin_reg_pipe, param_grid, scoring="neg_mean_squared_error", return_train_score=True, cv=5)
# %%
lin_grid.fit(diamonds,diamonds_labels)
# %%
lin_grid.best_params_
# %%
model_names = ["RandomForest", "LinearReg"]
grids = [forest_grid, lin_grid]
scores = []

for (model,grid) in zip(model_names,grids):
  score = grid.best_score_
  print(f"{model} Score: {score:.2f}")
  scores.append(score)

(val, idx) = max((val,idx) for (idx,val) in enumerate(scores))
best_model = grids[idx]
best_model
# %% [markdown]
# 95% confidence interval for the test RMSE:
# %%
y_pred = best_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# %%
from scipy import stats

def rmse_interval(y_true, y_pred, confidence=0.95):
  squared_errors = (y_pred - y_true) ** 2
  final_rmse = np.sqrt(stats.t.interval(confidence,len(squared_errors)-1,loc=mean_squared_error(y_true, y_pred),scale=stats.sem(squared_errors)))
  return tuple(final_rmse)
# %%
rmse_rng = rmse_interval(y_test, y_pred)
print("The RandomForest model can expect a RMSE from {0:.2f} to {1:.2f}!".format(rmse_rng[0], rmse_rng[1]))