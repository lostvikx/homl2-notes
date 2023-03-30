# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
sns.set_theme(context="notebook",style="darkgrid",palette="muted")
#%%
diamonds = pd.read_csv("data/diamonds.csv").iloc[:,1:]
diamonds.head()
# %% [markdown]
# ## Understanding the features given:
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
# We can clearly see a few outliers in our dataset.
#
# We are going to use Tukey's method to get rid of outliers in our data.
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
# Explore the data:
# %%
diamonds["cut"].value_counts().plot(kind="bar")
plt.show()
# %%
corr_matrix = diamonds.corr(numeric_only=True)
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
# Now we are ready to apply ML algorithms, specifically regression models.
#
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
# So far we have only seen how well does our models bias, linear regression shows a high bias, while the decision tree regressor shows a low bias indicating a high possibility of overfitting the training data.
#
# Now let us do a cross validation to get an idea of overfitting and see how good are models are the generalize to new data.