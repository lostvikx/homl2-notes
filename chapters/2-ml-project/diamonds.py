# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# # * x: length in mm
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
    print(df.shape)
  return df

has_outliers = ["depth","x","y","z"]
diamonds_cleaned = remove_outliers(diamonds,has_outliers)
diamonds_cleaned.head()
# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(diamonds_cleaned,test_size=0.2,random_state=42)
# %%
