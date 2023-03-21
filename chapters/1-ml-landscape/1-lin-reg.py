#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
#%%
sns.set_context("paper",rc={"font.size":10,"axes.titlesize":12}) # plt.rcParams
sns.set_style("dark")
sns.set_palette(palette="muted")
#%%
oecd = pd.read_csv("../../data/oecd_life.csv").iloc[:,[1,14]]
oecd = oecd.set_index("Country")
print(oecd.shape)
oecd.head()
#%%
gdp = pd.read_csv("../../data/gdp.csv").iloc[:,[0,2]]
gdp = gdp[gdp["Country"].notna()]
gdp = gdp.set_index("Country")
gdp.head()
# %%
country_stats = pd.merge(oecd,gdp,left_index=True,right_index=True)
country_stats = country_stats.reset_index()
country_stats.columns = ["country","life_satisfaction","gdp_per_capita"]
country_stats.head()
# %%
X = np.c_[country_stats["gdp_per_capita"]]
y = np.c_[country_stats["life_satisfaction"]]
# %%
sns.scatterplot(data=country_stats,x="gdp_per_capita",y="life_satisfaction")
plt.show()
# %%
model = linear_model.LinearRegression()
# %%
model.fit(X,y)
# %%
gdp_india = [[gdp.loc["Cyprus"]["2018"]]]
print(f"GDP per Capita: {gdp_india}")
y_pred = model.predict(gdp_india) # out: [[5.81941053]]
print(f"Life Satisfaction: {y_pred}")
# %% [markdown]
# Conclusion: If everything goes well, our model will make good predictions. If not, we may require more attributes (such as employment rate, health, air pollution, etc.), to get more or better quality training data. Or we could use a more powerful model, e.g. Polynomial Regression.