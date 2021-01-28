#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv("car.csv")
df.head()


# In[3]:


df.shape


# In[4]:


print("Seller Type:",df['Seller_Type'].unique())
print("Transmission:",df['Transmission'].unique())
print("Owner",df['Owner'].unique())
print("Fuel Type",df['Fuel_Type'].unique())


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.dtypes


# In[9]:


df.info()


# In[10]:


df.columns


# In[11]:


final_df = df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]


# In[12]:


final_df.head()


# In[13]:


final_df['current_year'] = 2020


# In[14]:


final_df.head()


# In[15]:


final_df['Age of car (Years)'] = final_df['current_year']-final_df['Year']


# In[16]:


final_df.head()


# In[17]:


final_df.drop(['Year','current_year'],axis=1,inplace=True)


# In[18]:


final_df.head()


# In[19]:


final_df = pd.get_dummies(final_df,drop_first=True)


# In[20]:


final_df.head()


# In[21]:


final_df.corr()


# In[22]:


# Plotting the heatmap of correlation between features
plt.figure(figsize=(10,10))
sns.heatmap(final_df.corr(), cbar=False, square= True, fmt='.2%', annot=True, cmap='Greens')


# In[23]:


sns.pairplot(final_df)


# In[ ]:





# In[24]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(13,13))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[25]:


final_df.head()


# In[26]:


X = final_df.iloc[:,1:]
y = final_df.iloc[:,0]


# In[27]:


X.head()


# In[28]:


y.head()


# In[29]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)


# In[30]:


print(model.feature_importances_)


# In[31]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[51]:


from sklearn.ensemble import RandomForestRegressor


# In[52]:


regressor=RandomForestRegressor()


# In[53]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[54]:


from sklearn.model_selection import RandomizedSearchCV


# In[55]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[56]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[57]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[58]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[59]:


rf_random.fit(X_train,y_train)


# In[60]:


rf_random.best_params_


# In[61]:


rf_random.best_score_


# In[62]:


predictions=rf_random.predict(X_test)


# In[63]:


predictions


# In[45]:


sns.distplot(y_test-predictions)


# In[46]:


plt.scatter(y_test,predictions)


# In[48]:


from sklearn import metrics


# In[64]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[65]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




