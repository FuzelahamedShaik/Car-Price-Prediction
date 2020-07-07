import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("car data.csv")
data.head()
data.columns
data.shape
data.dtypes
data["Car_Name"].unique()
data["Fuel_Type"].unique()
data["Seller_Type"].unique()
data["Transmission"].unique()
data["Owner"].unique()

#Check Missing Value
data.isnull().sum()

data.describe(include='all')

#Handling Categorical Features
cat_data = data[["Fuel_Type","Seller_Type","Transmission"]]
cat_data.head()
dummy_cat_data = pd.get_dummies(cat_data,drop_first=True)
dummy_cat_data.shape
dummy_cat_data.columns

#Feature Engineering
num_data = data[["Year","Selling_Price","Present_Price","Kms_Driven","Owner"]]
num_data.head()
num_data["PresentYear"]=2020
num_data["No_of_Years"] = num_data["PresentYear"] - num_data["Year"]
num_data.head()
updated_num_data = num_data.drop(["Year","PresentYear"],axis=1)
updated_num_data.head()

#Merging dataframes
final_data = pd.concat([updated_num_data,dummy_cat_data], axis=1, sort=False)
final_data.shape
final_data.columns

#Seeing Correlation
final_data.corr()
import seaborn as sns
sns.pairplot(final_data)

corrmat=final_data.corr()
top_corr_features=corrmat.index 
plt.figure(figsize=(20,20)) 
#plot heat map 
g=sns.heatmap(final_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Dependent and Independent features
X = final_data.iloc[:,1:]
y = final_data.iloc[:,0]
X.columns

#Which are important features
from sklearn.ensemble import ExtraTreesRegressor 
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)

#visualize Feature Importance
feature_imp = pd.Series(model.feature_importances_,index=X.columns)
feature_imp.nlargest(5).plot(kind="barh")
plt.show()

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()

#Hyperparameter tunning using RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_features = ["auto","sqrt"]
max_depth = [int(x) for x in np.linspace(5,30,num=5)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]

from sklearn.model_selection import RandomizedSearchCV
random_grid={
        "n_estimators":n_estimators,
        "max_features":max_features,
        "max_depth":max_depth,
        "min_samples_split":min_samples_split,
        "min_samples_leaf":min_samples_leaf}
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=10,scoring="neg_mean_squared_error",cv=5,verbose=2,random_state=42,n_jobs=1)
rf_random.fit(X_train,y_train)

#Predictions
pred = rf_random.predict(X_test)
sns.distplot(y_test-pred)
plt.scatter(y_test,pred)

#MSE
np.sum((y_test-pred)**2)/len(y_test)

#Creating a Pickle file
import pickle
file = open("random_forest_regressor_car_prediction_model.pkl","wb")
pickle.dump(rf_random,file)

import matplotlib
matplotlib.__version__
print(pd.__version__)
np.version.version








