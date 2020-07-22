# Car-Price-Prediction

The dataset of this mini project is collected from kaggle. The problem statement is to estimate or predict the selling price of a used car. Cardheko website has shared this data in kaggle to predict the selling price.

This car dataset contains of 9 features namely Car Name, Year of purchase, Selling Price(dependent feature), Present Price, Kms Driven, Fuel Type, Transmission, and No of Owners. And also this is small data set with 300 observations. Feature engineering has done like substracting year of purchase with present year to find how old is that car. This feature is very important to determine selling price. 

Since Car Name feature has many objects i.e., unique values, ingoring this feature will not effect much for our prediction. All other  can be consider to get a good predicted ML model. One hot encoding technique is used to deal with categorical variables by creating dummies and take caution not to fall in dummy variable trap.

RandomForestRegressor has given good accuracy and these parameters are tunned with RandomizedSearchCV because this is fast compared to other parameter tunning techniques. After building this model it is imported into pickle file.

A web api was built using flask frame work and html, css for web page design. The model was deployed into heroku cloud. Heroku is PaaS(Platform as a service) cloud where one can deploy their models into use easily.
