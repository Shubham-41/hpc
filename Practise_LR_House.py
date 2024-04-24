
# Linear regression by using Deep Neural network: 
# Implement Boston housing price prediction problem 
# by Linearregression using Deep Neural network. 
# Use Boston House price prediction dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data = pd.read_csv("boston.csv")
data = pd.read_csv("1_boston_housing_keggle.csv")


data.head()


data.tail()


data.describe()

data.columns

data.shape


data.isnull().sum()


data.info()

import seaborn as sns


sns.histplot(data['MEDV'])

sns.boxplot(data['MEDV'])


from sklearn.preprocessing import StandardScaler 

X = data.drop('MEDV',axis=1)

Y = data['MEDV']

# Scale the input features

scaler = StandardScaler()

X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# split the data into training and testing sets

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,
                                                 random_state=42)

print('Training set shape:', X_train.shape,Y_train.shape)

print('Testing Set Shape',X_test.shape,Y_test.shape)

from keras.models import Sequential

from keras.layers import Dense, Dropout



# Define the model architecture

model = Sequential()

model.add(Dense(128, activation='relu', input_dim = 13))

model.add(Dense(64, activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(16,activation='relu'))

# Output Layer
# For regression tasks, use a single neuron without activation function

model.add(Dense(1))


#  Display the model summary

print(model.summary())


# Compile the model

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])

X_val = X_train

Y_val = Y_train


history = model.fit(X_train,
                   Y_train,
                   epochs=15,
                   batch_size=512,
                   validation_data=(X_val,Y_val))


results = model.evaluate(X_test,Y_test)

y_pred = model.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# conf_matrix = confusion_matrix(Y_test,y_pred)

# print(conf_matrix)


# The error "continuous is not supported" typically occurs 
# when you're trying to compute a confusion matrix for a regression
# problem instead of a classification problem. Since your task is a
# regression problem (predicting house prices), 
# calculating a confusion matrix 
# doesn't make sense because confusion matrices are for
# classification problems, 
# not regression.


# Compute mean absolute error

mae = mean_absolute_error(Y_test,y_pred)
print("Mean absolute error (MAE):", mae)

# Compute mean squared error

mse = mean_squared_error(Y_test,y_pred)
print("Mean Squared error (MSE):",mse)

# Compute R-squared (R2) score

r2 = r2_score(Y_test,y_pred)
print("R-squared (R2) Score:",r2)


# new_model = Sequential()

# new_model.add(Dense(256, activation='relu', input_dim=13))  
# Increase the number of neurons in the first layer
# new_model.add(Dropout(0.5))  # Increase dropout rate to 50%

# new_model.add(Dense(128, activation='relu'))  
# Add additional hidden layers
# new_model.add(Dropout(0.3))  # Add dropout after each hidden layer
# new_model.add(Dense(64, activation='relu'))
# new_model.add(Dropout(0.3))
# new_model.add(Dense(32, activation='relu'))
# new_model.add(Dropout(0.3))
# new_model.add(Dense(16, activation='relu'))

# new_model.add(Dense(1))


# Mean Absolute Error (MAE): 13.776206224572427
# Mean Squared Error (MSE): 249.77224408948888
# R-squared (R2) Score: -2.4059642544751116

# Model 2 performs better than Model 1 in terms of MAE and MSE,
# as it has lower values for both metrics.
# However, both models have negative R-squared (R2) scores,
# which suggests poor fit to the data. A negative R-squared score 
# indicates that the model performs worse than a horizontal line 
# (the mean of the target variable).
# Considering the negative R-squared scores, 
# it appears that neither model provides a good fit to the data.
# In this case, you might need to revisit the model architecture, features,
# or data preprocessing steps to improve model performance. Additionally, 
# cross-validation and further experimentation could help 
# in selecting a better model.


