# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:48:54 2020

@author: neelabh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Reading the csv file
raw_df = pd.read_csv("Iris.csv")
new_df = raw_df.drop('Id', axis=1)
print(new_df.Species.unique())
# Assigning numerical values to the "Species" coloumn
h = new_df['Species'].str.get_dummies("EOL")
new_df = new_df.merge(h, left_index=True, right_index=True)
#new_df['Species'] = new_df['Species'].map({'Iris-setosa':1, 'Iris-versicolor':0, 'Iris-virginica':-1})

# Splitting the dataset in train and test sets

def setosa():# Classifier: Iris-setosa
    train_x, test_x, train_y, test_y = train_test_split(new_df.drop(['Species','Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],axis=1), new_df['Iris-setosa'], test_size=0.2, random_state=2)
    train_x = train_x.T
    test_x = test_x.T
    # To prevent dimensions of the form (m,)
    # instead we want of the form (m,n)
    train_y = pd.DataFrame(train_y).T
    test_y = pd.DataFrame(test_y).T
    # To obtain array
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    return train_x, test_x, train_y, test_y

def versicolor():# Classifier: Iris-versicolor
    train_x, test_x, train_y, test_y = train_test_split(new_df.drop(['Species','Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],axis=1), new_df['Iris-versicolor'], test_size=0.2, random_state=2)
    train_x = train_x.T
    test_x = test_x.T
    # To prevent dimensions of the form (m,)
    # instead we want of the form (m,n)
    train_y = pd.DataFrame(train_y).T
    test_y = pd.DataFrame(test_y).T
    # To obtain array
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    return train_x, test_x, train_y, test_y

def virginica():# Classifier: Iris-virginica
    train_x, test_x, train_y, test_y = train_test_split(new_df.drop(['Species','Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],axis=1), new_df['Iris-virginica'], test_size=0.2, random_state=2)
    train_x = train_x.T
    test_x = test_x.T
    # To prevent dimensions of the form (m,)
    # instead we want of the form (m,n)
    train_y = pd.DataFrame(train_y).T
    test_y = pd.DataFrame(test_y).T
    # To obtain array
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    return train_x, test_x, train_y, test_y


# Implementing single layer logistic regression
def initialize_parameters(df_dims):
    W = np.zeros((df_dims,1))
    b = 0
    
    assert(W.shape == (df_dims, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return W, b
def activation(z):
    a = 1/(1+np.exp(-z))
    return a
def propagate(W, b, X, Y):
    m = X.shape[1]
    
    # Forward propagation
    z = np.dot(W.T,X)+b
    #print(W.T.shape,X.shape)
    a = activation(z)
    cost = (-1/m)*np.sum(Y*np.log(a)+(1-Y)*np.log(1-a))
    #cost = (-1/m)*np.sum(((Y+1)/2)*np.log((a+1)/2)+(1-(Y+1)/2)*np.log(1-(a+1)/2))
    #print(cost)
    '''cost = (-1/m)*(((Y+1)/2)*np.log((a+1)/2)+(1-(Y+1)/2)*np.log(1-(a+1)/2))
    g_dash = 1-np.power(a,2)
    dZ = g_dash
    dw = np.dot(dZ,X.T)
    db = np.sum(dZ,axis = 1,keepdims = True)'''
    
    # Backward propagation
    dw = (1/m)*np.dot(X,(a-Y).T)
    db = (1/m)*np.sum(a-Y)
    #dw = ((-a*Y+1)*(1-(np.tanh(z)**2))*(X))/(np.log(10)*(a+1)(a-1))
    #db = ((-a*Y+1)*(1-(np.tanh(z)**2))*(1))/(np.log(10)*(a+1)(a-1))
    
    grads = {"dw": dw,
             "db": db}
    
    
    assert(dw.shape == W.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return grads, cost

def optimize(W, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(W, b , X, Y)
        dw = grads['dw']
        db = grads['db']
        
        W = W - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": W,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, cost

def predict(W, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    W = W.reshape(X.shape[0], 1)

    A = activation(np.dot(W.T,X)+b)
    
    for i in range(A.shape[0]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
    
        if (A[0,i] < 0.5):
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    assert(Y_prediction.shape == (1, m))
    return Y_prediction

def model(train_x, train_y, test_x, test_y, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    w, b = initialize_parameters(train_x.shape[0])
    parameters, grads, costs = optimize(w, b, train_x, train_y, num_iterations, learning_rate, print_cost = False)
    Y_pred_train = predict(w, b, train_x)
    Y_pred_test = predict(w, b, test_x)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - train_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - test_y)) * 100))

    
    d = {"Y_prediction_test": Y_pred_test, 
         "Y_prediction_train" : Y_pred_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

iris_list = [setosa(), versicolor(), virginica()]
iris_list_names = [setosa, versicolor, virginica]
for flower_type in range(len(iris_list)):
    print('\nIris-',iris_list_names[flower_type])
    train_x, test_x, train_y, test_y = iris_list[flower_type]
    d = model(train_x, train_y, test_x, test_y,  learning_rate = 0.005, print_cost = True)
    
