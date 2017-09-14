# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:43:06 2017

@author: Thomas
"""
# Objective
# The objective is to analyze the size of the sepal length, sepal width,
# petal length, petal-width to classify the type of flower using neural networks.

# Steps:
# Today you are going to:
# 1. Import a datasheet 
# 2. Randomize the data.
# 3. Split the data into a Training Portion
# 4. Spilt the data into a Validation Portion
# 5. Split the data into a Testing Portion


import numpy as np # Your spreadsheet manipulator tools.
import pandas as pd #Your Spreadsheet importer tools.
from sklearn.neural_network import MLPClassifier #Your neural network tools.
import matplotlib.pyplot as plt #To plot graphs and items.
from pandas.tools.plotting import scatter_matrix #To Make a scatter matrix
from sklearn.model_selection import train_test_split # This will help as randomize the data
from sklearn.metrics import accuracy_score #The way to determine how accurate we are.
from sklearn.metrics import confusion_matrix #The Matrix to show the what was correctly predicted and what wasn't.
#In the spyder program installed with anaconda, the following libraries should be included.

def preprocessEColi(pdDataFile):
    #Text to change, changing the mlp classes to numbers, so the neural net can use them.    
    pdDataFile['Species'].replace(
            to_replace = ['setosa'],
            value = 1, 
            inplace = True
    )
    pdDataFile['Species'].replace(
            to_replace = ['virginica'], 
            value = 2,
            inplace = True
    )
    pdDataFile['Species'].replace(
            to_replace = ['versicolor'],
            value = 3, 
            inplace = True
    )
    
    return pdDataFile

    

#You need to download anaconda and use spyder GUI to do this...
#https://www.continuum.io/downloads

#So first you need to get the download for the iris dataset.
#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

#Or you can use the auto downloader

# PART 1
# The names of each of the columns.
names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
# The url
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
fileName = 'datasets/datasets/iris.csv'
# This is the dataset from UC Irvine's Data Repository
dataset = pd.read_csv(fileName);
#Put the same Iris file in the same folder as this file.

#DATA IS NOW LOADED...
#Pre process the data, so classes aren't words but numbers.
dataset = preprocessEColi(dataset) # Look at the function above.
print("Here's the dataset")
print(dataset)
# Looking at the data shape
print(dataset.shape)
print("\n")
#print
#print ("\n")
#print(dataset.head(20))

#Let's check out the data.
print ("\nLet's get a statisitical summary")
print(dataset.describe())

print ("\nLet's get a better look at this data (class size)")
print("1 stands for 'Iris-setosa'\n")
print("2 stands for 'Iris-virginica'\n")
print("3 stands for 'Iris-versicolor'\n")

print(dataset.groupby('Species').size())


#Part 2 We need to split the data...
#Time to split our data set 80% to train, 20% for validation.
#Begin spliting out the validation set
array = dataset.values;
#Copies ['sepal-length', 'sepal-width', 'petal-length', 'petal-width'] into X 
X = array[:,1:5] #copies array[0],array[1],array[2],array[3]
#Gets the classifiers for output array[4] = ['class']
Y = array[:,5]  #copies array[4]

#Print y values. To see if they are string.
print("\nY Values!\n")
print(Y)

#Keeping 20% of the data for validation testing.
validation_size = 0.20
# This is how to randomize the numbers..
seed = 7


# STEP 3, 4 AND 5
# Step 3,4 and 5
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = validation_size, random_state = seed)


# BUILD THE CLASSIFIER!
theClassifier = MLPClassifier(hidden_layer_sizes=(10,),solver='sgd',learning_rate_init=0.01,max_iter=500)
theClassifier.fit(X_train, Y_train)

# PREDICT USING YOUR MODEL!
Y_pred = theClassifier.predict(X_test)

# HOW ACCURATE IS YOUR MODEL?
print("Accuracy Score: " + str(accuracy_score(Y_test, Y_pred))) #This let's you know.
print("Confusion Matrix:")
# Confusion Matrix, Basically it works like this.
print(confusion_matrix(Y_test, Y_pred))
# Lets assume that your MLP is going to identify based off several features (size, color type... etc)
# If the item is either Apples or oranges.
#                         Predicted Apple         Predicted Orange
# Actually Apples              4                          1
# Actually Orange              2                          5

# The number 4 states that We Predicted 4 apples that were actually apples.
# The number 1 states that We Predicted 1 Orange that was actually an apple.
# The number 2 states that We predicted 2 Apples that were actually oranges.
# The number 5 states that We predicted 5 Oranges that were actually oranges.
