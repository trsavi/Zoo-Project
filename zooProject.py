# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:26:53 2020

@author: Vukašin Vasiljević
"""

# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# =============================================================================
# PART I - Importing the dataset/ Splitting dataset/ Feature Sscaling
# =============================================================================

# import the dataset
dataset = pd.read_csv('zoo.csv')

    
x = dataset.iloc[:,1:17].values # independent variable vector
y = dataset.iloc[:,17].values # dependent variable vector

y = keras.utils.to_categorical(y, num_classes=None, dtype='float32')

y = y[:,1:]

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

scalers = [("MinMaxScaler", MinMaxScaler()),("StandardScaler", StandardScaler()), 
           ("RobustScaler",RobustScaler())]
layers = [8, 32]
epochs = [50, 150]

# =============================================================================
# Part II - Splitting data into test and train set/ Iinitializing the ANN
# =============================================================================

# Splitting the dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)

def scaleModel(index, layer, epoch, hidden=True):
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)
    sc = scalers[index][1]
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    
    # Initializing the Artificial Neuron Network    
    classifier = Sequential()
    
    # Adding the input layer and first hidden layer   
    classifier.add(Dense(units=layer, activation='relu', input_dim=16, kernel_initializer='uniform' ))
    
    if(hidden):
        # Adding the second hidden layer    
        classifier.add(Dense(units=layer, activation='relu', kernel_initializer='uniform'))
   
    
    # Adding the otput layer    
    classifier.add(Dense(units=7, activation='softmax', kernel_initializer='uniform'))
    
    # Compliling the ANN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fitting the ANN to the training set    
    classifier.fit(x_train,y_train, batch_size=32, epochs=epoch)

    return classifier.predict(x_test)


# Variable to store 
values = []

# Iterate through each scaler, layer and epoch for calculating four parameters
for i in range(3):
    for layer in layers:
        for epoch in epochs:
            
                # Making the predictions and evaluating the model            
                y_pred = scaleModel(i, layer, epoch) # set hidden to false in order to exclude the second hidden layer
                y_pred = (y_pred > 0.5) 
                # accuracy: (tp + tn) / (p + n)
                accuracy = accuracy_score(y_test, y_pred)
                # print('Accuracy: %f' % accuracy)
                # precision tp / (tp + fp)
                precision = precision_score(y_test, y_pred, average='micro')
                # print('Precision: %f' % precision)
                # recall: tp / (tp + fn)
                recall = recall_score(y_test, y_pred,average='micro')
                # print('Recall: %f' % recall)
                # f1: 2 tp / (2 tp + fp + fn)
                f1 = f1_score(y_test, y_pred, average='micro')
                # print('F1 score: %f' % f1)
                values.append((scalers[i][0], layer, epoch, accuracy, precision, recall, f1)) 
                


# ======================================================================================
# PART III - Visualizing the results/ Calculating accuracy, precision, recall, F1 score
# ======================================================================================
                
# Scalers with 8 input layers (50,150) epochs
minMax_8 = [value[3:] for value in values[:2]]
standard_8 = [value[3:] for value in values[4:6]]
robust_8 = [value[3:] for value in values[8:10]]

# Scalers with 32 input layers (50, 150) epochs
minMax_32 = [value[3:] for value in values[2:4]]
standard_32 = [value[3:] for value in values[6:8]]
robust_32 = [value[3:] for value in values[10:12]]

# Using patches to define the handles for plot legend
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Robust Scaler')
blue_patch = mpatches.Patch(color='blue', label='MinMax Scaler')
green_patch = mpatches.Patch(color='green', label='Standard Scaler')

# Plotting the calculated parameters with defined scaler

# Defining the x ticks to be parameters
x_ticks = ["accuracy","precision", "recall", "F1 score"]
x = [1,2,3,4]


# Function that plots based on input layers in scalers and epochs
def plot_layers(minmax_s, standard_s, robust_s, title):
    # Defining the sublot
    ax = plt.subplot(111)
    w = 0.2
    ax.set_title(title)
    for i in range(4):
        ax.bar(x[i]-w, minmax_s[i], width=w, color='b', align='center')
        ax.bar(x[i], standard_s[i], width=w, color='g', align='center')
        ax.bar(x[i]+w, robust_s[i], width=w, color='r', align='center')
        plt.xticks(x, x_ticks)
        ax.autoscale(tight=True)
        ax.set_ylim((0,1.5))
    plt.legend(handles=[blue_patch,green_patch,red_patch],loc="upper right")
    plt.show()

# Scalers with 8 layers
title1 = "ANN with 8 layers and 50 epochs"
plot_layers(minMax_8[0], standard_8[0], robust_8[0], title1)

title2 = "ANN with 8 layers and 150 epochs"
plot_layers(minMax_8[1], standard_8[1], robust_8[1], title2)

# Scalers with 32 layers
title3 = "ANN with 32 layers and 50 epochs"
plot_layers(minMax_32[0], standard_32[0], robust_32[0], title3)
title4 = "ANN with 32 layers and 150 epochs"
plot_layers(minMax_32[1], standard_32[1], robust_32[1], title4)

    


# Making the confusion matrix
from sklearn.metrics import multilabel_confusion_matrix
cm = multilabel_confusion_matrix(y_test, y_pred)

# Visualizing the Training set results 
# =============================================================================
# 
# from ann_visualizer.visualize import ann_viz
# 
# ann_viz(classifier, title="Zoo Project")
# =============================================================================
