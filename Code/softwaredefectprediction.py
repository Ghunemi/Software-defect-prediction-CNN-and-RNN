# -*- coding: utf-8 -*-
"""softwareDefectPrediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KvuP_zmdPlsXnu6IgJE0pW82uzqpQ7Vh
"""

''' 
Accessing google drive to get dataset and unzip
'''

from google.colab import drive
drive.mount('/content/drive/')

!cp 'drive/My Drive/preprocessing/featuress.csv' /content

!cp 'drive/My Drive/preprocessing/labelss.csv' /content

import pandas as pd
from pandas import DataFrame
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, Embedding, LSTM, Conv1D, MaxPool1D, GRU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from numpy import savetxt
import sklearn as sk
from keras import backend as K

files = pd.read_csv("featuress.csv").iloc[ 0:10019, 0:1200]
labels = pd.read_csv("labelss.csv").iloc[ : , 1]

RNN = pd.read_csv("featuress.csv").iloc[ 0:10019, 0:500]
RNNS = pd.read_csv("labelss.csv").iloc[ 0:10019 , 1]
RNN = sk.preprocessing.scale(RNN)
RNN = np.array(RNN)
RNN = RNN.astype('float32')
RNN_TrainFiles, RNN_TestFiles, RNN_TrainLabels , RNN_testLabels = train_test_split(RNN , RNNS, test_size=0.3)

files = sk.preprocessing.scale(files)

files = np.array(files)
labels = np.array(labels)
files = files.astype('float32')


'''
    Dividing the data into train/test split
'''

train_Files, test_Files, train_labels , test_labels = train_test_split(files , labels, test_size=0.3)
#RNN_TrainFiles = train_Files
#RNN_TestFiles = test_Files
#RNN_TrainLabels = train_labels
#RNN_testLabels = test_labels

print("Train shape" , train_Files.shape)
print("labels Shape", train_labels.shape)

train_Files = train_Files.reshape((7013,1200,1,1))
test_Files = test_Files.reshape((3006,1200,1,1))

train_Files2 = train_Files.reshape((7013,1200,1))
test_Files2 = test_Files.reshape((3006,1200,1))

RNN_TrainFiles = RNN_TrainFiles.reshape((7013,500,1)) #1200
RNN_TestFiles = RNN_TestFiles.reshape((3006,500,1)) #1200

train_labels = np.asarray(train_labels).astype('float32').reshape((-1,1))
test_labels = np.asarray(test_labels).astype('float32').reshape((-1,1))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




def softwareDefectCNN1D(): 
   

  '''
      CNN model with 1D convolutional layers
  '''

  classifier = Sequential()
  

  classifier.add(Conv1D(96, 1, input_shape=(1200,1), activation='relu'))
  classifier.add(MaxPool1D(pool_size= 1, strides=2))
  #classifier.add(BatchNormalization())

  classifier.add(Conv1D(256, 1, activation='relu'))
  classifier.add(MaxPool1D(pool_size= 1, strides=2))
 # classifier.add(BatchNormalization())

  classifier.add(Conv1D(384, 1, activation='relu'))
  classifier.add(MaxPool1D(pool_size= 1, strides=2))
  #classifier.add(BatchNormalization())

  #classifier.add(Conv2D(384, 1, activation='relu'))
  #classifier.add(MaxPooling2D(pool_size=(1, 1), strides=2))
  #classifier.add(BatchNormalization())

  classifier.add(Conv1D(256, 1, activation='relu'))

  classifier.add(Flatten())

  classifier.add(Dense(1024, activation='relu'))
  classifier.add(Dropout(0.6))

  classifier.add(Dense(512, activation='relu'))
  classifier.add(Dropout(0.6))
  

  classifier.add(Dense(64, activation='relu'))
  classifier.add(Dropout(0.6))

  classifier.add(Dense(64, activation='relu'))
  classifier.add(Dropout(0.6))

  classifier.add(Dense(16, activation='relu'))
  classifier.add(Dropout(0.6))

  classifier.add(Dense(1, activation='sigmoid'))
  
  return classifier



  '''
    Compilation and training of the model with 
    Epochs: 100
    Optimizer: Adam
    learning rate: 0.0001
    batch size: 16
    loss function: binary crossentropy
    Accuracy: 88%
'''
 

softwareDefectCNN1D = softwareDefectCNN1D()
softwareDefectCNN1D.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=["accuracy",f1_m])
hist = softwareDefectCNN1D.fit(train_Files2, train_labels, batch_size=16, epochs=100, verbose=1 ,validation_data=(test_Files2,test_labels), shuffle=True)

from sklearn.metrics import confusion_matrix

CNN_Predict = softwareDefectCNN1D.predict(test_Files)
rounded = [round(x[0]) for x in CNN_Predict]
conf_matrix = confusion_matrix(test_labels,rounded)
print("Confustion matrix: ")
print(conf_matrix)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def softwareDefectRNN():


  classifier = Sequential()
#512
  classifier.add(LSTM(512,input_shape=(500,1), recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True))# recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True

  #classifier.add(LSTM(512, recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True))

  classifier.add(Flatten())

  #classifier.add(Dense(128, activation='softplus'))

  #classifier.add(Dense(64, activation='relu'))

  #classifier.add(Dense(32, activation='sigmoid'))
  
  #classifier.add(Dense(16, activation='sigmoid'))

  classifier.add(Dense(1, activation='sigmoid'))

  return classifier


softwareDefectRNN = softwareDefectRNN()
softwareDefectRNN.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001), metrics=["accuracy"])
hist =softwareDefectRNN.fit(RNN_TrainFiles, RNN_TrainLabels, batch_size=4, epochs=15, verbose=1 ,validation_data=(RNN_TestFiles,RNN_testLabels), shuffle=True)

from sklearn.metrics import confusion_matrix

RNN_Predict = softwareDefectRNN.predict(RNN_TestFiles)
rounded = [round(x[0]) for x in RNN_Predict]
conf_matrix = confusion_matrix(test_labels,rounded)
print("Confustion matrix: ")
print(conf_matrix)