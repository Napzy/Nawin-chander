# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:24:01 2019

@author: Nawin chander
"""

#Machine language-wine data knn value
#k-nn values
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#to import the datset and to load its pandas data frame

dataset=pd.read_csv('wine.csv')
type(dataset)

dataset.dtypes

dataset.shape
dataset.columns
dataset.head()
dataset.Customer_Segment.unique()
dataset.info()

#preprocessing
#split the dataset into its attributes and labels
#stores all boservations and columns
#into x except the last columns

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#split the data into the training data and the test data
#out of the 150 outcomes

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=0)

X_train.shape
y_test.shape

#Feature scaling
#minmaxscalalr-applies minmax normalisation

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)

X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

X_train.shape
X_test.shape


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=12)
classifier.fit(X_train,y_train)

y_pred= classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print(cm)

accuracy = accuracy_score(y_test,y_pred)*100
print('Accuracy of our model is equal'+str(round(accuracy,2))+'%.')

error=[]

#calculating error for k values between 1 and 40

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error.append(np.mean(pred_i !=y_test))
    
plt.figure(figsize=(12,6))

plt.plot(range(1,40),error,color='brown',linestyle='dashed',marker='o',markerfacecolor='green',markersize=15)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

