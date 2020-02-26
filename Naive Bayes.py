# Naive Bayes Classification

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('wine.csv')

#looking at the first 5 values of the dataset
dataset.head()
dataset.columns
#Spliting the dataset in independent and dependent variables
X = dataset.iloc[:,:4].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size = 0.20, random_state = 82)

# Feature Scaling to bring the variable in a single scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes Classification to the Training set 
from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nvclassifier.predict(X_test)
print(y_pred)


from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
 


#BernoulliNB


# Fitting Naive Bayes Classification to the Training set 

from sklearn.naive_bayes import BernoulliNB
nvclassifier = BernoulliNB()
nvclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nvclassifier.predict(X_test)
print(y_pred)


from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
 