import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
#import the dataset and load it into pandas dataframe
# Assign colum names to the dataset
# Read dataset to pandas dataframe
dataset = pd.read_csv('Wine.csv')  
#Preview the first five observations

dataset.head()  


#Preprocessing
#split the dataset into its attributes and labels.
#Stores all observations and columns into X except the last column
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, -1].values  
y= y.astype(int) 

#splits the dataset into 80% train data and 20% test data.
#out of total 150 records, the training set will contain 120 records 
#and the test set contains 30 of those records.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10 , random_state = 1)

#Feature Scaling
#MinMaxScaler - Applies minmax normalisation
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

X_train.shape
X_test.shape

# Fitting Decision tree classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
 

# Fitting Decision tree classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=1 ,max_depth=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)


from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
 