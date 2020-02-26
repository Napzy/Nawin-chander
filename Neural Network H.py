import pandas as pd

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv('iris.data', names=names)  


irisdata.head()  


# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]

# Assign data from first fifth columns to y variable
y = irisdata.select_dtypes(include=[object])  


y.head()  

y.Class.unique()  



from sklearn import preprocessing  
le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)


y.Class.unique()  



from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 92)  



from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 


from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(18, 18, 18), max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel())  


predictions = mlp.predict(X_test)  





from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, predictions)
print(cm)

accuracy = accuracy_score(y_test, predictions)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')