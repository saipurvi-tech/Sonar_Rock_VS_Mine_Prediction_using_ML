import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

sonardata = pd.read_csv('SonarData.csv', header=None) 
#print(sonardata.head())
#print(sonardata.shape) 
#print(sonardata.describe()) 
#we are checking howmany mines are there and how many rocks are there, by retriving only the 60th column as it contains the details about this. 
#print(sonardata[60].value_counts())
#print(sonardata.groupby(60).mean)

#seperatinf the data and the labels
X = sonardata.drop(columns=60) 
Y = sonardata[60]
#print(X.head())
#print(Y.head())

#Spliting our dataset to Testing and Training 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)
#print(X.shape, X_train.shape, X_test.shape)
#print(Y.shape, Y_train.shape, Y_test.shape) 

#Model Training using the Logistic Regression Model -
model = LogisticRegression()
model.fit(X_train, Y_train)

#Let us check the accuracy score on the traing data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#print('Accuracy on the traing data: ', training_data_accuracy)

#Let us check the accuracy score on the traing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#print('Accuracy on the test data: ', test_data_accuracy)

#Mking the predictive system
input_data  = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
#changing the input_data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction == 'R':
    print('Rock')
else:
    print('Mine')
