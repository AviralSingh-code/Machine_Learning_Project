import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loding the dataset to pandas dataframe
sonar_data = pd.read_csv('/content/Copy of sonar data.csv', header = None)

sonar_data.head()

#number of rows and columns
sonar_data.shape

sonar_data.describe()   #gives stastical measures of the data

sonar_data.describe()   #gives stastical measures of the data

sonar_data[60].value_counts()       #this is to count the binary dataset

sonar_data.groupby(60).mean()           #this is to calculate the mean of every column under each binary dataset

# separating data and labels ----> as this is a supervised learning model
# for unsupervised model we don't use labels
X = sonar_data.drop(columns=60, axis=1)  #to store all the columns except the last column ---> that is why we are dropping the last column
Y = sonar_data[60]      #to store the 60th column in Y

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()


#training the logistic regression model with training data
model.fit(X_train,Y_train)

#accuracy on the training data
X_train_prediciton = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediciton, Y_train)


print('Accuracy on training data : ',training_data_accuracy)


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data : ',test_data_accuracy)

input_data = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data) #converts input data to numpy array for better processing

# reshaping the numpy array as we are predicting for 1 instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)


if(prediction[0]=='M'):
  print('The object is a Mine')
else:
  print('The object is a Rock')