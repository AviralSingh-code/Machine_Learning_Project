import numpy as np
import pandas as pd   #to put data to organized structure
from sklearn.preprocessing import StandardScaler #to standardize the data to a common range
from sklearn.model_selection import train_test_split
from sklearn import svm   #svm ----> support vector machine
from sklearn.metrics import accuracy_score

#loading the diabetes dataset to a pandas dataframe
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

#printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and columns in this dataset
diabetes_dataset.shape


# getting the stastical measures of the data
diabetes_dataset.describe()


diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

# separating data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

scaler = StandardScaler()


scaler.fit(X)   #we are fitting the data to standard scaler function


standardized_data = scaler.transform(X)   #we are standardizing the values of X

print(standardized_data)    #see that all the values are from 0 to 1

X = standardized_data   #reassigning to X
Y = diabetes_dataset['Outcome']

print(X)
print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


print(X.shape,X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear') #SVC ---> support vector classifier

#training the support vector mechine classifier
classifier.fit(X_train, Y_train)


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ',training_data_accuracy)


# accuracy score of the test data
X_test_prediction = classifier.predict(X_test)
test_data_prediction = accuracy_score(X_test_prediction, Y_test)


print('Accuracy score of the test data : ',test_data_prediction)



input_data = (11,143,94,33,146,36.6,0.254,51)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for 1 instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)

print(std_data)

prediction = classifier.predict(std_data)

print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')