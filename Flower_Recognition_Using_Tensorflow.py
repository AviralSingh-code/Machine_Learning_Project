#Importing libraries

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Data Preprocessing

#Training Image processing

train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
                'training_set',
                target_size=(64,64),batch_size=32,class_mode='categorical')



#Test Image Processing
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('test_set',target_size=(64,64),batch_size=32,class_mode='categorical')

#Building Model

cnn = tf.keras.models.Sequential() #cnn is the convolutional neural network


# Building the first layer ----> convolution layer

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[64,64,3]))  #filter is the first thing that we apply that puts the data to other
#matrix  so if we have 3x3 matrix then the filter will be filters=3
#activation function is the thing that performs the task of filteration
#input_shape defines the of the image 
#as after preprocessing the size of the image becomes 64x64 so we provide 64,64
#3 in input shape defines that the image is RGB


#Pooling layer --->second layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#We are using the max pool over here
#strides is the number of jumps


############# SECOND TIME TO IMPROVE THE ACCURACY

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))  #filter is the first thing that we apply that puts the data to other
#matrix  so if we have 3x3 matrix then the filter will be filters=3
#activation function is the thing that performs the task of filteration
#input_shape defines the of the image 
#as after preprocessing the size of the image becomes 64x64 so we provide 64,64
#3 in input shape defines that the image is RGB


#Pooling layer --->second layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#We are using the max pool over here
#strides is the number of jumps


cnn.add(tf.keras.layers.Dropout(0.5))  #it will optimize the process to get good result


# Flattening ----> after the 2nd layer
cnn.add(tf.keras.layers.Flatten())

# Now we have to make Artificial Neural Network
# For this we have to add hidden layer
cnn.add(tf.keras.layers.Dense(units=128,activation='relu')) #---> for hidden layer
#units is the number of hidden layers that we want ---> over here we have 128 hidden layers
#activation function for all computation

# For output layer
cnn.add(tf.keras.layers.Dense(units=5,activation='softmax'))
#units is the number of classes eg daisy, rose, sunflower, dandalion, tulip
# for binary data we have to provide units=1 and not 2
#for binary data we provide 1 because if we have 1 the image belongs to a particular class
# for 0 the image doesn't belong to that class that means it belongs to the other class
# for binary data we can use the Sigmoid activation function in the output layer


cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#optimizer is like a compiler
#for binary Adam optimizer is good 
#for binary loss is loss='binary_crossentropy'
#metrics is for getting the accuracy

cnn.fit(x=training_set,validation_data=test_set,epochs=30)
#epochs is like a cycle like during training 30 cycles will be performed


#Preprocess new image ---> input
from keras.preprocessing import image
test_image = image.load_img(r"C:\Users\Lenovo\OneDrive\Desktop\Flower_Recognition\Prediction\daisy.jpg.jpg",target_size=(64,64))
test_image = image.img_to_array(test_image) #converts the image to array
test_image = np.expand_dims(test_image, axis=0) #dims means dimensions
result = cnn.predict(test_image)
training_set.class_indices  #gives indices to all the different classes



print(result)


if result[0][0]==1:
    print('Daisy')
elif result[0][1]==1:
    print('Dandelion')
elif result[0][2]==1:
    print('Rose')
elif result[0][3]==1:
    print('Sunflower')
elif result[0][4]==1:
    print('Tulip')