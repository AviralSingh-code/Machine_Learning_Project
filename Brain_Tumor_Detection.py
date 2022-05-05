import numpy as np
import matplotlib.pyplot as plt
import os   #for handeling the files like opening closing deleting etc
import math
import shutil # this is to transfer files
import glob #for providing simple file paths

import warnings
warnings.filterwarnings('ignore') #to ignore the warnings

# for unzipping the data file
from zipfile import ZipFile
file_name = "Brain_Tumor_Data_MRI.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('Done')



# count the number of images in each classes
ROOT_DIR = "/content/Training"    #this is for the training data
number_of_images = {}

for dir in os.listdir(ROOT_DIR):
  number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))    #to get the number of images in each class

#illustration of listdir
os.listdir('/content/Training') #this basically gives the list of all the different classes of data

len(os.listdir('/content/Training'))  #to get the number of classes

# we will create a train folder
if not os.path.exists('./train'):   #for root directory we use dot(.)
  os.mkdir('./train')

  for dir in os.listdir(ROOT_DIR):
    os.makedirs("./train/"+dir) #+dir is to get the same distribution in the of classes i.e if we have glioma_tumor and no_tumor etc. classes then the train data will also have this


    for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR,dir)), size = (math.floor(70/100*number_of_images[dir])-5),replace=False): #np.random.choice helps use to choose the data randomly
      O = os.path.join(ROOT_DIR,dir,img)

      D = os.path.join("./train",dir)

      os.remove(O)

else:
  print("The folder exists")



# Thid function is used to split the data to training and test sets
def dataFolder(p,split):
  # we will create a train folder
  if not os.path.exists('./'+p):   #for root directory we use dot(.)
    os.mkdir('./'+p)

    for dir in os.listdir(ROOT_DIR):
      os.makedirs('./'+p+"/"+dir) #+dir is to get the same distribution in the of classes i.e if we have glioma_tumor and no_tumor etc. classes then the train data will also have this


      for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR,dir)), size = (math.floor(split*number_of_images[dir])-5),replace=False): #np.random.choice helps use to choose the data randomly
        O = os.path.join(ROOT_DIR,dir,img)

        D = os.path.join('./'+p,dir)
        shutil.copy(O,D)
        os.remove(O)

  else:
    print(f"{p}The folder exists")

dataFolder("train",0.7)

dataFolder("val", 0.15)

dataFolder("test", 0.15)

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization,GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras

# CNN Model


model = Sequential()

model.add(Conv2D(filters = 16, kernel_size=(3,3), activation='relu', input_shape = (224,224,3)))

model.add(Conv2D(filters = 36, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))  #to prevent overfitting

model.add(Conv2D(filters = 64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))  #to prevent overfitting

model.add(Conv2D(filters = 128, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))  #to prevent overfitting

model.add(Dropout(rate=0.25))   #rate is to dropout percentage --> 0.25 is 25 %

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))   #this is the dense layer
model.add(Dropout(rate=0.25))
model.add(Dense(units=4, activation='softmax'))   #output layer

model.summary()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

def preprocessingImages1(path):
  '''
  input : Path
  output : Pre Processed Images'''

  image_data = ImageDataGenerator(zoom_range=0.2, shear_range= 0.2, rescale=1/255, horizontal_flip= True) # Data augementation
  image =  image_data.flow_from_directory(directory = path, target_size=(224,224), batch_size= 32, class_mode='categorical')

  return image



path = '/content/Training'
train_data= preprocessingImages1(path)


def preprocessingImages2(path):
  '''
  input : Path
  output : Pre Processed Images'''

  image_data = ImageDataGenerator(rescale=1/255)
  image =  image_data.flow_from_directory(directory = path, target_size=(224,224), batch_size= 32, class_mode='categorical')

  return image


path = '/content/Testing'
test_data = preprocessingImages2(path)

path = '/content/val'
val_data = preprocessingImages2(path)


# early stopping and model check point
#Early stopping is to stop if the result comes up early
from keras.callbacks import ModelCheckpoint, EarlyStopping


#early stopping
es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience= 3, verbose= 1, mode = 'auto')


#Model checkpoint
mc = ModelCheckpoint(monitor="val_accuracy", filepath="./bestmodel.h5", verbose=1, save_best_only=True, mode = 'auto')

cd = [es,mc]    #cd -- > callback 

# Model Training
#Verbose means whatever execution is happening we want to display that
hs = model.fit_generator(generator= train_data, steps_per_epoch= 8, epochs= 30, verbose= 1, validation_data= val_data, validation_steps= 16, callbacks= cd)

# Model Graphical Interpretation

h = hs.history
h.keys()


plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c= 'red')

plt.title("acc vs val-acc")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c= 'red')

plt.title("loss vs val-loss")
plt.show()

# Model Accuracy
from keras.models import load_model

model= load_model('/content/bestmodel.h5')

# Accuarcy

acc = model.evaluate_generator(test_data)

print(f"The accuracy of our model is (acc)")

from keras.preprocessing.image import load_img, img_to_array

path = "/content/Training/pituitary_tumor/p (101).jpg"

img = load_img(path, target_size=(224,224))
input_arr = img_to_array(img)/255

plt.imshow(input_arr)
plt.show() 

input_arr.shape

input_arr = np.expand_dims(input_arr, axis=0)

pred = model.predict_classes(input_arr)[0][0]
# pred = (model.predict(test_data) > 0.5).astype("int32")
# pred = np.argmax(model.predict((input_arr)[0][0]),axis=1)
pred 

# print(pred)

if pred == 0:
  print('glioma_tumor')
elif pred == 1:
  print('meningioma_tumor')
elif pred == 2:
  print('no_tumor')
elif pred == 3:
  print('pituitary_tumor')

