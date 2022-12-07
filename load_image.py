# =============================================================================
# from skimage.io import imread_collection,imread, show, imshow_collection, imshow, ImageCollection
# 
# #your path 
# col_dir = 'C:/Users/jibra/pieas/icv/Ox_uni/*.jpg'
# 
# #creating a collection with the available images
# sex = ImageCollection(col_dir)
# print(len(sex))
# imshow(sex[69])
# show()
# 
# col = imread_collection(sex)
# imshow_collection(col)
# show()
#     
# print('sex')
# =============================================================================


import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf

import cv2
import os

import numpy as np

print('alright')

labels = ['Chapman', 'Cockcroft', 'Library', 'Maxwell', 'Media_City_Campus', 'New_Adelphi', 'New_Science', 'Newton', 'Sports_Center', 'University_House']
img_size = 64
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
print('here')

train = get_data("C:/Users/jibra/pieas/icv/uos/dataset")


# print(train)
plt.figure(figsize = (5,5))
plt.imshow(train[3][0])
# print(train[0][1])
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])

# =============================================================================
train1, val = train_test_split(train, test_size=0.2, random_state=25)
# print(train1.shape)
# print(val.shape)
# # 
# =============================================================================


x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train1:
  x_train.append(feature)
  y_train.append(label)

# =============================================================================
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
# 
# =============================================================================
# print(x_train.shape)
# print(x_val.shape)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
datagen.fit(x_val)
print(x_train.shape)
print(x_val.shape)

model = Sequential()
model.add(Conv2D(32, 3,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),  activation="relu", input_shape=(64,64,3)))
model.add(MaxPool2D())


model.add(Conv2D(32, 3,kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1), activation="relu"))
model.add(MaxPool2D())



model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))


model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(10, activation="softmax"))

# # Create the model
# model = Sequential()

# # Add the convolutional layers
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# # Add the dense layers
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))



# # Fit the model on the training data
# model.fit(x_train, y_train, epochs=10, batch_size=32)

print(model.summary())


# opt = SGD(lr=0.01)
# model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs = 100, validation_data=(x_val, y_val),  batch_size = 32)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(100)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict(x_train)

print(predictions)

score = model.evaluate(x_val, y_val)
print(score)








