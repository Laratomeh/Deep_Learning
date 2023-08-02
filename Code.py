#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import os
from os import listdir
import keras
import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ReLU
from keras.layers.core import Activation
from tensorflow.keras.preprocessing import image
from keras.regularizers import l2


# ## Reading Images and labels

# In[56]:


# Read and normalize training images
train_folder='D:/Uni_Buc/Uni_Buc_semester_3/Deep_Learning/Project1/train_images/'
train_images = []
# Normalizing the images
for filename in os.listdir(train_folder):
    img = mpimg.imread(os.path.join(train_folder, filename))
    img = image.img_to_array(img)
    img = img.astype('float32')
#     img = img/255
    train_images.append(img)
    
X_train = np.array(train_images)
X_train.shape


# In[57]:


# Read training lebels
train_labels = pd.read_csv('train_labels.csv')
y_train = np.array(train_labels.drop(['id'],axis=1))
y_train.shape


# In[58]:


# Read and normalize validation images
val_folder='D:/Uni_Buc/Uni_Buc_semester_3/Deep_Learning/Project1/val_images/'
val_images = []
for filename in os.listdir(val_folder):
    img = mpimg.imread(os.path.join(val_folder, filename))
    img = image.img_to_array(img)
    img = img.astype('float32')
#     img = img/255
    val_images.append(img)

X_val = np.array(val_images)
X_val.shape


# In[59]:


# Read validation labels
val_labels = pd.read_csv('val_labels.csv')
y_val = np.array(val_labels.drop(['id'],axis=1))
y_val.shape


# In[60]:


# Read and normalize testing images
test_folder='D:/Uni_Buc/Uni_Buc_semester_3/Deep_Learning/Project1/test_images/'
test_images = []
for filename in os.listdir(test_folder):
    img = mpimg.imread(os.path.join(test_folder, filename))
    img = image.img_to_array(img)
    img = img.astype('float32')
#     img = img/255
    test_images.append(img)

X_test = np.array(test_images)
X_test.shape


# In[61]:


# get the path/directory of testing images
folder_dir = "D:/Uni_Buc/Uni_Buc_semester_3/Deep_Learning/Project1/test_images"
img_ids = []
for images in os.listdir(folder_dir):
        if (images.endswith(".jpeg") or images.endswith(".png")\
            or images.endswith(".jpg")):
            img_ids.append(images)

img_ids = np.array(img_ids)
df = pd.DataFrame(img_ids, columns = ['id'])


# ## Recall, Precision, F1 scores Function

# In[62]:


def precision_score(y_true, y_pred):
        true_p = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_p = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_p / (predicted_p + K.epsilon())
        return precision
def recall_score(y_true, y_pred):
        true_p = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_p = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_p / (possible_p + K.epsilon())
        return recall
def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# ## Building, Compiling and Fitting the model

# ### Model 1

# In[63]:


# # Building the model
# # Model 1 ### This is my best model with 20 epoch and 64 batch size adam optimizer without defining any params
# model = Sequential()
# model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (64, 64, 1)))
# model.add(MaxPooling2D(pool_size = (3,3)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (3,3)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (3,3)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(3, activation = 'sigmoid'))
# model.summary()


# In[64]:


# # Compiling and fitting the model
# epochs = 20
# batch_size = 64
# optimizer='adam'
# model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy' ,recall_score, precision_score, f1_score])
# history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_val, y_val))


# ### Model 2

# In[65]:


# # Building the model
# # Model 2
# model = Sequential()
# model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (64, 64, 1)))
# model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (3,3)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (3,3)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (3,3)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(3, activation = 'sigmoid'))
# model.summary()


# In[66]:


# # Compiling and fitting the model
# epochs = 20
# batch_size = 64
# optimizer = keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999)
# model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy' ,recall_score, precision_score, f1_score])
# history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_val, y_val))


# ### Model 3

# In[67]:


# # Building the model
# # Model 3
# model = Sequential()
# model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (64, 64, 1)))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.6))
# model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.6))
# model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.6))
# model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.6))
# model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.6))
# model.add(Conv2D(filters = 128, kernel_size = (5,5), padding = 'Same', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.6))
# model.add(Flatten())
# model.add(Dense(128, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
# model.add(Dropout(0.6))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(3, activation = 'sigmoid'))
# model.summary()


# In[68]:


# # Compiling and fitting the model
# epochs = 15
# batch_size = 128
# optimizer = keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999)
# # optimizer='adam'
# model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy' ,recall_score, precision_score, f1_score])
# history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_val, y_val))


# ### Model 4

# In[69]:


# Building the model
# Model 4 
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (64, 64, 1)))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(3, activation = 'sigmoid'))
model.summary()


# In[70]:


# Compiling and fitting the model
epochs = 20
batch_size = 128
optimizer='adam'
model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy' ,recall_score, precision_score, f1_score])
history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_val, y_val))


# ### Model 5

# In[71]:


# # Building the model
# model = Sequential()
# model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (64, 64, 1)))
# model.add(MaxPooling2D(pool_size = (1,1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (1,1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (1,1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(3, activation = 'sigmoid'))
# model.summary()


# In[72]:


# # Compiling and fitting the model
# epochs = 20
# batch_size = 128
# optimizer='adam'
# model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy' ,recall_score, precision_score, f1_score])
# history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_val, y_val))


# ## Training and Validation Accuracy

# In[73]:


# Print out the training data metrics
print('Training Accuracy')
training = model.evaluate(X_train,y_train)
print('Loss: {} \nAccuracy: {} \nRecall: {} \nPrecision: {} \nF1-Score: {}'.format(training[0],
                                                                                    training[1],training[2],
                                                                                    training[3],training[4]))


# In[74]:


# Print out the validation data metrics
print('Validation Accuracy')
validation = model.evaluate(X_val,y_val)
print('Loss: {} \nAccuracy: {} \nRecall: {} \nPrecision: {} \nF1-Score: {}'.format(validation[0],
                                                                                    validation[1],validation[2],
                                                                                    validation[3],validation[4]))


# In[75]:


# Print out the classification report of training data 
y_train_pred = np.round(model.predict(X_train), 0)
X_train_report = metrics.classification_report(y_train, y_train_pred)
print('Classification report of Training data ')
print(X_train_report)


# In[76]:


# Print out the classification report of validation data
y_val_pred = np.round(model.predict(X_val), 0)
X_val_report = metrics.classification_report(y_val, y_val_pred)
print('Classification report of Validating data ')
print(X_val_report)


# In[77]:


# Plotting the accuracy, loss, recall, and precision
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
recall_score = history.history['recall_score']
val_recall_score = history.history['val_recall_score']
plt.plot(epochs, recall_score, 'b', label='Training Recall')
plt.plot(epochs, val_recall_score, 'g', label='Validation Recall')
plt.title('Training and Validation Recall')
plt.legend()

plt.subplot(1, 2, 2)
precision_score = history.history['precision_score']
val_precision_score = history.history['val_precision_score']
plt.plot(epochs, precision_score, 'b', label='Training Precision')
plt.plot(epochs, val_precision_score, 'g', label='Validation Precision')
plt.title('Training and Validation Precision')
plt.legend()
plt.show()


# ## Save the Model and make Predictions

# In[78]:


# Save the model
model.save(os.path.join('models','DL_model.h5'))


# In[79]:


# Make predictions on testing data
test_predictions = model.predict(X_test)
test_predictions[0]


# In[80]:


# Choose a threshold of predictions
test_predictions = np.where(test_predictions > 0.65, 1, 0) # pick a value
test_labels = pd.DataFrame(test_predictions, columns = ['label1','label2','label3'])


# In[81]:


# Build the dataframe
final_df=pd.concat([df, test_labels.reindex(df.index)], axis=1)
final_df


# In[82]:


# save the dataframe to csv file
final_df.to_csv('submission_model.csv',index=False)

