# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:43:50 2020

@author: sabab
"""
import tensorflow as tf
from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import os
import matplotlib.pyplot as plt

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape = input_Shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape = input_Shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu',kernel_initializer='he_uniform'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(classes_n, activation = 'softmax')
        ])
    print(model.summary())
    return model


def plot_graphs(f, history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

#initialize parameters
epochs = 10
batch_size= 128
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
met = 'accuracy'


# load dataset
(x_train, y_train), (x_test, y_test) = load_data()

# reshape data to have a single channel
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))


input_Shape = x_train.shape[1:]
classes_n = len(unique(y_train))

#normamize
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#model weight save
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#define model
model = create_model()

# Save the entire model as a SavedModel.
my_model_path = os.path.dirname('saved_model/my_model')
model.save(my_model_path) 

#compile model
model.compile(optimizer=optimizer, loss = loss, metrics=[met])

#fit the mdoel
history = model.fit(x_train, y_train, validation_data= (x_test,y_test), epochs=epochs, batch_size=batch_size, callbacks=[cp_callback],verbose=2)

#plot 
f1 = plt.figure()
plot_graphs(f1,history, "accuracy")
f2 = plt.figure()
plot_graphs(f2,history, "loss")

# =============================================================================
# #load checkpoint (load weights)
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# 
# # Load the previously saved weights
# model.load_weights(latest)
# 
# =============================================================================
# =============================================================================
# Loading model
# my_model_path = os.path.dirname('saved_model/my_model')
# new_model = tf.keras.models.load_model(my_model_path)
# 
# # Check its architecture
# new_model.summary()
# =============================================================================


# evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy on testing: {:.3f}".format(acc))


