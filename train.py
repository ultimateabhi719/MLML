import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from keras.preprocessing.image import ImageDataGenerator


# Load Data
data_csv = "./labels.txt"
img_directory = "./images"
data_raw = pd.read_csv(data_csv,sep=" ",names=['img','attr1','attr2','attr3','attr4'])


# Simple Data Imputation
## a simple data imputation based on mean values of the multi-label classification problem
my_imputer = SimpleImputer()
data_imputed = data_raw.copy(deep=True)
data_imputed[list(data_raw.columns[1:])] = my_imputer.fit_transform(data_raw[list(data_raw.columns[1:])])
data_imputed[list(data_raw.columns[1:])] = data_imputed[list(data_raw.columns[1:])].applymap(lambda x:1.0 if x>=0.5 else 0.0)



# Train, Test, and Validation DataGenerator
df = data_imputed.sample(frac = 1)
N_train = int(df.shape[0]*0.8)
N_train_ = int(df.shape[0]*0.8*0.8)
batch_size = 32

columns=["attr1", "attr2", "attr3", "attr4"]
datagen=ImageDataGenerator(rescale = 1./255)
test_datagen=ImageDataGenerator(rescale = 1./255)
train_generator=datagen.flow_from_dataframe(
    dataframe=df[:N_train_],
    directory=img_directory,
    x_col="img",
    y_col=columns,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(224,224))

valid_generator=test_datagen.flow_from_dataframe(
    dataframe=df[N_train_:N_train],
    directory=img_directory,
    x_col="img",
    y_col=columns,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(224,224))

test_generator=test_datagen.flow_from_dataframe(
    dataframe=df[N_train:],
    directory="/Users/abhiag/AImonk/images",
    x_col="img",
    y_col=columns,
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode="raw",
    target_size=(224,224))

## Weighted Binary Entropy with different weights for each class
def custom_loss(Wp,Wn):
    def _custom_loss(y_true, y_logit):
        '''
        Multi-label cross-entropy
        y_true: true value
        y_logit: predicted value
        '''
        # print("logit:",K.int_shape(y_logit), "\t true:", K.int_shape(y_true))
        loss = float(0)
        # print( K.int_shape(y_true), K.int_shape(y_logit)) 
        first_term = Wp * y_true * K.log(y_logit + K.epsilon())
        second_term = Wn * (1 - y_true) * K.log(1 - y_logit + K.epsilon())
        loss -= (first_term + second_term)
        return K.sum(loss)
    return _custom_loss

## compute F1-score as a checkpoint for validation and training data while training the model
def f1_score(y_true, y_logit):
    '''
    Calculate F1 score
    y_true: true value
    y_logit: predicted value
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_logit, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    predicted_positives = K.sum(K.round(K.clip(y_logit, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return (2 * precision * recall) / (precision + recall + K.epsilon())

# Training

## use ResNet50 for transfer learning

def model():
    tf_model = Sequential()

    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                       input_shape=(224,224,3),
                       pooling='avg',classes=4,
                       weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable=False

    tf_model.add(pretrained_model)

    tf_model.add(Flatten())
    tf_model.add(Dense(64, activation='relu'))
    tf_model.add(Dense(4, activation='sigmoid',name='multi_label'))

    return tf_model

resnet_model = model()



# positive_weights = {}; negative_weights = {}
# for label in sorted(columns):
#     positive_weights[label] = df.shape[0] /(2 * sum(df[label] == 1))
#     negative_weights[label] = df.shape[0] /(2 * sum(df[label] == 0))
# _Wp = [positive_weights[k] for k in Wp.keys()]
# _Wn = [negative_weights[k] for k in Wp.keys()]
# resnet_model.compile(optimizer=Adam(lr=0.001),loss = {'multi_label': custom_loss(_Wp, _Wn)},metrics=['accuracy',f1_score])
resnet_model.compile(optimizer=Adam(lr=0.0001),loss = 'binary_crossentropy',metrics=['accuracy',f1_score])

resnet_model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor = 'val_f1_score', save_best_only = True, mode = 'max', verbose = 1)

history = resnet_model.fit(train_generator, validation_data=valid_generator, epochs=30, callbacks=[cp_callback])


# Save Loss Curve Plot
fig1 = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Aimonk_multilabel_problem')
plt.ylabel('training_loss')
plt.xlabel('iteration_number')
plt.legend(['train', 'validation'])
# plt.show()
plt.savefig('loss_curve.png')

