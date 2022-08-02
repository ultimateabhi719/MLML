import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential

input_path = "./images/image_525.jpg"
model_weights = "./most_acc_model.h5"

Inp = np.expand_dims(np.asarray(Image.open(input_path).resize((180,180)))/255.0, axis=0)

def model():
    tf_model = Sequential()

    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                       input_shape=(180,180,3),
                       pooling='avg',classes=4,
                       weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable=False

    tf_model.add(pretrained_model)

    tf_model.add(Flatten())
    tf_model.add(Dense(64, activation='relu'))
    tf_model.add(Dense(4, activation='sigmoid',name='multi_label'))

    return tf_model

model = model()

model.load_weights(model_weights)
prediction = model.predict(Inp)[0]

print(prediction)
print([1 if x>0.5 else 0 for x in prediction])