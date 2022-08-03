import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential

# input Variables
input_path = "./images/image_525.jpg"
model_weights = "./best_model.h5"

# Model Function should be exactly the same as in train.py
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

model = model()

model.load_weights(model_weights)

## convert image to numpy array input
Inp = np.expand_dims(np.asarray(Image.open(input_path).resize((224,224)))/255.0, axis=0)
prediction = model.predict(Inp)[0]
pred = {'attr'+str(i+1):prediction[i] for i in range(4)}

print("model prediction probabilities: attr1 : {attr1:.3f}, attr2 : {attr2:.3f}, attr3 : {attr3:.3f}, attr4 : {attr4:.3f}".format(**pred))
print("predicted attributes:", [k for k in pred.keys() if pred[k]>0.5])