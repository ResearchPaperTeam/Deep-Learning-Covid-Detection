import os
import tensorflow as tf
import numpy as np
import pandas as pd

train_dir='Datasets/impli/xray_dataset_covid19/train'
test_dir='Datasets/impli/xray_dataset_covid19/test'

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

from tensorflow import keras
IMAGE_SIZE = [224, 224] 

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
  layer.trainable = False

from glob import glob
import matplotlib.pyplot as plt

folders = glob('Datasets/impli/xray_dataset_covid19/train/*')

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

classes=os.listdir(train_dir)
print(classes)

