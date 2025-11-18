import numpy as nptest_loss, test_acc = model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import shutil
import random
from glob import glob

os.chdir('/Users/dianhaoli/Documents/data/')
SRC_DIR = 'train'

valid_ratio = 0.1
test_ratio = 0.1

for split in ['valid', 'test']:
    for label in os.listdir(SRC_DIR):
        os.makedirs(os.path.join(split, label), exist_ok=True)

for label in os.listdir(SRC_DIR):
    files = glob(os.path.join(SRC_DIR, label, '*.jpg'))
    random.shuffle(files)

    valid_end = int(len(files) * valid_ratio)
    test_end  = valid_end + int(len(files) * test_ratio)

    for f in files[:valid_end]:
        shutil.move(f, os.path.join('valid', label))

    for f in files[valid_end:test_end]:
        shutil.move(f, os.path.join('test', label))

print("Created 'valid' and 'test' folders inside /Users/dianhaoli/Documents/data/")

train_newdata = ImageDataGenerator(
    rescale=1./255,           
    rotation_range=30,       
    width_shift_range=0.1,    
    height_shift_range=0.1,  
    shear_range=0.1,         
    zoom_range=0.25,          
    horizontal_flip=True,     
    fill_mode='nearest'
    )    

valid_newdata = ImageDataGenerator(rescale=1./255)
test_newdata  = ImageDataGenerator(rescale=1./255)


train_generator = train_newdata.flow_from_directory(
    'train',
    target_size=(64, 64),    
    batch_size=32,
    class_mode='categorical'  
)

valid_generator = valid_newdata.flow_from_directory(
    'valid',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_newdata.flow_from_directory(
    'test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False

)

model = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(29, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

model.save('asl_model_v1.keras')


