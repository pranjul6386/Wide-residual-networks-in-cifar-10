import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras.datasets import cifar10
import keras.utils.np_utils as kutils
import wide_residual_network_fix_v4 as wrn

(train_x,train_y),(test_x,test_y)=cifar10.load_data()
print(test_x.shape)
print(train_x.shape)
print(train_y.shape)
train_y=kutils.to_categorical(train_y)
test_y=kutils.to_categorical(test_y)
train_norm=train_x/255.0
test_norm=test_x/255.0
