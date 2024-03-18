import matplotlib.pyplot as plt
from keras.src.layers import Reshape, SimpleRNN, LSTM
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,BatchNormalization,Dropout

from tensorflow.keras.applications.vgg16 import VGG16

from keras.regularizers import l2

class DeepANN():
    def simple_model(self,input_shape=(32,32,3),op="sgd"):
        model = Sequential()

        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(7, activation='sigmoid'))

        model.compile(loss="binary_crossentropy",optimizer=op,metrics=['accuracy'])

        return model
