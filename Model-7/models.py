import matplotlib.pyplot as plt
from keras.src.layers import Reshape, SimpleRNN, LSTM
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,BatchNormalization,Dropout

from tensorflow.keras.applications.vgg16 import VGG16

from keras.regularizers import l2

class DeepANN():
    def simple_model(self,input_shape=(28,28),op="sgd"):
        model = Sequential()

        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(7, activation='softmax'))

        model.compile(loss="categorical_crossentropy",optimizer=op,metrics=['accuracy'])

        return model

class DeepCNN():

    def simple_model(self, input_shape=(28,28,3), op="sgd", regularizers_sterength=0.01):
        cnn = Sequential()
        cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28,28,3]))
        cnn.add(MaxPool2D(pool_size=2, strides=2))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.2))
        cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(MaxPool2D(pool_size=2, strides=2))
        cnn.add(Flatten())
        cnn.add(Dense(500, kernel_regularizer=l2(regularizers_sterength), activation="relu"))
        cnn.add(Dense(units=7, activation='sigmoid',kernel_regularizer=l2(regularizers_sterength)))
        cnn.add(Dropout(0.2))

        cnn.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return cnn

    def cnn_transform_model(self,input_shape=(28,28,3),op="sgd"):
        cnn=Sequential()
        cnn.add(VGG16(weights="imagenet", include_top=False ,input_shape=(32,32,3)))
        cnn.add(Flatten())
        cnn.add(Dense(1024,activation='relu', input_shape=[32, 32,3]))
        cnn.add(Dense(128, activation='relu'))
        cnn.add(Dense(64, activation='relu'))
        cnn.add(Dense(7, activation='softmax'))

        cnn.compile(loss="categorical_crossentropy", optimizer=op, metrics=['accuracy'])

        return cnn


def train_model(model,tr_gen,vv_gen,epoches=5):
    history= model.fit(tr_gen, epochs=epoches, validation_data=vv_gen, batch_size=32)
    return history

def comapre_model(models,train_ge,val_gen,epoches=10):
    histories=[]

    for model in models:
        history=train_model(model,train_ge,val_gen,epoches)
        histories.append(history)


    plt.figure(figsize=(10,6))
    for i,history in enumerate(histories):
        plt.plot(history.history['accuracy'],label=f'Model {i+1}')
    plt.title('Model Training Accuracy Comparison')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(r"C:\Users\16307\PycharmProjects\DL-Project\app\static\image\acc.jpg")
    plt.show()

    for i,history in enumerate(histories):
        plt.plot(history.history['loss'],label=f'Model {i+1}')
    plt.title('Model Training Loss Comparison')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()
    plt.savefig(r"C:\Users\16307\PycharmProjects\DL-Project\app\static\image\acc2.jpg")
    plt.show()


class DeepRNN():

    def create_rnn_model(self,input_shape,no_of_classes):
        model=Sequential()

        #reshape layer to flatten the input images 28,28,3
        model.add(Reshape(  ( input_shape[0] * input_shape[1],input_shape[2]),input_shape=input_shape) )
        model.add(SimpleRNN(128))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(7,activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

        return model

    def create_LSTM_rnn_model(self, input_shape, no_of_classes):
        model = Sequential()

        # reshape layer to flatten the input images 28,28,3
        model.add(Reshape((input_shape[0] * input_shape[1], input_shape[2]), input_shape=input_shape))
        model.add(LSTM(128))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_cnn_model(self,input_shape,no_of_classes):
        model=Sequential()

        model.add(Conv2D(32,kernel_size=3, activation='relu',input_shape=input_shape))
        model.add(MaxPool2D((2,2)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(MaxPool2D((2, 2)))

        conv_output_shape = model.layers[-1].output_shape
        num_features = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]

        model.add(Reshape((conv_output_shape[1], num_features // conv_output_shape[1])))

        model.add(LSTM(128))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        return model







