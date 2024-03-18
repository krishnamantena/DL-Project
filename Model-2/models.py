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








