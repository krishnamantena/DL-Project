import preprocossingclassfile as pp

import matplotlib.pyplot as plt
import models
def fun():
    data = pp.preprocess_data()
    data.visualize_images(r"C:\Users\16307\PycharmProjects\DL-Project\images\train", nimages=5)
    image_df, train, label = data.preprocess(r"C:\Users\16307\PycharmProjects\DL-Project\images\train")
    image_df.to_csv('image_df.csv')

    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df, train, label)

    ANN_model = models.DeepANN
    Model1 = ANN_model.simple_model(ANN_model,[32,32,3],"sgd")
    print("train generator", tr_gen)
    ANN_history = Model1.fit(tr_gen, epochs=10, validation_data=vv_gen, batch_size=32)

    Ann_test_loss, Ann_test_acc = Model1.evaluate(tt_gen)

    Model1.save("my_model1.keras")

    print("The ann architecture is")

    print(Model1.summary())

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(ANN_history.history['loss'], label='Training')
    plt.plot(ANN_history.history['val_loss'], label='validation')
    plt.title('Training and Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ANN_history.history['accuracy'], label='Training')
    plt.plot(ANN_history.history['val_accuracy'], label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    plt.show()
if __name__ == '__main__':
    fun()






