import preprocossingclassfile as pp

import matplotlib.pyplot as plt
import models
def fun():
    data = pp.preprocess_data()
    data.visualize_images(r"C:\Users\16307\PycharmProjects\DL-Project\images\train", nimages=5)
    image_df, train, label = data.preprocess(r"C:\Users\16307\PycharmProjects\DL-Project\images\train")
    image_df.to_csv('image_df.csv')

    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df, train, label)




    # '''This multiple optimizers

    ANN_model = models.DeepANN
    Model1 = ANN_model.simple_model(ANN_model, (32, 32,3),"sgd")
    Model2 = ANN_model.simple_model(ANN_model, (32, 32,3),"Adam")
    Model3 = ANN_model.simple_model(ANN_model, (32, 32,3), "rmsprop")

    print("train generator", tr_gen)
    ANN_history = models.comapre_model([Model1, Model2, Model3], tr_gen, vv_gen, 3)


if __name__ == '__main__':
    fun()






