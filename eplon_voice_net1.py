#coding: utf-8

from keras import optimizers, losses
from keras.layers import *
from keras.models import Model
from keras.backend import int_shape
from keras.utils import to_categorical, plot_model
import numpy as np
import matplotlib.pyplot as plt
import eplon_voice_deeplearning

def prepare_simple_CNN_model(input_shape=(28, 28, 1), class_num=2):
    input = Input(input_shape)
    kernel_size = (3, 3)
    max_pool_size = (2, 2)

    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(input)
    cnn = Dropout(0.1)(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

    fc = Flatten()(cnn)
    fc = Dense(1024, activation='relu')(fc)
    softmax = Dense(class_num, activation='softmax')(fc)
    model = Model(input=input, output=softmax)

    return model


def learning():

    model = prepare_simple_CNN_model()
    print(model.summary())

    # Training configuration
    model.compile(loss=losses.categorical_crossentropy,
    optimizer=optimizers.Adam(),
    metrics=['accuracy'])

    # Data preparation
    print("-----readed data------------------")

    input_shape = (28, 28, 1)
    (X_train, y_train), (X_test, y_test) = eplon_voice_deeplearning.eplon_voice_deeplearning_input_learning_data("./dataset_441kHz_output")

    print("SHAPE: ",X_train.shape)

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train / 127
    X_test = X_test / 127


    # 正解ラベルをダミー変数に変換
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Training
    train_history = model.fit(X_train, y_train,
                                batch_size=128,
                                epochs=60,
                                verbose=1,
                                validation_data=(X_test, y_test))
    # Evaluation
    result = model.evaluate(X_test, y_test, verbose=0)

    print("Test Loss", result[0])
    print("Test Accuracy", result[1])

    # Plotting loss and accuracy metrics
    accuracy = train_history.history['acc']
    val_accuracy = train_history.history['val_acc']
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Training accuracy')
    plt.title('Naive_CNN Training accuracy')
    plt.savefig('Naive_CNN_train_acc.png')
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Training loss')
    plt.title('Naive_CNN Training loss')
    plt.savefig('Naive_CNN_train_loss.png')
    plt.figure()


    #モデルの保存
    model_json_str = model.to_json()
    open('net1.json', 'w').write(model_json_str)
    model.save_weights('net1.h5')
