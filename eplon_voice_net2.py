#coding: utf-8

from keras import optimizers, losses
from keras.layers import *
from keras.models import Model
from keras.backend import int_shape
from keras.utils import to_categorical, plot_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import eplon_voice_deeplearning

def ss_block(block_input, num_filters, ratio=8):                             # SS

	'''
		Args:
			block_input: input tensor to the ss block
			num_filters: no. of filters/channels in block_input
			ratio: a hyperparameter that denotes the ratio by which no. of channels will be reduced

		Returns:
			scale: scaled tensor after getting multiplied by new channel weights
	'''

	pool1 = GlobalAveragePooling2D()(block_input)
	flat = Reshape((1, 1, num_filters))(pool1)
	dense1 = Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = Dense(num_filters, activation='sigmoid')(dense1)
	scale = multiply([block_input, dense2])

	return scale

def secretnet_block(block_input, num_filters):                                  # Single secretnet block

	'''
		Args:
			block_input: input tensor to the secretnet block
			num_filters: no. of filters/channels in block_input

		Returns:
			relu2: activated tensor after addition with original input
	'''

	if int_shape(block_input)[3] != num_filters:
		block_input = Conv2D(num_filters, kernel_size=(1, 1))(block_input)

	conv1 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
	norm1 = BatchNormalization()(conv1)
	relu1 = Activation('relu')(norm1)
	conv2 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(relu1)
	norm2 = BatchNormalization()(conv2)

	se = ss_block(norm2, num_filters=num_filters)

	sum = Add()([block_input, se])
	relu2 = Activation('relu')(sum)

	return relu2

def secretnet():

	'''
		Adapted for MNIST dataset.
		Input size is 28x28x1 representing images in MNIST.
		Output size is 10 representing classes to which images belong.
	'''

	input = Input(shape=(28, 28, 1))
	conv1 = Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(input)
	pool1 = MaxPooling2D((2, 2), strides=2)(conv1)

	block1 = secretnet_block(pool1, 64)
	block2 = secretnet_block(block1, 64)

	pool2 = MaxPooling2D((2, 2), strides=2)(block2)

	block3 = secretnet_block(pool2, 128)
	block4 = secretnet_block(block3, 128)

	pool3 = MaxPooling2D((3, 3), strides=2)(block4)

	block5 = secretnet_block(pool3, 256)
	block6 = secretnet_block(block5, 256)

	pool4 = MaxPooling2D((3, 3), strides=2)(block6)
	flat = Flatten()(pool4)

	output = Dense(2, activation='softmax')(flat)

	model = Model(inputs=input, outputs=output)
	return model


def learning():

    model = secretnet()
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

    X_train = X_train / 25
    X_test = X_test / 25


    # 正解ラベルをダミー変数に変換
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Training
    train_history = model.fit(X_train, y_train,
                                batch_size=128,
                                epochs=150,
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
    plt.title('SS Training accuracy')
    plt.savefig('SS_train_acc.png')
	plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Training loss')
    plt.title('SS Training loss')
    plt.savefig('SS_train_loss.png')
	plt.figure()


    #モデルの保存
    model_json_str = model.to_json()
    open('net2.json', 'w').write(model_json_str)
    model.save_weights('net2.h5')
