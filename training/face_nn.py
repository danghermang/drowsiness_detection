import os
import pickle

import cv2
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import AlphaDropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import interactive

from utils import image_to_package
from configs.config import *
# script that trains convolutional neural network

INIT_LR = 1e-3
epochs = 100
BS = 10
num_classes = 5
batch_size = 32
max_training_images = 15000
max_training_images_per_class = 1000
patience = 15
crossentropy = True
relu = True


def show_plot(history):
    plt.figure(1)
    plt.plot(history['val_acc'])
    plt.plot(history['acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    interactive(True)
    plt.figure(2)

    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    interactive(False)
    plt.show()


def get_model_relu(height, width, depth, classes, crossentropy=True):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape
                     , kernel_initializer=keras.initializers.lecun_normal()
                     ))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same",
                     input_shape=inputShape,
                     kernel_initializer=keras.initializers.lecun_normal()
                     ))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same",
                     input_shape=inputShape,
                     kernel_initializer=keras.initializers.lecun_normal()
                     ))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same",
                     input_shape=inputShape,
                     kernel_initializer=keras.initializers.lecun_normal()
                     ))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same",
                     input_shape=inputShape,
                     kernel_initializer=keras.initializers.lecun_normal()
                     ))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024
                    , kernel_initializer=keras.initializers.lecun_normal()
                    ))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    if crossentropy:
        model.add(Activation("softmax"))
    else:
        model.add(Activation("sigmoid"))
    print("Loaded relu model")
    return model


def get_model_selu(height, width, depth, classes, crossentropy=True):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape,
                     bias_initializer='zeros',
                     kernel_initializer=keras.initializers.lecun_normal()))
    model.add(Activation("selu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(AlphaDropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same",
                     input_shape=inputShape, bias_initializer='zeros',
                     kernel_initializer=keras.initializers.lecun_normal()))
    model.add(Activation("selu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same",
                     input_shape=inputShape, bias_initializer='zeros',
                     kernel_initializer=keras.initializers.lecun_normal()))
    model.add(Activation("selu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AlphaDropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same",
                     input_shape=inputShape, bias_initializer='zeros',
                     kernel_initializer=keras.initializers.lecun_normal()))
    model.add(Activation("selu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same",
                     input_shape=inputShape, bias_initializer='zeros',
                     kernel_initializer=keras.initializers.lecun_normal()))
    model.add(Activation("selu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AlphaDropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, bias_initializer='zeros',
                    kernel_initializer=keras.initializers.lecun_normal()))
    model.add(Activation("selu"))
    model.add(BatchNormalization())
    model.add(AlphaDropout(0.5))

    model.add(Dense(classes, bias_initializer='zeros',
                    kernel_initializer=keras.initializers.lecun_normal()))
    if crossentropy:
        model.add(Activation("softmax"))
    else:
        model.add(Activation("sigmoid"))
    print("loaded selu model")
    return model


def train_generator_model(train_gen, x_test, y_test, steps,
                          model=None, batch_size=32, epochs=10, patience=5, crossentropy=True, relu=True):
    if model is None:
        if crossentropy:
            if relu:
                model = get_model_relu(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], 2, crossentropy)
            else:
                model = get_model_selu(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], 2, crossentropy)
        else:
            if relu:
                model = get_model_relu(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], 1, crossentropy)
            else:
                model = get_model_selu(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], 1, crossentropy)

    model.summary()

    if crossentropy:
        opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
        loss = 'categorical_crossentropy'
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    else:
        opt = Adam(lr=1e-4)
        loss = 'binary_crossentropy'
        model.compile(loss=loss, optimizer=opt, metrics=['acc'])
    es_val_acc = keras.callbacks.EarlyStopping(monitor='val_acc', patience=patience, mode='auto')
    es_acc = keras.callbacks.EarlyStopping(monitor='acc', patience=patience, mode='auto')
    callbacks_vector = [es_val_acc]

    checkpoint = keras.callbacks.ModelCheckpoint(IMAGE_MODEL_BEST_WEIGHTS, monitor='val_acc', verbose=1,
                                                 save_best_only=True, mode='max')
    callbacks_vector.append(checkpoint)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                  patience=patience // 2, min_lr=0.001)
    callbacks_vector.append(reduce_lr)
    history = model.fit_generator(train_gen, shuffle=False, epochs=epochs,
                                  verbose=1, steps_per_epoch=steps // batch_size,
                                  validation_data=(x_test, y_test),
                                  callbacks=callbacks_vector)

    model_json = model.to_json()
    with open(IMAGE_MODEL, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(IMAGE_MODEL_WEIGHTS)

    return history.history


if __name__ == "__main__":
    x_test, y_test = image_to_package.pre_load_images(training=False, width=IMAGE_DIMS[1],
                                                      max_img=20000, crossentropy=crossentropy)
    # train_gen = image_to_package.gen_load_images(training=True, width=IMAGE_DIMS[1],
    #                                              max_type_img=1000,
    #                                              batch_size=batch_size, augumented=False,
    #                                              crossentropy=crossentropy)
    # steps_per_batch = len(image_to_package.get_files(PROCESSED_IMAGES_TRAINING))
    #
    # history = train_generator_model(train_gen, x_test, y_test, batch_size=batch_size,
    #                                 epochs=epochs, steps=min(1000*2, steps_per_batch),
    #                                 patience=patience, crossentropy=crossentropy, relu=relu)
    # pickle.dump(history, open(IMAGE_MODEL_HISTORY, "wb"))
    #
    # history = pickle.load(open(IMAGE_MODEL_HISTORY, "rb"))
    # print(history)
    # show_plot(history)

    train_gen = image_to_package.gen_load_images(training=True, width=IMAGE_DIMS[1],
                                                 max_type_img=max_training_images_per_class,
                                                 batch_size=batch_size, augumented=True, crossentropy=crossentropy)
    steps_per_batch = len(image_to_package.get_files(PROCESSED_IMAGES_TRAINING))

    with open(IMAGE_MODEL, 'r') as f:
        loaded_model_json = f.read()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(IMAGE_MODEL_BEST_WEIGHTS)
    history = train_generator_model(train_gen, x_test, y_test, batch_size=batch_size, model=loaded_model,
                                    epochs=epochs, steps=min(max_training_images_per_class * 2, steps_per_batch),
                                    patience=patience, crossentropy=crossentropy, relu=relu)
    show_plot(history)
    pickle.dump(history, open(IMAGE_MODEL_HISTORY, "wb"))

    # history = pickle.load(open(IMAGE_MODEL_HISTORY, "rb"))
    # print(history)
    # from keras.utils import plot_model
    #
    # plot_model(loaded_model, to_file='model.png')
    # raise Exception
    prediction_generator = image_to_package.gen_load_images(training=False, width=IMAGE_DIMS[1],
                                                            max_type_img=100000,
                                                            batch_size=1)

    # cv2.namedWindow("Cadru", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Cadru", 720, 480)
    count = 0
    try:
        for i, (image, y_test) in enumerate(prediction_generator):
            image = image[0]
            prediction = image_to_package.get_prediction_image(loaded_model, image, normalize=False)
            if prediction:
                color = GREEN
            else:
                color = RED
            # cv2.putText(image, "Is awake: " + str(prediction),
            #             (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            # cv2.imshow("Cadru", image)
            if y_test[0][0] == 1:
                tag = False
            else:
                tag = True
            if prediction == tag:
                print("mask", i, True)
            else:
                count += 1
                print("mask", i, False)
            # key = cv2.waitKey(3000) & 0xFF
            # # if the `q` key was pressed, break from the loop
            # if key == ord("q") or key == ord("Q"):
            #     break
            # if key == ord("c") or key == ord("C"):
            #     continue
            if i > 4000:
                break
    finally:
        cv2.destroyAllWindows()
        print(count, "wrong")
        print("done")
        K.clear_session()
