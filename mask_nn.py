import os
import pickle

import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Dense, AlphaDropout, Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import interactive

import image_to_package

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
IMAGE_DIMS = (200, 200, 3)
INIT_LR = 1e-3
epochs = 3000
BS = 10
batch_size = 300
max_training_images = 15000
max_training_images_per_class = 60000
patience = 5
processed_images_path = '../data/processed_images'
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


def get_model_relu(input_size, output, crossentropy=True):
    model = Sequential()
    model.add(Dense(500, input_shape=(input_size,)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.25))

    for index in range(5):
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(0.25))

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))

    model.add(Dense(output))
    if crossentropy:
        model.add(Activation("softmax"))
    else:
        model.add(Activation("sigmoid"))

    model.summary()

    return model


def get_model_selu(input_size, output, crossentropy=True):
    model = Sequential()
    model.add(Dense(500, input_shape=(input_size,), bias_initializer='zeros',
                    kernel_initializer=keras.initializers.lecun_normal()))
    model.add(Activation('selu'))
    model.add(BatchNormalization(axis=1))
    model.add(AlphaDropout(0.25))

    for index in range(5):
        model.add(Dense(500, bias_initializer='zeros',
                        kernel_initializer=keras.initializers.lecun_normal()))
        model.add(Activation('selu'))
        model.add(BatchNormalization(axis=1))
        model.add(AlphaDropout(0.25))

    model.add(Dense(500, bias_initializer='zeros',
                    kernel_initializer=keras.initializers.lecun_normal()))
    model.add(Activation('selu'))
    model.add(BatchNormalization(axis=1))
    model.add(AlphaDropout(0.5))

    model.add(Dense(output, bias_initializer='zeros',
                    kernel_initializer=keras.initializers.lecun_normal()))
    if crossentropy:
        model.add(Activation("softmax"))
    else:
        model.add(Activation("sigmoid"))

    model.summary()

    return model


def train_generator_model(train_gen, x_test, y_test, steps, crossentropy=True, relu=True,
                          model=None, batch_size=32, epochs=10, patience=5):
    if model is None:
        if crossentropy:
            if relu:
                model = get_model_relu(12 * 2, 2, crossentropy)
            else:
                model = get_model_selu(12 * 2, 2, crossentropy)
        else:
            if relu:
                model = get_model_relu(12 * 2, 1, crossentropy)
            else:
                model = get_model_selu(12 * 2, 1, crossentropy)

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
    filepath = "../data/model_mask_best.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                 mode='max')
    callbacks_vector.append(checkpoint)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                  patience=patience // 2, min_lr=0.001)
    callbacks_vector.append(reduce_lr)
    history = model.fit_generator(train_gen, shuffle=True, epochs=epochs,
                                  verbose=1, steps_per_epoch=steps // batch_size,
                                  validation_data=(x_test, y_test),
                                  callbacks=callbacks_vector)

    model_json = model.to_json()
    with open("../data/model_mask.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("../data/model_mask.h5")

    return history.history


if __name__ == "__main__":
    x_test, y_test = image_to_package.pre_load_mask(training=False, width=IMAGE_DIMS[1],
                                                    max_img=1000, crossentropy=crossentropy)
    # train_gen = image_to_package.gen_load_mask(training=True, width=IMAGE_DIMS[1],
    #                                            max_type_img=max_training_images,
    #                                            batch_size=batch_size, augumented=False, crossentropy=crossentropy)
    # steps_per_batch = len(image_to_package.get_files(os.path.join(processed_images_path, 'training')))
    #
    # history = train_generator_model(train_gen, x_test, y_test, batch_size=batch_size,
    #                                 epochs=epochs, steps=min(max_training_images_per_class *2, steps_per_batch),
    #                                 patience=patience, crossentropy=crossentropy, relu=relu)
    # pickle.dump(history, open("../data/history_mask.p", "wb"))
    #
    # history = pickle.load(open("../data/history_mask.p", "rb"))
    # print(history)
    # show_plot(history)

    train_gen = image_to_package.gen_load_mask(training=True, width=IMAGE_DIMS[1],
                                               max_type_img=max_training_images_per_class,
                                               batch_size=batch_size, augumented=True,
                                               crossentropy=crossentropy)
    steps_per_batch = len(image_to_package.get_files(os.path.join(processed_images_path, 'training')))

    json_file = open('../data/model_mask.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("../data/model_mask_best.h5")
    history = train_generator_model(train_gen, x_test, y_test, batch_size=batch_size, model=loaded_model,
                                    epochs=epochs, steps=min(max_training_images_per_class * 2, steps_per_batch),
                                    patience=patience, crossentropy=crossentropy, relu=relu)
    pickle.dump(history, open("../data/history_mask_best.p", "wb"))
    #
    # history = pickle.load(open("../data/history_mask_best.p", "rb"))
    # print(history)

    # cv2.namedWindow("Cadru", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Cadru", 720, 480)
    prediction_generator = image_to_package.gen_load_mask(training=False, width=IMAGE_DIMS[1],
                                                          max_type_img=100000,
                                                          batch_size=1)

    count = 0
    try:
        for i, (image, y_test) in enumerate(prediction_generator):
            prediction = image_to_package.get_prediction_mask(loaded_model, image, normalize=False)
            if y_test[0][0] == 1:
                tag = False
            else:
                tag = True
            if prediction == tag:
                print("mask", i, True)
            else:
                count += 1
                print("mask", i, False)
            if i > 4000:
                break
    finally:
        print(count, "wrong")
        print("done")
        K.clear_session()
