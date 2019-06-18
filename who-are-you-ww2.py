import numpy as np
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import cv2

# Data Load
DATASET_DIR = "./WW2_Dataset/"

df = pd.read_csv(DATASET_DIR + "Names")
y_names = np.array(df)
y_names = y_names.flatten()
print(y_names)

l_names = np.copy(y_names)
for i in range(len(l_names)):
    l_names[i] = l_names[i].replace(" ", "")
print(l_names)

y_full_result_images = []
for i in range(len(l_names)):
    print(DATASET_DIR + l_names[i] + '.jpg')
    img = cv2.imread(DATASET_DIR + l_names[i] + '.jpg', 0)
    img = img.astype(float)
    for k in range(len(img)):
        for l in range(len(img[k])):
            img[k][l] = float(img[k][l]) / 255.0
    # plt.imshow(img)
    # plt.show()
    y_full_result_images.append(img)
print("len result images :", len(y_full_result_images), y_full_result_images[0].shape)

yyy_train = np.arange(1, 18)
y_train = []
for i in range(len(yyy_train)):
    yy_train = [0] * 17
    yy_train[i] = 1
    y_train.append(yy_train)
y_train = np.array(y_train)
print(y_train, len(y_train))

ll = y_names.shape[0]
print("Nb train images: ", ll)

x_train = []

for i in range(1, ll + 1):
    img = cv2.imread(DATASET_DIR + str(1000 + i) + '.jpg', 0)
    img = img.astype(float)
    for k in range(len(img)):
        for l in range(len(img[k])):
            img[k][l] = float(img[k][l]) / 255.0
    x_train.append(img)
# Data Prep

print("len image train: ", len(x_train))

x_train = np.array(x_train)

print("x_train shape ", x_train.shape)

x_train = x_train.reshape((17, (160 * 100)))

print("x_train shape ", x_train.shape)
print("pix :", x_train[0][23])


# Generate Encoder


def gen_auto_encoder(num_pixels):
    """
    Generate Encoder
    :param num_pixels:
    :return:
    """
    encoding_dim = 64
    auto_encoder = Sequential()
    auto_encoder.add(Dense(encoding_dim, input_dim=num_pixels, activation='relu'))
    auto_encoder.add(Dense(17, activation='softmax'))
    auto_encoder.summary()
    return auto_encoder


# Display

def gen_display(encoded_imgs, decoded_imgs, x_test, i):
    plt.figure(figsize=(18, 4))
    ax = plt.subplot(3, 1, 1)
    plt.imshow(x_test[i].reshape(160, 100))
    ax = plt.subplot(3, 1, 2)
    plt.imshow(encoded_imgs[i].reshape(8, 8))
    ax = plt.subplot(3, 1, 3)
    plt.imshow(decoded_imgs[i].reshape(160, 100))
    plt.show()


def gen_result(encoder, image_input, result_imgs=None):
    res = encoder.predict(image_input)
    # print(res)
    result = np.where(res == np.amax(res))
    print(np.amax(res), result, result[1])
    real_res = result[1][0]
    # if isinstance(result, tuple):
    print("Result", y_names[real_res], real_res)
    if result_imgs is not None:
        plt.figure(figsize=(6, 4))
        ax = plt.subplot(1, 2, 1)
        plt.imshow(image_input.reshape(160, 100))
        plt.gray()
        ax = plt.subplot(1, 2, 2)
        r_img = result_imgs[real_res]
        if r_img.shape:
            print(r_img.shape)
        plt.imshow(r_img)
        plt.gray()
        plt.show()


def encoder_decoder_res(auto_encoder, x_train):
    input_img = Input(shape=(160 * 100,))
    encoder_layer = auto_encoder.layers[0]
    encoder = Model(input_img, encoder_layer(input_img))
    encoder.summary()
    encoded_imgs = encoder.predict(x_train)
    decoded_imgs = auto_encoder.predict(x_train)
    return encoded_imgs, decoded_imgs


# Training
def train(x_train, y_train, res_imgs=None):
    auto_encoder = gen_auto_encoder(160 * 100)
    auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
    print("train shape: ", x_train.shape, "y train shape: ", y_train.shape)
    auto_encoder.fit(x_train, y_train, epochs=200, batch_size=100, validation_data=(x_train, y_train))
    # Scores
    scores = auto_encoder.evaluate(x_train, y_train)
    print("Scores: ", scores)
    return auto_encoder


def test_result(auto_encoder, x, res_imgs):
    # gen_display(encoded_imgs, decoded_imgs, x_train, 10)
    for i in range(17):
        gen_result(auto_encoder, np.array([x_train[i]]), res_imgs)


auto_encode = train(x_train, y_train, y_full_result_images)


# Save Training

# Usage

def process_160x100(fname):
    use_case = fname
    use_case_list = []
    img = cv2.imread(DATASET_DIR + use_case + '.jpg', 0)
    img = img.astype(float)
    for k in range(len(img)):
        for l in range(len(img[k])):
            img[k][l] = float(img[k][l]) / 255.0
    use_case_list.append(img)
    arr_use_case_list = np.array(use_case_list)
    print(arr_use_case_list.shape)
    arr_use_case_list = arr_use_case_list.reshape((160 * 100))
    return arr_use_case_list


def predict(auto_encode, array_of_imgs, y_full_result_images):
    gen_result(auto_encode, np.array([array_of_imgs]), y_full_result_images)


predict(auto_encode, process_160x100("Sm160x100"), y_full_result_images)
predict(auto_encode, process_160x100("Marc160x100"), y_full_result_images)
