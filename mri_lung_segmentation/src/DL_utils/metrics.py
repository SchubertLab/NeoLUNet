from __future__ import print_function
import numpy as np
from keras.losses import *
from keras.losses import categorical_crossentropy as cct

def dice_coef(image, prediction):
    image_f = K.flatten(image)
    y_pred = K.cast(prediction, 'float32')
    prediction_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = image_f * prediction_f
    score = 2. * K.sum(intersection) / (K.sum(image_f) + K.sum(prediction_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    image_f = K.flatten(y_true)
    prediction_f = K.flatten(y_pred)
    intersection = image_f * prediction_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(K.square(image_f),-1) + K.sum(K.square(prediction_f),-1) + smooth)
    return 1. - score

def dice_coef_loss_crossentropy(image, prediction):
    total_loss = dice_loss(image, prediction) + cct(image, prediction)
    return total_loss

def softmax(X):
    X = np.array(X)
    exps = np.exp(X-np.max(X))
    exps = exps /np.sum(exps)
    return exps

def DW_crossentropy(image, prediction):
    m = prediction.shape[0]
    p = softmax(image)
    log_likelihood = -np.log(p[range(m)],prediction)
    loss = np.sum(log_likelihood) / m
    return loss


def total_variation_loss(prediction):
    if K.ndim(prediction) == 5:
        img_nrows = prediction.shape[1]
        img_ncols = prediction.shape[2]
        if K.image_data_format() == 'channels_first':
            print("Total Variation Loss Channel first")
            a = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :, :img_nrows - 1, 1:, :])
        else:
            print("Total Variation Loss Channel last")
            a = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :, :] - prediction[:, 1:, :img_ncols - 1, :, :])
            b = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :, :] - prediction[:, :img_nrows - 1, 1:, :, :])
        return K.sum(K.pow(a + b, 1.25))
    if K.ndim(prediction) == 4:
        img_nrows = prediction.shape[1]
        img_ncols = prediction.shape[2]
        if K.image_data_format() == 'channels_first':
            print("Total Variation Loss Channel first")
            a = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1] - prediction[:, :, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1] - prediction[:, :, :img_nrows - 1, 1:])
        else:
            print("Total Variation Loss Channel last")
            a = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

''' Combine the total variation loss with the mean squarred error loss. The tot_var should be 2% of the mse loss
according to https://arxiv.org/abs/1803.11293 '''

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def total_variaton_loss_mse(image, prediction):
    tot_var = total_variation_loss(prediction)
    mse = mean_squared_error(image, prediction)
    total_loss = mse + (0.01 * mse * tot_var)
    return total_loss
