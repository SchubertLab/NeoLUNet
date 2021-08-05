from keras.models import *
from keras.layers import *
from keras.optimizers import *
import numpy as np
from keras.losses import *
from DL_utils import metrics

def unet2D(pretrained_weights,input_size, num_channels, loss_function, base_n_filters = 64, learning_rate=0.001):
    print("started UNet")
    inputs = Input(input_size)
    conv1 = Conv2D(base_n_filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    print("finished 1. convolution")
    conv1 = Conv2D(base_n_filters, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    drop1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Conv2D(base_n_filters*2, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(base_n_filters*2, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(base_n_filters*4, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(base_n_filters*4, 3,  padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    drop3 = Dropout(0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(base_n_filters*8, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(base_n_filters*8, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(base_n_filters*16, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(base_n_filters*16, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.1)(conv5)

    up6 = Conv2D(base_n_filters*8, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(base_n_filters*8, 3,activation = 'relu',  padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(base_n_filters*8, 3, activation = 'relu',  padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(base_n_filters*4, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([drop3, up7], axis=3)
    conv7 = Conv2D(base_n_filters*4, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(base_n_filters*4, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(base_n_filters*2, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([drop2, up8], axis=3)
    conv8 = Conv2D(base_n_filters*2, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(base_n_filters*2, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(base_n_filters, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([drop1, up9], axis=3)

    conv9 = Conv2D(base_n_filters, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(base_n_filters, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(3, 3,padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9) #Changed activiation from Relu to linear
    
    if loss_function == "dice_loss":
        loss_function = eval(loss_function)
    model = Model(inputs, conv10)  # model = Model(input=inputs, output=conv10)
    print("shape input UNet:", np.shape(inputs))
    print("shape output UNet:", np.shape(conv10))
    model.compile(optimizer=Adam(lr = learning_rate), loss = loss_function, metrics=['mae'])
    #To Do: Try beta_1 value of 0.5 as in ocemoglus paper

    '''model.compile(optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics=['accuracy'])
    Other possible losses = 
    - binary_crossentropy 
    - categorical_crossentropy
    - sparse_categorical_crossentropy
    - mean_squared_error 
    - dice_coef_loss
    - total_variaton_loss_mse
    - mean_absolute_error
    - mean_squared_logarithmic_error

    metrics = 
    - 'accuracy' 
    - dice_coef '''

    #model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
