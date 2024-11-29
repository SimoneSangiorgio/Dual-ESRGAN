from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model

filters = 64
kernel_size = 3
strides = 1
padding = "same"
momentum = 0.8
beta = 0.2


def build_discriminator():

    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)

    # Add the first convolution block
    dis1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding=padding)(input_layer)
    dis1 = LeakyReLU(beta)(dis1)

    # Add the 2nd convolution block
    dis2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding=padding)(dis1)
    dis2 = LeakyReLU(beta)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    # Add the third convolution block
    dis3 = Conv2D(filters=filters*2, kernel_size=kernel_size, strides=1, padding=padding)(dis2)
    dis3 = LeakyReLU(beta)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    # Add the fourth convolution block
    dis4 = Conv2D(filters=filters*2, kernel_size=kernel_size, strides=2, padding=padding)(dis3)
    dis4 = LeakyReLU(beta)(dis4)
    dis4 = BatchNormalization(momentum=momentum)(dis4)

    # Add the fifth convolution block
    dis5 = Conv2D(filters=filters*4, kernel_size=kernel_size, strides=1, padding=padding)(dis4)
    dis5 = LeakyReLU(beta)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    # Add the sixth convolution block
    dis6 = Conv2D(filters=filters*4, kernel_size=kernel_size, strides=2, padding=padding)(dis5)
    dis6 = LeakyReLU(beta)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    # Add the seventh convolution block
    dis7 = Conv2D(filters=filters*8, kernel_size=kernel_size, strides=1, padding=padding)(dis6)
    dis7 = LeakyReLU(beta)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    # Add the eight convolution block
    dis8 = Conv2D(filters=filters*8, kernel_size=kernel_size, strides=2, padding=padding)(dis7)
    dis8 = LeakyReLU(beta)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)

    # Add a dense layer
    dis9 = Dense(units=filters*16)(dis8)
    dis9 = LeakyReLU(beta)(dis9)

    # Last dense layer - for classification
    output = Dropout(0.4)(dis9)
    output = Dense(units=1, activation='sigmoid')(output)

    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    return model