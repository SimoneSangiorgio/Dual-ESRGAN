import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Add, LeakyReLU, PReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization

filters = 64
kernel_size = 3
strides = 1
padding = "same"
momentum = 0.8
beta = 0.2
scale = 2 # upsampling 2x

def dense_block(input):

    den1 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    den1 = LeakyReLU(beta)(den1)
    den1 = Add()([input, den1])

    den2 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(den1)
    den2 = LeakyReLU(beta)(den2)
    den2 = Add()([input, den1, den2])

    den3 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(den2)
    den3 = LeakyReLU(beta)(den3)
    den3 = Add()([input, den1, den2, den3])

    den4 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(den3)
    den4 = LeakyReLU(beta)(den4)
    den4 = Add()([input, den1, den2, den3, den4])  

    den5 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(den4)
    den5 = Lambda(lambda x: x * beta)(den5)
    den = Add()([input, den5])
    
    return den


def residual_in_residual_block(input):

    res = dense_block(input)
    res = dense_block(res)
    res = dense_block(res)
    res = dense_block(res)
    res = Lambda(lambda x: x * beta)(res)
    res = Add()([res, input])    
    return res

def sub_pixel_layer(scale):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale))

def upsample(input):
    x = Conv2D(filters=filters*4, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    x = sub_pixel_layer(scale)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x

'''def vision_transformer_encoder(inputs):

    num_heads=8
    mlp_dim=2048
    num_layers=6

    x = inputs
    for _ in range(num_layers):
        # Layer Normalization
        x1 = LayerNormalization(epsilon=1e-6)(x)
        # Multi-Head Self-Attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(x1, x1)
        # Residual Connection
        x2 = x + attention_output
        # Layer Normalization
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP Block
        mlp_output = Dense(mlp_dim, activation='relu')(x3)
        mlp_output = Dense(inputs.shape[-1])(mlp_output)
        # Residual Connection
        x = x2 + mlp_output
    return x'''


def build_generator():

    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # Add the pre-residual block
    gen1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_layer)
    gen1 = LeakyReLU(beta)(gen1)

    # vision transformer encoder
    #trans = vision_transformer_encoder(gen1)

    # residual in residual blocks
    res = residual_in_residual_block(gen1)

    # Add the post-residual block
    gen2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(res)
    gen2 = Lambda(lambda x: x * beta)(gen2)

    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])
    #gen3 = Add()([gen3, trans])

    # Add an upsampling block
    gen4 = upsample(gen3)

    # Add another upsampling block
    gen5 = upsample(gen4) 

    # Output convolution layer
    gen6 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(gen5)
    gen6 = LeakyReLU(beta)(gen6)

    # Output convolution layer
    output = Conv2D(filters=3, kernel_size=kernel_size, strides=strides, padding=padding, activation='tanh')(gen6)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model