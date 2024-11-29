from tensorflow.keras.applications import VGG19
from tensorflow import keras

def build_vgg():

    input_shape = (256, 256, 3)

    vgg = VGG19(include_top = False ,  input_shape = input_shape , weights="imagenet")
    features = vgg.get_layer(index = 9).output
    
    model = keras.Model(inputs=[vgg.inputs], outputs=[features])
    return model