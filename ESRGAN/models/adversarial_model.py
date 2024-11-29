from tensorflow.keras import Input, Model
import sys
from pathlib import Path
from tensorflow.keras.optimizers import Adam

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import high_resolution_shape, low_resolution_shape
from models import build_generator, build_discriminator, build_vgg

common_optimizer = Adam(0.0002, 0.5)

# Build and compile the VGG19
vgg = build_vgg()
vgg.trainable = False
vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

#Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

# Build the generator network
generator = build_generator()

"""Build and compile the adversarial model"""

# Input layers for high-resolution and low-resolution images
input_high_resolution = Input(shape=high_resolution_shape)
input_low_resolution = Input(shape=low_resolution_shape)

# Generate high-resolution images from low-resolution images
generated_high_resolution_images = generator(input_low_resolution)

# Extract feature maps of the generated images
features = vgg(generated_high_resolution_images)

# Get the probability of generated high-resolution images
probs = discriminator(generated_high_resolution_images)

# Create an adversarial model
adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])

# Get the list of trainable variables
variables = adversarial_model.trainable_variables

# Build the optimizer with the list of trainable variables
common_optimizer.build(variables)

# Compile the adversarial model
adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer)
