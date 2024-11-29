import numpy as np
from tensorflow.keras import Input, Model

from config import common_optimizer, high_resolution_shape, low_resolution_shape, training_images_path, batch_size, epochs, res_im_path, models_path, n_epochs_per_image

from utils import save_images
from utils import input_pipeline  

from models import build_generator
from models import build_discriminator
from models import build_vgg


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


mode = "train"


if mode == 'train':
    def train():

        dataloader = iter(input_pipeline(training_images_path , batch_size , high_resolution_shape[:2] , low_resolution_shape[:2]))

        for epoch in range(epochs):

            """Train the discriminator network"""

            # Sample a batch of images
            high_resolution_images, low_resolution_images = next(dataloader)

            # Generate high-resolution images from low-resolution images
            generated_high_resolution_images = generator.predict(low_resolution_images)

            # Generate batch of real and fake labels
            real_labels = np.ones((batch_size, 16, 16, 1))
            fake_labels = np.zeros((batch_size, 16, 16, 1))

            # Train the discriminator network on real and fake images
            d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)

            d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)


            # Calculate total discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            """Train the generator network"""

            # Sample a batch of images
            high_resolution_images, low_resolution_images = next(dataloader)

            # Extract feature maps for real high-resolution images
            image_features = vgg.predict(high_resolution_images)

            # Train the generator network
            g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],[real_labels, image_features])

            print("Epoch {} : g_loss: {} , d_loss: {}".format(epoch+1 , g_loss[0] , d_loss[0]))

            # Save image of first epoch
            if (epoch+1) == 1:
                high_resolution_images, low_resolution_images = next(dataloader)

                # Normalize image
                generated_images = generator.predict_on_batch(low_resolution_images)

                for index, img in enumerate(generated_images):
                    save_images(res_im_path.joinpath("img_{}_{}".format(epoch+1, index)),low_resolution_images[index], generated_images[index] , high_resolution_images[index])

            # Sample and save images after every n epochs
            if (epoch+1) % n_epochs_per_image == 0:
                high_resolution_images, low_resolution_images = next(dataloader)

                # Normalize images
                generated_images = generator.predict_on_batch(low_resolution_images)

                for index, img in enumerate(generated_images):
                    save_images(res_im_path.joinpath("img_{}_{}".format(epoch+1, index)),low_resolution_images[index], generated_images[index] , high_resolution_images[index])

                # Save models
                generator.save_weights(models_path.joinpath("generator_{}.h5".format(epoch+1)))
                discriminator.save_weights(models_path.joinpath("discriminator_{}.h5".format(epoch+1)))
    train()



