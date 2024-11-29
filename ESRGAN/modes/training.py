import numpy as np

def train(high_resolution_shape, low_resolution_shape, training_images_path, batch_size, epochs, res_im_path, models_path, 
          n_epochs_per_image, generator, discriminator, vgg, adversarial_model, save_images, input_pipeline):


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
            generator.save_weights(models_path.joinpath("generator_ESRGAN{}.h5".format(epoch+1)))
            discriminator.save_weights(models_path.joinpath("discriminator_ESRGAN{}.h5".format(epoch+1)))

