from ESRGAN import high_resolution_shape, low_resolution_shape, training_images_path, batch_size, epochs, res_im_path, models_path
from ESRGAN import n_epochs_per_image, testing_images_path, model_to_be_evaluated #, n_images_to_be_evaluated
from ESRGAN import generator, discriminator, vgg, adversarial_model
from ESRGAN import save_images, input_pipeline, train, evaluate, generate

mode = "train"

image_name = "images.jpg"

if mode == 'train':
    train(high_resolution_shape, low_resolution_shape, training_images_path, batch_size, epochs, res_im_path, models_path, 
          n_epochs_per_image, generator, discriminator, vgg, adversarial_model, save_images, input_pipeline)
    
if mode == 'evaluate':
    evaluate(high_resolution_shape, low_resolution_shape, batch_size, models_path, 
          generator, save_images, input_pipeline, testing_images_path, model_to_be_evaluated)
    
if mode == 'generate':
    generate(models_path, generator, model_to_be_evaluated, image_name)