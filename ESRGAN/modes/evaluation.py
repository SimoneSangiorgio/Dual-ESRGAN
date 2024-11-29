import os
import numpy as np
import cv2

from pathlib import Path

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR.parent / "evaluation"

def evaluate(high_resolution_shape, low_resolution_shape, batch_size, models_path, 
          generator, save_images, input_pipeline, testing_images_path, model_to_be_evaluated):

  generator.load_weights(os.path.join(models_path , model_to_be_evaluated))

  dataloader = iter(input_pipeline(testing_images_path , batch_size , high_resolution_shape[:2] , low_resolution_shape[:2]))

  for i in range(1):
      high_resolution_images, low_resolution_images = next(dataloader)
      generated_images = generator.predict_on_batch(low_resolution_images)
      
      save_images(EVALUATION_DIR.joinpath("evaluation_images","img{}.jpeg".format(i)), low_resolution_images[0], generated_images[0] , high_resolution_images[0])

      upscaled_images_bilinear = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
      upscaled_images_bicubic = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
      upscaled_image_lanczos = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

      save_images(EVALUATION_DIR.joinpath("evaluation_images","img{}_bilinear.jpeg".format(i)), low_resolution_images[0], upscaled_images_bilinear, high_resolution_images[0])
      save_images(EVALUATION_DIR.joinpath("evaluation_images","img{}_bicubic.jpeg".format(i)), low_resolution_images[0], upscaled_images_bicubic, high_resolution_images[0])
      save_images(EVALUATION_DIR.joinpath("evaluation_images","img{}_lanczos.jpeg".format(i)), low_resolution_images[0], upscaled_image_lanczos, high_resolution_images[0])

      high_resolution_images_resized = cv2.resize(np.array(high_resolution_images[0]), (upscaled_images_bilinear.shape[1], upscaled_images_bilinear.shape[0]))
      
      psnr_value_bilinear = psnr(high_resolution_images_resized, np.array(upscaled_images_bilinear))
      mse_value_bilinear = mse(high_resolution_images_resized, np.array(upscaled_images_bilinear))

      psnr_value_bicubic = psnr(high_resolution_images_resized, np.array(upscaled_images_bicubic))
      mse_value_bicubic = mse(high_resolution_images_resized, np.array(upscaled_images_bicubic))

      psnr_value_lanczos = psnr(high_resolution_images_resized, np.array(upscaled_image_lanczos))
      mse_value_lanczos = mse(high_resolution_images_resized, np.array(upscaled_image_lanczos))

      psnr_value = psnr(np.array(high_resolution_images), np.array(generated_images))    
      mse_value = mse(np.array(high_resolution_images), np.array(generated_images))

      print(f"SRGAN - PSNR: {psnr_value}, MSE: {mse_value}")
      print(f"Bilinear - PSNR: {psnr_value_bilinear}, MSE: {mse_value_bilinear}")
      print(f"Bicubic - PSNR: {psnr_value_bicubic}, MSE: {mse_value_bicubic}")
      print(f"Lanczos - PSNR: {psnr_value_lanczos}, MSE: {mse_value_lanczos}")


  
  psnr_total = 0
  mse_total = 0
  psnr_total_bilinear = 0
  mse_total_bilinear = 0
  psnr_total_bicubic = 0
  mse_total_bicubic = 0
  psnr_total_lanczos = 0
  mse_total_lanczos = 0
  n_images_to_be_evaluated = 50

  

  for i in range(n_images_to_be_evaluated):
      high_resolution_images, low_resolution_images = next(dataloader)
      generated_images = generator.predict_on_batch(low_resolution_images)
      upscaled_images_bilinear = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
      upscaled_images_bicubic = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
      upscaled_image_lanczos = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

      high_resolution_images_resized = cv2.resize(np.array(high_resolution_images[0]), (upscaled_images_bilinear.shape[1], upscaled_images_bilinear.shape[0]))
      
      psnr_value_bilinear = psnr(high_resolution_images_resized, np.array(upscaled_images_bilinear))
      mse_value_bilinear = mse(high_resolution_images_resized, np.array(upscaled_images_bilinear))

      psnr_total_bilinear += psnr_value_bilinear
      mse_total_bilinear += mse_value_bilinear

      psnr_value_bicubic = psnr(high_resolution_images_resized, np.array(upscaled_images_bicubic))
      mse_value_bicubic = mse(high_resolution_images_resized, np.array(upscaled_images_bicubic))

      psnr_total_bicubic += psnr_value_bicubic
      mse_total_bicubic += mse_value_bicubic

      psnr_value_lanczos = psnr(high_resolution_images_resized, np.array(upscaled_image_lanczos))
      mse_value_lanczos = mse(high_resolution_images_resized, np.array(upscaled_image_lanczos))

      psnr_total_lanczos += psnr_value_lanczos
      mse_total_lanczos += mse_value_lanczos      

      psnr_value = psnr(np.array(high_resolution_images), np.array(generated_images))
      mse_value = mse(np.array(high_resolution_images), np.array(generated_images))
      
      psnr_total += psnr_value
      mse_total += mse_value

  psnr_average_bilinear = psnr_total_bilinear / n_images_to_be_evaluated
  mse_average_bilinear = mse_total_bilinear / n_images_to_be_evaluated

  psnr_average_bicubic = psnr_total_bicubic / n_images_to_be_evaluated
  mse_average_bicubic = mse_total_bicubic / n_images_to_be_evaluated

  psnr_average_lanczos = psnr_total_lanczos / n_images_to_be_evaluated
  mse_average_lanczos = mse_total_lanczos / n_images_to_be_evaluated

  psnr_average = psnr_total / n_images_to_be_evaluated
  mse_average = mse_total / n_images_to_be_evaluated

  print(f"SRGAN - Average PSNR: {psnr_average}, Average MSE: {mse_average}")
  print(f"Bilinear - Average PSNR: {psnr_average_bilinear}, Average MSE: {mse_average_bilinear}")
  print(f"Bicubic - Average PSNR: {psnr_average_bicubic}, Average MSE: {mse_average_bicubic}")
  print(f"Lanczos - Average PSNR: {psnr_average_lanczos}, Average MSE: {mse_average_lanczos}")