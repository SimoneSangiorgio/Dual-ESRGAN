from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR.parent / "generation"


def generate(models_path, generator, model_to_be_evaluated, image_name):

  generator.load_weights(os.path.join(models_path , model_to_be_evaluated))

  def divide_image(input_image_path):
      image = Image.open(input_image_path).convert('RGB')
      width, height = image.size
      print(f"Original Dimensions: {width}x{height}")
      print(f"Upscaled Dimensions: {width*4}x{height*4}")

      # Calculate dimensions with padding
      padded_width = (width + 47) // 48 * 48
      padded_height = (height + 47) // 48 * 48
      padding_pixels_width = padded_width - width
      padding_pixels_height = padded_height - height

      # Create a new image with reflection padding
      padded_image_array = np.pad(image, ((0, padding_pixels_height), (0, padding_pixels_width), (0, 0)), mode='edge')
      padded_image = Image.fromarray(padded_image_array)


      # Divides the image into 62x62 batches and applies upscaling
      pieces = []
      for i in range(0, padded_height, 48):
          for j in range(0, padded_width, 48):
              box = (j, i, j+48, i+48)
              piece = padded_image.crop(box)
              piece = np.pad(piece, ((8, 8), (8, 8), (0, 0)), mode='edge')                
              upscaled_piece = np.asarray(piece, dtype=np.float32) / 127.5 - 1
              upscaled_piece = generator.predict(np.expand_dims(upscaled_piece, axis=0))
              upscaled_piece = np.squeeze(upscaled_piece, axis=0)    
              upscaled_piece = Image.fromarray(((upscaled_piece + 1) * 127.5).astype('uint8')) 
              upscaled_piece = upscaled_piece.crop((32, 32, 224, 224))
              upscaled_piece = np.asarray(upscaled_piece, dtype=np.float32) / 127.5 - 1


              pieces.append(upscaled_piece)

      return pieces, width, height


  def reassemble_images(pieces, original_width, original_height):
      # Calculates the size of the final image
      image_width = pieces[0].shape[1] 
      image_height = pieces[0].shape[0]
      num_images = len(pieces)
      num_images_per_row = (original_width + 47) // 48
      total_width = image_width * num_images_per_row
      total_height = image_height * (num_images // num_images_per_row)

      # Create a new image to contain all images
      new_image = Image.new('RGB', (total_width, total_height))

      # Paste each image in the right place
      for index, piece in enumerate(pieces):
          x = (index % num_images_per_row) * image_width
          y = (index // num_images_per_row) * image_height
          piece = Image.fromarray(((piece + 1) * 127.5).astype('uint8')) 
          new_image.paste(piece, (x, y))

      # Removes padding
      new_image = new_image.crop((0, 0, original_width*4, original_height*4))

      # shows the final image
      new_image.show()

      # save the final image
      new_image.save(EVALUATION_DIR / "upscaled_images" / image_name)


  pieces, original_width, original_height = divide_image(EVALUATION_DIR / "images_to_upscale" / image_name)
  reassemble_images(pieces, original_width, original_height)




