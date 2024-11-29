
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from pathlib import Path

def input_pipeline(data_path , batch_size , highres_shape , lowres_shape):

  data_path = Path(data_path)
  all_images = list(data_path.glob("*"))
  cntall = len(all_images)

  def gen():
    while True:
      
      all_highres = []
      all_lowres = []
      
      idxes = np.random.choice(cntall , batch_size , replace = False)
      for idx in idxes:

        fname = all_images[idx]

        orig = Image.open(fname)

        # Data augmentation
        if np.random.random() < 0.5:
          # Adjust brightness
          enhancer = ImageEnhance.Brightness(orig)
          orig = enhancer.enhance(np.random.uniform(0.7, 1.3))

        if np.random.random() < 0.5:
          # Rotate image
          orig = orig.rotate(np.random.uniform(-15, 15))

        if np.random.random() < 0.5:
          # Zoom image
          x_center = orig.width / 2
          y_center = orig.height / 2
          width = orig.width * np.random.uniform(0.9, 1.1)
          height = orig.height * np.random.uniform(0.9, 1.1)
          left = x_center - width / 2
          top = y_center - height / 2
          right = x_center + width / 2
          bottom = y_center + height / 2
          orig = orig.crop((left, top, right, bottom))

        high_img = orig.resize(highres_shape , resample=Image.BICUBIC)
        low_img = orig.resize(lowres_shape , resample=Image.BICUBIC)

        if np.random.random() < 0.5:
          high_img = ImageOps.mirror(high_img)
          low_img = ImageOps.mirror(low_img)

        all_highres.append(np.asarray(high_img , dtype = np.float32))
        all_lowres.append(np.asarray(low_img , dtype = np.float32))

        high_res_ret = np.array(all_highres)/127.5 - 1
        low_res_ret = np.array(all_lowres)/127.5 - 1

      yield (high_res_ret , low_res_ret)

  return tf.data.Dataset.from_generator(gen , (tf.float32 , tf.float32)).prefetch(5)
