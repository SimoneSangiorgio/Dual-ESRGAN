import numpy as np
import matplotlib.pyplot as plt

def save_images(data_path , lowres , highres , orig):
  lowres = np.squeeze( (lowres.numpy() + 1)/2.0 )
  highres = np.squeeze( (highres + 1)/2.0 )
  orig = np.squeeze( (orig.numpy() + 1)/2.0 )

  fig = plt.figure(figsize=(12 , 4))

  ax = fig.add_subplot(1, 3, 1)
  ax.imshow(lowres)
  ax.axis("off")
  ax.set_title("Low-resolution")

  ax = fig.add_subplot(1, 3, 2)
  ax.imshow(orig)
  ax.axis("off")
  ax.set_title("Original")

  ax = fig.add_subplot(1, 3, 3)
  ax.imshow(highres)
  ax.axis("off")
  ax.set_title("Generated")

  plt.savefig(data_path)
