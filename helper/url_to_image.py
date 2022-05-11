import numpy as np
from skimage import io
import tensorflow as tf

def url_to_image(URL, img_size = 320, int8 = False):
    img = io.imread(URL)
    im = tf.image.resize(img, (img_size, img_size))
    im = np.expand_dims(im, axis=0).astype(np.int8 if int8 else np.float32)/255.0
    return img, im
