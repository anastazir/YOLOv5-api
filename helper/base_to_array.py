from PIL import Image
import numpy as np
from io import BytesIO
import base64
import cv2

def base_to_array(data, img_size = 320, int8 = False):
    """
    Takes the request as a parameter, then finds the base64 string attached to it and
    converts it into a numpy array.
    """
    if data["base64"]:
        base64str= data["base64"]
        imageDecoded = Image.open(BytesIO(base64.b64decode(base64str)))
        numpydata = np.asarray(imageDecoded)
        im = cv2.resize(numpydata, (img_size, img_size), 3).astype(np.int8 if int8 else np.float32)
        im = np.expand_dims(im, axis=0)/255.0
        return numpydata, im