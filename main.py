import cv2
import json
import requests
import numpy    as np
from skimage    import io
from flask      import Flask, request
from flask_cors import CORS

# CLASS IMPORTS
from YOLO import Yolo

CLASSES1 = [ "Bird", "Cat", "Dog", "Flower", "Face" ]  # class names for model 1
CLASSES2 = ["Insect", "Fish", "Fast_food", "Animal", "Fruit", "Traffic_light", "Vehicle_registration_plate", "Car", "Weapon"]

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def transform(URL, img_size = 320, int8 = False):
    int_type = np.int8 if int8 else np.float32
    img = io.imread(URL)
    im = cv2.resize(img, (img_size, img_size), 3).astype(int_type)
    im = np.expand_dims(im, axis=0)/255.0
    return img, im

@app.route('/urlRoute', methods=['POST'])
def urlRoute():

    URL = request.form['url']

    int8 = True if request.form['int8'] == 'True' else False

    type = 1 if request.form['type'] == '1' else 2

    classes = CLASSES1 if type == 1 else CLASSES2
    print(classes)
    img, im = transform(URL)

    MODEL_PATH = f'tflite_models/custom_int80{type}.tflite' if int8 else f'tflite_models/custom0{type}.tflite'
    YOLO = Yolo(model_path = MODEL_PATH, CLASSES = classes, int8 = int8)

    H = img.shape[0]
    W = img.shape[1]

    YOLO.pred(im)
    YOLO.extract_results()
    YOLO.return_bbox()
    data = YOLO.return_results(H, W)

    del YOLO

    return data

@app.route('/')
def home():
    return {'data': 'server is up and running.'}