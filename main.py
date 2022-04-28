import cv2
import json
import requests
import numpy    as np
from skimage    import io
from flask      import Flask, request
from flask_cors import CORS

# CLASS IMPORTS
from YOLO import Yolo

CLASSES = [ "Bird", "Cat", "Dog", "Flower", "Face" ]  # class names

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
    int8 = request.form['int8']
    int8 = bool(int8)
    img, im = transform(URL)

    MODEL_PATH = 'tflite_models/custom_int800.tflite' if int8 else 'tflite_models/custom01.tflite'
    YOLO = Yolo(model_path = MODEL_PATH, CLASSES = CLASSES, int8 = int8)

    H = img.shape[0]
    W = img.shape[1]
    YOLO.pred(im)

    scores = YOLO.YOLOdetect()
    bbox = YOLO.return_bbox(scores)
    data = YOLO.return_results(scores, bbox, H, W)

    del YOLO, bbox, scores

    return data

@app.route('/')
def home():
    return {'data': 'server is up and running.'}