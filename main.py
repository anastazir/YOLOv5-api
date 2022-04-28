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

def transform(URL, img_size = 320):
    img = io.imread(URL)
    im = cv2.resize(img, (img_size, img_size)).astype(np.float32)
    im = np.expand_dims(im, axis=0)/255.0
    return img, im

@app.route('/urlRoute', methods=['POST'])
def urlRoute():

    URL = request.form['url']
    YOLO = Yolo(model_path = 'tflite_models/custom01.tflite', CLASSES = CLASSES)
    img, im = transform(URL)

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