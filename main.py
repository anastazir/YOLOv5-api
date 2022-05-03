import cv2
import json
import requests
from skimage    import io
from flask      import Flask, request
from flask_cors import CORS

# CLASS IMPORTS
from YOLO import Yolo

# FUNCTION IMPORTS
from helper.base_to_array import base_to_array
from helper.url_to_image import url_to_image

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

CLASSES1 = [ "Bird", "Cat", "Dog", "Flower", "Face" ]  # class names for model 1
CLASSES2 = ["Insect", "Fish", "Fast_food", "Animal", "Fruit", "Traffic_light", "Vehicle_registration_plate", "Car", "Weapon"]

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/fileRoute', methods=['POST'])
def fileRouter():
    if request.method != 'POST':
        return {"The request must be a POST request"}

    try:
        data = json.loads(request.data) # for React

    except:
        data = request.form  # for postman

    if not data['base64']:

        return  {'data': 'unable to  read file'}

    int8 = True if data['int8'] == 'True' else False
    type = 1 if data['type'] == '1' else 2

    classes = CLASSES1 if type == 1 else CLASSES2

    img, im = base_to_array(data, int8=int8)

    MODEL_PATH = f'tflite_models/custom_int80{type}.tflite' if int8 else f'tflite_models/custom0{type}.tflite'
    YOLO = Yolo(model_path = MODEL_PATH, CLASSES = classes, int8 = int8)

    H = img.shape[0]
    W = img.shape[1]

    YOLO.pred(im)
    YOLO.extract_results()
    YOLO.return_bbox(iou_threshold = 0.0)
    data = YOLO.return_results(H, W)

    del YOLO

    return data


@app.route('/urlRoute', methods=['POST'])
def urlRoute():

    try:
        data = json.loads(request.data) # for React

    except:
        data = request.form  # for postman

    URL = data['url']

    int8 = True if data['int8'] == 'True' else False

    type = 1 if data['type'] == '1' else 2

    classes = CLASSES1 if type == 1 else CLASSES2


    img, im = url_to_image(URL)

    MODEL_PATH = f'tflite_models/custom_int80{type}.tflite' if int8 else f'tflite_models/custom0{type}.tflite'
    YOLO = Yolo(model_path = MODEL_PATH, CLASSES = classes, int8 = int8)

    H = img.shape[0]
    W = img.shape[1]

    YOLO.pred(im)
    YOLO.extract_results()
    YOLO.return_bbox(iou_threshold = 0.0)
    data = YOLO.return_results(H, W)

    del YOLO

    return data

@app.route('/')
def home():
    return {'data': 'server is up and running.'}