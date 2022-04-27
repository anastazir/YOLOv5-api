import tensorflow as tf
import numpy as np
import cv2
from skimage import io
import json
import requests
from flask import Flask, request
from flask_cors import CORS

CLASSES = [ "Bird", "Cat", "Dog", "Flower", "Face" ]  # class names

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def pred_model(model_path, im):
    Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()  # allocate
    input_details = interpreter.get_input_details()  # inputs
    output_details = interpreter.get_output_details()  # outputs
    interpreter.set_tensor(input_details[0]['index'], im)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def transform(URL, img_size = 320):
    img = io.imread(URL)
    im = cv2.resize(img, (img_size, img_size)).astype("float32")
    im = np.expand_dims(im, axis=0)/255.0
    return img, im

def classFilter(classdata):
    classes = []                                # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes                              # return classes (int)

def YOLOdetect(output_data):
    output_data = output_data[0]                                            # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])                                # boxes  [25200, 4]
    scores = np.squeeze(output_data[..., 4:5])                              # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:])                             # get classes
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]                     # xywh to xyxy [4, 25200]

    return xyxy, classes, scores

def return_results(xyxy, classes, scores, img, bbox):
    class_names = []
    class_scores = []
    coordinates = []
    for i in bbox:

        H = img.shape[0]
        W = img.shape[1]
        xmin = int(xyxy[0][i] * W)
        ymin = int(xyxy[1][i] * H)
        xmax = int(xyxy[2][i] * W)
        ymax = int(xyxy[3][i] * H)
        if xmin < 0:
            xmin = 0
        if ymin > H - 1:
            ymin = H - 1
        if xmax < 0:
            xmax = 0
        if ymax > H - 1:
            ymax = H - 1

        coordinates.append([xmin,ymin,xmax,ymax])
        class_names.append(CLASSES[classes[i]])
        class_scores.append(int(scores[i]*100))

    return {"data":{"class_names":class_names,
                    "class_scores":class_scores,
                    "coordinates":coordinates}}


@app.route('/urlRoute', methods=['POST'])
def urlRoute():

    URL = request.form['url']
    img, im = transform(URL)
    output_data = pred_model("tflite_models/custom01.tflite", im)
    xyxy, classes, scores = YOLOdetect(output_data)
    bbox = tf.image.non_max_suppression(np.squeeze(output_data[..., :4]), scores, max_output_size=50, iou_threshold=0.05, \
                                        name=None, score_threshold = 0.65)
    bbox = np.array(bbox)
    data = return_results(xyxy, classes, scores, img, bbox)

    return data

@app.route('/')
def home():
    return {'data': 'server is up and running.'}