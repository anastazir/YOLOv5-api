import tensorflow as tf
import numpy as np

class Yolo:
    def __init__(self, model_path, CLASSES):
        self.CLASSES = CLASSES
        Interpreter = tf.lite.Interpreter
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def pred(self, im):
        self.interpreter.set_tensor(self.input_details[0]['index'], im)
        self.interpreter.invoke()
        self.output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

    def classFilter(self, classdata):
        classes = []
        for i in range(classdata.shape[0]):
            classes.append(classdata[i].argmax())
        return classes    

    def YOLOdetect(self):
        output_data = self.output_data[0]
        boxes = np.squeeze(output_data[..., :4])
        scores = np.squeeze(output_data[..., 4:5])
        classes = self.classFilter(output_data[..., 5:])
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        self.xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        self.classes = classes
        return scores

    def return_results(self, scores, bbox, H, W):
        class_names = []
        class_scores = []
        coordinates = []
        for i in bbox:
            xmin = int(self.xyxy[0][i] * W)
            ymin = int(self.xyxy[1][i] * H)
            xmax = int(self.xyxy[2][i] * W)
            ymax = int(self.xyxy[3][i] * H)
            if xmin < 0:
                xmin = 0
            if ymin > H - 1:
                ymin = H - 1
            if xmax < 0:
                xmax = 0
            if ymax > H - 1:
                ymax = H - 1

            coordinates.append([xmin, ymin, xmax, ymax])
            class_names.append(self.CLASSES[self.classes[i]])
            class_scores.append(int(scores[i] * 100))

        return {"data": {"class_names": class_names,
                         "class_scores": class_scores,
                         "coordinates": coordinates}}

    def return_bbox(self, scores, score_threshold = 0.65, max_size = 10, iou_threshold = 0.5):
        bbox = tf.image.non_max_suppression(np.squeeze(self.output_data[..., :4]), scores, max_output_size=max_size, \
               iou_threshold=iou_threshold, score_threshold = score_threshold)
        return np.array(bbox)