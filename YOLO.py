import tensorflow as tf
import numpy as np

class Yolo:
    def __init__(self, model_path, CLASSES, int8 = False):
        self.int8 = int8
        self.CLASSES = CLASSES
        Interpreter = tf.lite.Interpreter
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def pred(self, im):
        '''
        im: image in the form of normalized numpy array
        '''
        input, output = self.input_details[0], self.output_details[0]
        if self.int8:
            scale, zero_point = input['quantization']
            im = (im / scale + zero_point).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], im)
        self.interpreter.invoke()
        self.output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        if self.int8:
            scale, zero_point = output['quantization']
            self.output_data = (self.output_data.astype(np.float32) - zero_point) * scale  # re-scale

    def classFilter(self, classdata):
        '''
        returns the index of the class with highest probability score
        '''
        classes = []
        for i in range(classdata.shape[0]):
            classes.append(classdata[i].argmax())
        return classes    

    def YOLOdetect(self):
        '''
        extract scores, class probabilities and bounding boxes from the output tensor
        '''
        output_data = self.output_data[0]
        boxes = np.squeeze(output_data[..., :4]) # the first 4 elements are contain box coordinates
        scores = np.squeeze(output_data[..., 4:5]) # the 5th element is the confidence score
        self.classes = self.classFilter(output_data[..., 5:]) # the remaining elements are class probabilities
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] # extract the coordinates of the bounding boxes
        self.xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2] # convert xywh to xyxy
        return scores

    def return_results(self, scores, bbox, H, W):
        ''''
        return the results in the form of json
        '''
        class_names = []
        class_scores = []
        coordinates = []
        final = []
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

        for coordinate, class_name, class_score in zip(coordinates, class_names, class_scores):
            final.append(
                [
                    coordinate,
                    class_score,
                    class_name,
                ]
            )

        return {"class_names": class_names,
                "class_scores": class_scores,
                "coordinates": coordinates,
                "final": final}

    def return_bbox(self, scores, score_threshold = 0.65, max_size = 10, iou_threshold = 0.5):
        '''
        filters bounding boxes based on the score threshold and the maximum size to prevent overlapping boxes.
        Returns indexes of the bounding boxes
        '''
        bbox = tf.image.non_max_suppression(np.squeeze(self.output_data[..., :4]), scores, max_output_size=max_size, \
               iou_threshold=iou_threshold, score_threshold = score_threshold)
        return np.array(bbox)