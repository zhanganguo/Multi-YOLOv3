import os, time
from keras import backend as K
from keras.layers import Input
from .detector import Detector
import colorsys
from ..utils import *
from helper import image_helper
from helper.input_helper import InputHelper
from tqdm import tqdm


class YOLOv3(Detector):
    def __init__(self, config, input_shape=(416, 416, 3), type='darknet-yolo3'):
        super().__init__()

        self.iou = 0.45
        self.score_threshold = 0.1
        self.input_image_size = (input_shape[0], input_shape[1])
        self.input_shape = input_shape

        self.type = type

        self.model_path = config['model_path']
        self.classes_path = config['classes_path']
        self.anchors_path = config['anchors_path']

        self.session = K.get_session()
        self.anchors = None

        self._generate()

    def _generate(self):
        self._load_anchors(self.anchors_path)
        self._load_classes(self.classes_path)
        self._load_model_data(self.model_path)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.

        if self.type == 'DarkNet-YOLOv3':
            from ..model_DarkNet import yolo_eval, yolo_body
        elif self.type == 'MobileNet-YOLOv3':
            from ..model_Mobilenet import yolo_eval, yolo_body
        elif self.type == 'ShuffleDarkNet-YOLOv3':
            from ..model_ShuffleDarkNet import yolo_eval, yolo_body
        elif self.type == 'Xception-YOLOv3':
            from ..model_Xception import yolo_eval, yolo_body
        elif self.type == 'MobileNetV2-YOLOv3':
            from ..model_MobileNetV2 import yolo_eval, yolo_body
        else:
            raise NotImplementedError

        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(self.model.output, self.anchors,
                                                          len(self.class_names), self.input_image_shape,
                                                          score_threshold=self.score_threshold, iou_threshold=self.iou)

    def _load_model_data(self, model_path):
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        model_path = os.path.expanduser(model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        if self.type == 'DarkNet-YOLOv3':
            from ..model_DarkNet import yolo_eval, yolo_body
        elif self.type == 'MobileNet-YOLOv3':
            from ..model_Mobilenet import yolo_eval, yolo_body
        elif self.type == 'ShuffleDarkNet-YOLOv3':
            from ..model_ShuffleDarkNet import yolo_eval, yolo_body
        elif self.type == 'Xception-YOLOv3':
            from ..model_Xception import yolo_eval, yolo_body
        elif self.type == 'MobileNetV2-YOLOv3':
            from ..model_MobileNetV2 import yolo_eval, yolo_body
        else:
            raise NotImplementedError

        self.model = yolo_body(Input(shape=self.input_shape), num_anchors // 3, num_classes)
        self.model.load_weights(model_path)  # make sure model, anchors and classes match

    def _load_classes(self, classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        self.class_names = [c.strip() for c in class_names]

    def _load_anchors(self, anchors_path):
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        self.anchors = np.array(anchors).reshape(-1, 2)

    def preprocess(self, image_data):
        image = Image.fromarray(image_data.astype('uint8')).convert('RGB')
        if self.input_image_size != (None, None):
            assert self.input_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.input_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.input_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')

        if self.input_shape[2] == 3:  # RGB mode
            pass
        elif self.input_shape[2] == 2:  # Gray and Edge mode
            image_edges = image_helper.find_edges(boxed_image)
            boxed_image = boxed_image.convert('L')
            image_edges = image_edges.convert('L')
            image_data = np.array(boxed_image, dtype='float32')
            image_edges_data = np.array(image_edges, dtype='float32')
            image_data = np.expand_dims(image_data, axis=2)
            image_edges_data = np.expand_dims(image_edges_data, axis=2)
            image_data = np.concatenate((image_data, image_edges_data), axis=2)
        elif self.input_shape[2] == 6:  # RGB and GRB Edge mode
            image_edges = image_helper.find_edges(boxed_image)
            image_edges_data = np.array(image_edges, dtype='float32')
            image_data = np.concatenate((image_data, image_edges_data), axis=2)
        else:
            raise NotImplementedError

        image_data /= 255

        return image_data

    def _detect(self, image_data):
        im_h, im_w = image_data.shape[0], image_data.shape[1]

        image_data = self.preprocess(image_data)

        image_data = np.expand_dims(image_data, 0)  # Add batch dimension

        out_boxes, out_scores, out_classes = self.session.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: image_data,
                self.input_image_shape: [im_h, im_w],
                K.learning_phase(): 0
            })

        # print('     LARGE: -> Found {} boxes for {}'.format(len(out_boxes), 'img'))

        result = dict()
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            result[str(i)] = {
                'class': predicted_class,
                'box': [np.round(box[1]).astype('int32'), np.round(box[0]).astype('int32'),
                        np.round(box[3]).astype('int32'), np.round(box[2]).astype('int32')],
                'score': score
            }
        return result

    def detect(self, image_data):
        yolo3_detection_results = list()

        HD_image = Image.fromarray(image_data.astype('uint8')).convert('RGB')

        # 对高清大图进行滑动窗口裁剪
        input_helper = InputHelper(cropped_image_size=self.input_image_size)
        input_helper.load_image_by_cache(image_data)

        image_list = tqdm(input_helper.generate())

        process_index = 0
        for image, image_offset in image_list:
            image_list.set_description('    已完成：{}/{}'.format(process_index, round(
                HD_image.width / self.input_image_size[0] * HD_image.height / self.input_image_size[1] * 16 / 9)))
            process_index += 1

            image_data = np.array(image, dtype='uint8')

            yolo3_results = self._detect(image_data)

            for obj in list(yolo3_results.keys()):
                obj_item = yolo3_results.get(obj)
                class_name = obj_item['class']
                score = obj_item['score']
                box = obj_item['box']

                # # 滤除掉检测框不在中央区域的
                # if not (self.cropped_image_size[0]/8 < (box[0]+box[2])/2 < self.cropped_image_size[0]*0.875 and
                #         self.cropped_image_size[1]/8 < (box[1]+box[3])/2 < self.cropped_image_size[1]*0.875):
                #     yolo3_results.pop(obj)
                #     continue
                if not (50 < (box[0] + box[2]) / 2 < self.input_image_size[0] - 50 and 50 < (box[1] + box[3]) / 2 <
                        self.input_image_size[1] - 50) and \
                        (image_offset[0] > 50 or image_offset[1] > 50):
                    yolo3_results.pop(obj)
                    continue

            for yolo3_key in list(yolo3_results.keys()):
                yolo3_result = yolo3_results.get(yolo3_key)
                x1 = int(max(0, yolo3_result['box'][0]) + image_offset[0])
                y1 = int(max(0, yolo3_result['box'][1]) + image_offset[1])
                x2 = int(min(self.input_image_size[0], yolo3_result['box'][2]) + image_offset[0])
                y2 = int(min(self.input_image_size[1], yolo3_result['box'][3]) + image_offset[1])
                yolo3_results[yolo3_key]['box'] = [x1, y1, x2, y2]

            for key in yolo3_results:
                yolo3_detection_results.append(yolo3_results[key])

        return yolo3_detection_results
