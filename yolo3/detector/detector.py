class Detector:
    """
    检测器模板类
    """
    def __init__(self):
        self.model = None
        self.input_image_size = (None, None)
        self.class_names = None
        self.score_threshold = 0.3

    def _generate(self):
        pass

    def preprocess(self, image_data):
        pass

    def detect(self, image_data):
        """
        :param image_data: 图像数据，numpy的array类型，通道顺序为[R, G, B]
        :return:
        """
        pass
