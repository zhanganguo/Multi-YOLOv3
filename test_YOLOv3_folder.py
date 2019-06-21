import os
from helper.utils import *
from yolo3.detector.YOLOv3 import YOLOv3


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ------------------------------- YOLOv3模型文件路径 ------------------------------------
yolo3_models = dict(
    darknet_yoloV3=dict(name='DarkNet-YOLOv3', path='logs/Xception-MultiGPU/model_2.h5'),
    mobilenet_yoloV3=dict(name='MobileNet-YOLOv3', path='logs/MobileNet-YOLOv3/ep054-loss4.365-val_loss4.389.h5'),
    xception_yoloV3=dict(name='Xception-YOLOv3', path='logs/Xception-MultiGPU/model_0.h5'),
    mobilenetV2_yoloV3=dict(name='MobileNetV2-YOLOv3', path='logs/MobileNetV2-YOLOv3/ep033-loss6623.484-val_loss7848.784.h5')
)
model_type = 'xception_yoloV3'

yolo3_anchors_path = 'dataset/yolo_anchors.txt'
yolo3_classes_name_path = 'dataset/object_classes.txt'


if __name__ == '__main__':
    config = {
        'model_path': yolo3_models[model_type]['path'],
        'anchors_path': yolo3_anchors_path,
        'classes_path': yolo3_classes_name_path,
    }
    detector = YOLOv3(config, input_shape=(416, 416, 3), type=yolo3_models[model_type]['name'])

    sub_folders = [
        'test_image_folder1', 'test_image_folder2'
    ]

    data_folder_root = 'M:\\dataset_commit'
    output_folder_root = 'Q:\\detection\\'
    test_categories = [
        {
            'image_folder': os.path.join(data_folder_root, sub_folder, 'test', 'img'),
            'groundtruth_xml_folder': os.path.join(data_folder_root, sub_folder, 'test', 'img'),
            'output_folder': os.path.join(output_folder_root, model_type, sub_folder, 'test', 'img')
        } for sub_folder in sub_folders
    ]

    for category in test_categories:
        in_image_folder_path = category['image_folder']
        groundtruth_xml_folder_path = category['groundtruth_xml_folder']
        out_image_folder_path = category['output_folder']
        images = os.listdir(in_image_folder_path)

        for file_name in images:
            if os.path.exists(os.path.join(out_image_folder_path, file_name[:-4] + '.xml')):
                print('SKIP FILE: {}'.format(os.path.join(out_image_folder_path, file_name)))
                continue
            if not (file_name.lower().endswith('.bmp') or file_name.lower().endswith('.jpg') or
                    file_name.lower().endswith('.png')):
                continue
            elif file_name.lower().endswith('.bmp') or file_name.lower().endswith('.png'):
                image_path = os.path.join(in_image_folder_path, file_name)
                bmp_image = Image.open(image_path)
                bmp_image.save(image_path[:-4] + '.jpg', format='jpeg')
                image_path = image_path[:-4] + '.jpg'
            elif file_name.lower().endswith('.jpg'):
                image_path = os.path.join(in_image_folder_path, file_name)
            else:
                continue
            print('file: {}  -------------------------->'.format(image_path))
            HD_image = Image.open(image_path)
            HD_image_data = np.array(HD_image, dtype='float32')

            yolo_results = detector.detect(HD_image_data)

            HD_image = plot_detection_on_image(image=HD_image, detection=yolo_results,
                                               class_names=['class1', 'class2', 'class3'])

            if not os.path.exists(out_image_folder_path):
                os.makedirs(out_image_folder_path)
            #
            groundtruth_xml_file = os.path.join(groundtruth_xml_folder_path, file_name[:-4] + '.xml')
            if os.path.exists(groundtruth_xml_file):
                groundtruth = load_groundtruth_defects_from_xml(groundtruth_xml_file)
                HD_image = plot_groundtruth_on_image(HD_image, groundtruth)
            # # HD_image.show()
            HD_image.save(os.path.join(out_image_folder_path, file_name[:-4] + '.jpg'), format='jpeg')
            save_annotation_to_xml(os.path.join(out_image_folder_path, file_name[:-4] + '.xml'), yolo_results)
