import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from PIL import Image, ImageFont, ImageDraw
import colorsys
import numpy as np


def load_groundtruth_defects_from_xml(xml_file):
    groundtruth_defects = list()

    with open(xml_file, 'r', encoding='utf-8') as xml_f:
        tree = ET.parse(xml_f)
        root = tree.getroot()

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            b = (int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text),
                 int(bndbox.find('ymax').text))

            GT_defect = {
                'class': class_name,
                'box': b
            }

            groundtruth_defects.append(GT_defect)

    return groundtruth_defects


def load_detected_defects_from_xml(xml_file, whitelist_class='baimingdan'):
    yolo3_detected_defects = list()
    baimingdan_defects = list()

    with open(xml_file, 'r', encoding='utf-8') as xml_f:
        tree = ET.parse(xml_f)
        root = tree.getroot()

        for obj in root.iter('object'):
            if obj is None:
                continue
            score = float(obj.find('score').text)
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            b = (int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text),
                 int(bndbox.find('ymax').text))

            detected_result = {
                'class': class_name,
                'score': score,
                'box': b
            }

            if whitelist_class in class_name:
                baimingdan_defects.append(detected_result)
            else:
                yolo3_detected_defects.append(detected_result)

    return yolo3_detected_defects, baimingdan_defects


def extract_boxes_from_detected_results(detected_results):
    detected_boxes = list()
    for detected_result in detected_results:
        if isinstance(detected_result, tuple):
            detected_boxes.append(detected_result[2])
        elif isinstance(detected_result, dict):
            detected_boxes.append(detected_result['box'])
    return detected_boxes


def merge_box_as_rectangle_with_neighbors(detection_boxes, neighbor_threshold=0):
    if len(detection_boxes) <= 1:
        return detection_boxes

    has_overlap = True
    while has_overlap:
        has_overlap = False
        for idx, box in enumerate(detection_boxes):
            if box is None:
                continue
            [x1, y1, x2, y2] = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            merged_cmp_box = None
            new_box = None
            for cmp_idx in range(idx+1, len(detection_boxes)):
                cmp_box = detection_boxes[cmp_idx]
                if cmp_box is None:
                    continue
                [cmp_x1, cmp_y1, cmp_x2, cmp_y2] = cmp_box
                cmp_center_x = (cmp_x1 + cmp_x2) / 2
                cmp_center_y = (cmp_y1 + cmp_y2) / 2
                # 若两个矩形框产生了重叠或者相邻
                if np.abs(cmp_center_x - center_x) <= np.abs(x2 - x1) / 2 + np.abs(cmp_x2 - cmp_x1) / 2 + neighbor_threshold and \
                        np.abs(cmp_center_y - center_y) <= np.abs(y2 - y1) / 2 + np.abs(cmp_y2 - cmp_y1) / 2 + neighbor_threshold:
                    new_x1 = min(x1, cmp_x1)
                    new_y1 = min(y1, cmp_y1)
                    new_x2 = max(x2, cmp_x2)
                    new_y2 = max(y2, cmp_y2)
                    new_box = [new_x1, new_y1, new_x2, new_y2]
                    merged_cmp_box = cmp_box
                    has_overlap = True
                    break
                else:
                    has_overlap = False
            if has_overlap:
                if box in detection_boxes:
                    detection_boxes.remove(box)
                if merged_cmp_box in detection_boxes:
                    detection_boxes.remove(merged_cmp_box)
                detection_boxes.append(new_box)
                break

    return detection_boxes

def crop_image_by_center(image, desired_size, object_box):
    xmin, ymin, xmax, ymax = object_box
    xcenter = int((xmin + xmax) / 2)
    ycenter = int((ymin + ymax) / 2)

    im_width, im_height = image.size
    desired_width, desired_height = desired_size[0], desired_size[1]

    if im_width <= desired_width:
        crop_x_left, crop_x_right = 0, im_width
    else:
        if xcenter >= int(desired_width / 2):
            if im_width - xcenter >= int(desired_width / 2):
                crop_x_left, crop_x_right = xcenter - int(desired_width / 2), xcenter + int(desired_width / 2)
            else:
                crop_x_left, crop_x_right = im_width - desired_width, im_width
        else:
            crop_x_left, crop_x_right = 0, desired_width
    if im_height <= desired_height:
        crop_y_top, crop_y_bottom = 0, im_height
    else:
        if ycenter >= int(desired_height / 2):
            if im_height - ycenter >= int(desired_height / 2):
                crop_y_top, crop_y_bottom = ycenter - int(desired_height / 2), ycenter + int(desired_height / 2)
            else:
                crop_y_top, crop_y_bottom = im_height - desired_height, im_height
        else:
            crop_y_top, crop_y_bottom = 0, desired_height

    cropped_image = image.crop((crop_x_left, crop_y_top, crop_x_right, crop_y_bottom))
    offset = (crop_x_left, crop_y_top)

    return cropped_image, offset


def save_annotation_to_xml(xml_file, yolo3_results):
    root = Element('annotation')
    folder_ = SubElement(root, 'folder')
    filename_ = SubElement(root, 'filename')
    path_ = SubElement(root, 'path')
    source_ = SubElement(root, 'source')
    database__ = SubElement(source_, 'database')
    size_ = SubElement(root, 'size')
    width__ = SubElement(size_, 'width')
    height__ = SubElement(size_, 'height')
    depth__ = SubElement(size_, 'depth')
    segmented_ = SubElement(root, 'segmented')

    folder_.text = os.path.split(xml_file)[0]
    filename_.text = os.path.split(xml_file)[1]
    path_.text = xml_file[:-4]+'.jpg'
    database__.text = ''
    width__.text = '5000'
    height__.text = '8000'
    depth__.text = '3'
    segmented_.text = '0'

    if len(yolo3_results) > 0:
        for yolo3_result in yolo3_results:
            class_name = yolo3_result['class']
            score = yolo3_result['score']
            box = yolo3_result['box']
            object_ = SubElement(root, 'object')
            name__ = SubElement(object_, 'name')
            pose__ = SubElement(object_, 'pose')
            truncated__ = SubElement(object_, 'truncated')
            difficult__ = SubElement(object_, 'difficult')
            score__ = SubElement(object_, 'score')
            bndbox__ = SubElement(object_, 'bndbox')
            xmin___ = SubElement(bndbox__, 'xmin')
            ymin___ = SubElement(bndbox__, 'ymin')
            xmax___ = SubElement(bndbox__, 'xmax')
            ymax___ = SubElement(bndbox__, 'ymax')
            name__.text = class_name
            score__.text = str(score)
            pose__.text = '0'
            truncated__.text = '0'
            difficult__.text = '0'
            xmin___.text = str(box[0])
            ymin___.text = str(box[1])
            xmax___.text = str(box[2])
            ymax___.text = str(box[3])

    tree = ElementTree(root)
    tree.write(xml_file, encoding='utf-8')


def plot_groundtruth_on_image(image, groundtruth):
    # Generate colors for drawing bounding boxes.
    # class_names = ['lp', 'face']
    color = (255, 255, 255)

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = 3

    for item in groundtruth:
        predicted_class = item['class']
        box = item['box']

        label = '{}'.format(predicted_class)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box
        top = max(0, top)
        left = max(0, left)
        bottom = min(image.size[1], bottom)
        right = min(image.size[0], right)
        # print('          {} {} {}'.format(label, (left, top), (right, bottom)))

        if bottom + label_size[1] >= image.size[1]:
            text_origin = np.array([left, image.size[1] - label_size[1]])
        else:
            text_origin = np.array([left, bottom + 1])
        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=color)
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image


def plot_detection_on_image(image, class_names, detection, label_location='top'):
    # Generate colors for drawing bounding boxes.
    # class_names = ['lp', 'face']
    hsv_tuples = [(x / len(class_names)+1, 1., 1.)
                  for x in range(len(class_names)+1)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = 1

    for item in detection:
        if isinstance(item, dict):
            predicted_class = item['class']
            # if 'baimingdan' in predicted_class:
            #     continue
            box = item['box']
            score = item['score']
        else:
            predicted_class = item[0]
            # if 'baimingdan' in predicted_class:
            #     continue
            box = item[2]
            score = float(item[1])

        if predicted_class not in class_names:
            continue

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box
        top = max(0, top)
        left = max(0, left)
        bottom = min(image.size[1], bottom)
        right = min(image.size[0], right)
        # print('          {} {} {}'.format(label, (left, top), (right, bottom)))

        if label_location == 'bottom':
            if bottom + label_size[1] >= image.size[1]:
                text_origin = np.array([left, image.size[1] - label_size[1]])
            else:
                text_origin = np.array([left, bottom + 1])
        elif label_location == 'right':
            text_origin = np.array([right + 1, int((top + bottom)/2)])
        else:
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
        c = class_names.index(predicted_class) if predicted_class in class_names else len(class_names)
        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def is_overlap(box1, box2):
    [x1, y1, x2, y2] = box1
    [x1_, y1_, x2_, y2_] = box2
    if x2 > x1_ and x2_ > x1 and y2 > y1_ and y2_ > y1:
        return True
    else:
        return False


def evaluate_result_with_xml_and_xml(detected_boxes, groundtruth_boxes):
    num_groundtruth_boxes = len(groundtruth_boxes)  # 真实的目标数量
    num_correctly_detected = 0  # 正确检测到的数量
    num_incorrectly_detected = 0  # 误检数量
    num_not_detected = 0  # 漏检数量

    for detected_box in detected_boxes:
        has_matched = False
        has_finished = False
        while (not has_matched) and (not has_finished):
            for groundtruth_box in groundtruth_boxes:
                if is_overlap(detected_box, groundtruth_box):
                    has_matched = True  # 找到匹配的真实瑕疵框
                    continue
            has_finished = True     # 未找到匹配的真实瑕疵框
        if not has_matched:
            num_incorrectly_detected += 1  # 误检

    for groundtruth_box in groundtruth_boxes:
        has_matched = False
        has_finished = False
        while (not has_matched) and (not has_finished):
            for detected_box in detected_boxes:
                if is_overlap(detected_box, groundtruth_box):
                    has_matched = True  # 找到匹配的真实瑕疵框
                    continue
            has_finished = True  # 未找到匹配的真实瑕疵框
        if has_matched:
            num_correctly_detected += 1  # 准确检测到瑕疵
        else:
            num_not_detected += 1  # 漏检

    # 返回：总共的瑕疵数量，正确检测到的数量，误检数量，漏检数量
    return num_groundtruth_boxes, num_correctly_detected, num_incorrectly_detected, num_not_detected


def evaluate_result_with_single_xml(detected_boxes):
    """
    在没有瑕疵的布匹上，由于不会产生xml标注文件，因此需要对检测结果的xml文件进行独立分析
    :param detected_boxes: 检测结果
    :return: 总共的瑕疵数量，正确检测到的数量，误检数量，漏检数量
    """
    num_groundtruth_boxes = 0  # 没有groundtruth的xml文件，因此真实的目标数量为0
    num_correctly_detected = 0  # 正确检测到的数量
    num_incorrectly_detected = len(detected_boxes)  # 误检数量为检测结果的xml文件中所有box
    num_not_detected = 0  # 漏检数量

    # 返回：总共的瑕疵数量，正确检测到的数量，误检数量，漏检数量
    return num_groundtruth_boxes, num_correctly_detected, num_incorrectly_detected, num_not_detected




