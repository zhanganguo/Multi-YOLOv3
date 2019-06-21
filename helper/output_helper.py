import os
from xml.etree.ElementTree import Element, SubElement, ElementTree


def save_annotation_to_xml(xml_file, annotation_boxes, annotation_classes):
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

    if len(annotation_boxes) > 0:
        for box, class_name in zip(annotation_boxes, annotation_classes):
            object_ = SubElement(root, 'object')
            name__ = SubElement(object_, 'name')
            pose__ = SubElement(object_, 'pose')
            truncated__ = SubElement(object_, 'truncated')
            difficult__ = SubElement(object_, 'difficult')
            bndbox__ = SubElement(object_, 'bndbox')
            xmin___ = SubElement(bndbox__, 'xmin')
            ymin___ = SubElement(bndbox__, 'ymin')
            xmax___ = SubElement(bndbox__, 'xmax')
            ymax___ = SubElement(bndbox__, 'ymax')
            name__.text = class_name
            pose__.text = '0'
            truncated__.text = '0'
            difficult__.text = '0'
            xmin___.text = str(box[0])
            ymin___.text = str(box[1])
            xmax___.text = str(box[2])
            ymax___.text = str(box[3])

    tree = ElementTree(root)
    tree.write(xml_file, encoding='utf-8')


def save_detection_to_xml(xml_file, detection_results):
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

    if len(detection_results) > 0:
        for detection_result in detection_results:
            object_ = SubElement(root, 'object')
            name__ = SubElement(object_, 'name')
            pose__ = SubElement(object_, 'pose')
            truncated__ = SubElement(object_, 'truncated')
            difficult__ = SubElement(object_, 'difficult')
            bndbox__ = SubElement(object_, 'bndbox')
            xmin___ = SubElement(bndbox__, 'xmin')
            ymin___ = SubElement(bndbox__, 'ymin')
            xmax___ = SubElement(bndbox__, 'xmax')
            ymax___ = SubElement(bndbox__, 'ymax')
            score__ = SubElement(object_, 'score')
            name__.text = detection_result[0]
            pose__.text = '0'
            truncated__.text = '0'
            difficult__.text = '0'
            xmin___.text = str(detection_result[2][0])
            ymin___.text = str(detection_result[2][1])
            xmax___.text = str(detection_result[2][2])
            ymax___.text = str(detection_result[2][3])
            score__.text = str(detection_result[1])

    tree = ElementTree(root)
    tree.write(xml_file, encoding='utf-8')


