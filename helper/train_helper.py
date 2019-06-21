from PIL import Image
import json
import os
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree


def crop_image_and_annotation(crop_size, image_file, annotation_file, output_image_folder, output_annotation_folder):
    with open(annotation_file, 'r') as f:
        annot = json.load(f)
    # location = np.asarray(annot['shapes'][0]['points'])
    image = Image.open(image_file)
    # print(location.tolist())
    info = dict()
    info['folder'] = 'none'
    info['filename'] = 'none'
    info['path'] = image_file
    info['database'] = 'Unknown'
    info['width'] = str(crop_size[0])
    info['height'] = str(crop_size[1])
    info['depth'] = str(3)
    info['segmented'] = str('0')
    if 'shapes' in annot.keys():
        info['object'] = list()
        for item in annot['shapes']:
            info['object'].append({
                'name': item.get('label'),
                'pose': 'Unspecified',
                'truncated': str(0),
                'difficult': str(0),
                'points': item.get('points')
            })
    print(info)

    index = 0
    (image_path, image_name) = os.path.split(image_file)
    (image_name, extension) = os.path.splitext(image_name)
    for cropped_image, cropped_image_offset, info in _crop_image_and_annotation(image=image, info=info):
        generate_image_file(cropped_image, output_image_file=os.path.join(output_image_folder,
                                                                          image_name+'_'+str(index)+extension))
        generate_xml_file(info, output_xml_file=os.path.join(output_image_folder, '{}_{}.xml'.format(image_name, index)))
        index += 1


def _crop_image_and_annotation(image, info):
    left_x, top_y = 0, 0
    is_finished = False
    while not is_finished:
        right_x = left_x + int(info.get('width'))
        bottom_y = top_y + int(info.get('height'))
        if right_x >= image.width:
            left_x = image.width - int(info.get('width'))
            right_x = image.width
            is_x_reloop = True
        else:
            is_x_reloop = False
        if bottom_y >= image.height:
            top_y = image.height - int(info.get('height'))
            bottom_y = image.height

        cropped_image = image.crop((left_x, top_y, right_x, bottom_y))
        cropped_image_offset = (left_x, top_y)

        for item in info.get('object'):
            inner_points = []
            for [x, y] in item.get('points'):
                if left_x <= x <= right_x and top_y <= y <= bottom_y:
                    inner_points.append([x, y])
                    # print('{}<={}<={}, {}<={}<={}'.format(left_x, x, right_x, top_y, y, bottom_y))
            if len(inner_points) < 2:
                continue
            inner_points = np.asarray(inner_points)
            min_x = np.min(inner_points[:, 0])
            max_x = np.max(inner_points[:, 0])
            min_y = np.min(inner_points[:, 1])
            max_y = np.max(inner_points[:, 1])
            # print('minX: {}, minY: {}, maxX: {}, maxY: {}， offset: {},{},{},{}'.format(min_x, min_y, max_x, max_y,
            #                                                                            left_x, top_y, right_x, bottom_y))

            # bbox = [min_x, min_y, max_x, max_y]
            item['xmin'] = min_x - cropped_image_offset[0]
            item['ymin'] = min_y - cropped_image_offset[1]
            item['xmax'] = max_x - cropped_image_offset[0]
            item['ymax'] = max_y - cropped_image_offset[1]

        if is_x_reloop:
            left_x = 0
            top_y += int(int(info.get('height')) * 0.5)
        else:
            left_x += int(int(info.get('width')) * 0.5)

        if right_x >= image.width and bottom_y >= image.height:
            is_finished = True

        yield cropped_image, cropped_image_offset, info


def generate_image_file(image, output_image_file):
    image.save(output_image_file, format='jpeg')


def generate_xml_file(info, output_xml_file):
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

    folder_.text = info.get('folder')
    filename_.text = info.get('filename')
    path_.text = info.get('path')
    database__.text = info.get('database')
    width__.text = info.get('width')
    height__.text = info.get('height')
    depth__.text = info.get('depth')
    segmented_.text = info.get('segmented')
    if len(info.get('object')):
        for item in info.get('object'):
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
            name__.text = item.get('name')
            pose__.text = item.get('pose')
            truncated__.text = item.get('truncated')
            difficult__.text = item.get('difficult')
            xmin___.text = str(item.get('xmin'))
            ymin___.text = str(item.get('ymin'))
            xmax___.text = str(item.get('xmax'))
            ymax___.text = str(item.get('ymax'))

    tree = ElementTree(root)
    tree.write(output_xml_file, encoding='utf-8')


if __name__ == '__main__':
    image_folder = 'E:\\cloth_defect\\HD_xml'
    annotation_folder = 'E:\\cloth_defect\\HD_xml'
    small_image_output_folder = 'E:\\cloth_defect\\416_by_HD_xml'
    xml_annotation_output_folder = 'E:\\cloth_defect\\416_by_HD_xml'
    images = os.listdir(image_folder)
    for image in images:
        if image.lower().endswith('.jpg') or image.lower().endswith('.png'):
            annotation_file = os.path.join(annotation_folder, image[:-4]+'.json')
            image_file = os.path.join(image_folder, image)
            if not os.path.exists(small_image_output_folder):
                os.makedirs(small_image_output_folder)
            if not os.path.exists(xml_annotation_output_folder):
                os.makedirs(xml_annotation_output_folder)
            crop_image_and_annotation(crop_size=[416, 416], image_file=image_file, annotation_file=annotation_file,
                                      output_image_folder=small_image_output_folder,
                                      output_annotation_folder=xml_annotation_output_folder)
