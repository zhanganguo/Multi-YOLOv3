from PIL import Image, ImageFont, ImageDraw, ImageFilter
import colorsys
import numpy as np


def find_edges(image):
    """
    :param image: 图像数据，PIL Image类型，通道顺序为[R,G,B]，像素值为0~255
    :return: 三个通道经过处理后的图像边缘数据，通道顺序为[R, G, B]，像素值为0~255
    """
    image = image.filter(ImageFilter.MinFilter(size=3))
    image = image.filter(ImageFilter.EDGE_ENHANCE)
    image = image.filter(ImageFilter.SMOOTH)
    image = image.filter(ImageFilter.FIND_EDGES)

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


def plot_detection_on_image_no_label_and_score(image, detection):
    # Generate colors for drawing bounding boxes.
    # class_names = ['lp', 'face']
    hsv_tuples = [(x / 1, 1., 1.) for x in range(1)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    font = ImageFont.truetype(font='asset/font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for box in detection:
        label = '{}'.format('defect')
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        if len(box) < 1:
            continue
        left, top, right, bottom = box
        top = max(0, top)
        left = max(0, left)
        bottom = min(image.size[1], bottom)
        right = min(image.size[0], right)
        print('          {} {} {}'.format(label, (left, top), (right, bottom)))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        c = 0
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


def plot_detection_on_image(image, class_names, detection):
    # Generate colors for drawing bounding boxes.
    # class_names = ['lp', 'face']
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    font = ImageFont.truetype(font='asset/font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for key in detection.keys():
        item = detection.get(key)
        predicted_class = item['class']
        box = item['box']
        score = item['score']

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box
        top = max(0, top)
        left = max(0, left)
        bottom = min(image.size[1], bottom)
        right = min(image.size[0], right)
        print('          {} {} {}'.format(label, (left, top), (right, bottom)))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        c = class_names.index(predicted_class)
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


def plot_detection_on_image_by_model(image, class_names, detection, symbol):
    if len(detection.keys()) < 1:
        return image
    # Generate colors for drawing bounding boxes.
    # class_names = ['lp', 'face']
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    font = ImageFont.truetype(font='asset/font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for key in detection.keys():
        item = detection.get(key)
        predicted_class = item['class']
        box = item['box']
        score = item['score']

        label = '{}{}{} {:.2f}'.format(symbol, predicted_class, symbol, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box
        top = max(0, top)
        left = max(0, left)
        bottom = min(image.size[1], bottom)
        right = min(image.size[0], right)
        print('          {} {} {}'.format(label, (left, top), (right, bottom)))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        c = class_names.index(predicted_class)
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


def merge_box_as_rectangle(detection_boxes):
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
                if np.abs(cmp_center_x - center_x) <= np.abs(x2 - x1) / 2 + np.abs(cmp_x2 - cmp_x1) / 2 and \
                        np.abs(cmp_center_y - center_y) <= np.abs(y2 - y1) / 2 + np.abs(cmp_y2 - cmp_y1) / 2:
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


def merge_result_as_rectangle_with_neighbors(big_results, small_results, neighbor_threshold=0):
    if len(big_results) <= 1:
        return big_results

    has_overlap = True
    while has_overlap:
        has_overlap = False
        for idx, big_result in enumerate(big_results):
            if big_result is None:
                continue
            [x1, y1, x2, y2] = big_result['box']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            merged_cmp_big_result = None
            new_result = None
            for cmp_idx in range(idx+1, len(big_results)):
                cmp_big_result = big_results[cmp_idx]
                if cmp_big_result is None:
                    continue
                [cmp_x1, cmp_y1, cmp_x2, cmp_y2] = cmp_big_result['box']
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
                    max_score = max(big_result['score'], cmp_big_result['score'])
                    new_class = '{}_{}'.format(big_result['class'], cmp_big_result['class'])
                    new_result = {
                        'class': new_class,
                        'score': max_score,
                        'box': new_box
                    }
                    merged_cmp_big_result = cmp_big_result
                    has_overlap = True
                    break
                else:
                    has_overlap = False
            if has_overlap:
                if big_result in big_results:
                    big_results.remove(big_result)
                if merged_cmp_big_result in big_results:
                    big_results.remove(merged_cmp_big_result)
                big_results.append(new_result)
                break

    return big_results


if __name__ == '__main__':
    boxes = [[10, 10, 20, 20], [5, 5, 9, 9], [15, 15, 25, 25], [100, 100, 150, 150], [110, 110, 120, 120]]
    boxes = merge_box_as_rectangle(boxes)
    print('boxes: {}'.format(boxes))
