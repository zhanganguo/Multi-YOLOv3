from PIL import Image


class InputHelper:
    def __init__(self, cropped_image_size):
        self.image = None
        self.cropped_image_size = cropped_image_size

    def load_image_by_file(self, file_path):
        self.image = Image.open(file_path)

    def load_image_by_cache(self, cache):
        self.image = Image.fromarray(cache.astype('uint8')).convert('RGB')

    def generate(self):
        left_x, top_y = 0, 0
        is_finished = False
        while not is_finished:
            right_x = left_x + self.cropped_image_size[0]
            bottom_y = top_y + self.cropped_image_size[1]
            if right_x >= self.image.width:
                left_x = self.image.width - self.cropped_image_size[0]
                right_x = self.image.width
                is_x_reloop = True
            else:
                is_x_reloop = False
            if bottom_y >= self.image.height:
                top_y = self.image.height - self.cropped_image_size[1]
                bottom_y = self.image.height

            cropped_image = self.image.crop((left_x, top_y, right_x, bottom_y))
            cropped_image_offset = (left_x, top_y)

            if is_x_reloop:
                left_x = 0
                top_y += int(self.cropped_image_size[1] * 0.75)
            else:
                left_x += int(self.cropped_image_size[0] * 0.75)

            if right_x >= self.image.width and bottom_y >= self.image.height:
                is_finished = True

            yield cropped_image, cropped_image_offset


if __name__ == '__main__':
    image_file = 'D:/2011-07-14 16.51.12.jpg'
    input_helper = InputHelper(cropped_image_size=[416, 416])
    input_helper.load_image_by_file(file_path=image_file)
    i = 0
    for image, image_offset in input_helper.generate():
        print('image_offset: {}'.format(image_offset))
        image.save('D:/tt/{}.jpg'.format(i), format='jpeg')
        i += 1




