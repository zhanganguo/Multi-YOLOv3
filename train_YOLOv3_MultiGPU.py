import os

import numpy as np
import keras.backend as K
import tensorflow as tf

from keras.layers import Input, Lambda, add
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.backend.tensorflow_backend import set_session

from yolo3.utils import get_random_data

BACKBONES = ['DarkNet', 'MobileNet', 'MobileNetV2', 'Xception']
BACKBONE = BACKBONES[0]
if BACKBONE == BACKBONES[0]:
    from yolo3.model_DarkNet import preprocess_true_boxes, yolo_body, yolo_loss  #No model_DarkNet.py available
elif BACKBONE == BACKBONES[1]:
    from yolo3.model_Mobilenet import preprocess_true_boxes, yolo_body, yolo_loss
elif BACKBONE == BACKBONES[2]:
    from yolo3.model_MobileNetV2 import preprocess_true_boxes, yolo_body, yolo_loss
elif BACKBONE == BACKBONES[3]:
    from yolo3.model_Xception import preprocess_true_boxes, yolo_body, yolo_loss
else:
    from yolo3.model_DarkNet import preprocess_true_boxes, yolo_body, yolo_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
GPUs = 6


def _main():
    annotation_path = 'dataset/train.txt'
    log_dir = 'logs/{}-MultiGPU'.format(BACKBONE)
    classes_path = 'dataset/object_classes.txt'
    anchors_path = 'dataset/yolo_anchors.txt'

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    with tf.device('/cpu:0'):
        template_model = create_model(input_shape, anchors, num_classes, load_pretrained=False,
                                      freeze_body=1, weights_path="")  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    model = multi_gpu_model(template_model, gpus=GPUs)
    print('use the multi_gpu_model for model training ')

    modelsave_checkpoint = ModelSaveCheckpoint(model=template_model, folder_path=log_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        batch_size = 32 * GPUs
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=10,
            initial_epoch=0,
            callbacks=[logging, modelsave_checkpoint]
        )

        '''save the model config and weigths on cpu field share the weights'''
        with tf.device('/cpu:0'):
            template_model.save(log_dir + 'frozen_weight.h5')

    if True:
        batch_size = 32 * GPUs
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Unfreeze all of the layers.')

        model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=100,
            initial_epoch=10,
            callbacks=[logging, reduce_lr, early_stopping, modelsave_checkpoint])

        '''save the model config and weigths on cpu field share the weights'''
        with tf.device('/cpu:0'):
            template_model.save(log_dir + 'final_weight.h5')


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='/model_data/yolo_weights.h5'):
    """create the training model"""
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
        else:
            for i in range(len(model_body.layers)):
                model_body.layers[i].trainable = True
            print('UnFreeze all of the model_body layers.')

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def _get_available_devices():
    return [x.name for x in K.get_session().list_devices()]


def _normalize_device_name(name):
    name = '/' + ':'.join(name.lower().replace('/', '').split(':')[-2:])
    return name


'''modify the devices('/cpu:0') concatenate to add for yolo loss tensor(1,)'''


def multi_gpu_model(model, gpus=None):
    """Replicates a model on different GPUs.
    Specifically, this function implements single-machine
    multi-GPU data parallelism. It works in the following way:
    - Divide the model's input(s) into multiple sub-batches.
    - Apply a model copy on each sub-batch. Every model copy
        is executed on a dedicated GPU.
    - Concatenate the results (on CPU) into one big batch.
    E.g. if your `batch_size` is 64 and you use `gpus=2`,
    then we will divide the input into 2 sub-batches of 32 samples,
    process each sub-batch on one GPU, then return the full
    batch of 64 processed samples.
    This induces quasi-linear speedup on up to 8 GPUs.
    This function is only available with the TensorFlow backend
    for the time being.
    # Arguments
        model: A Keras model instance. To avoid OOM errors,
            this model could have been built on CPU, for instance
            (see usage example below).
        gpus: Integer >= 2 or list of integers, number of GPUs or
            list of GPU IDs on which to create model replicas.
    # Returns
        A Keras `Model` instance which can be used just like the initial
        `model` argument, but which distributes its workload on multiple GPUs.
    # Example
    ```python
        import tensorflow as tf
        from keras.applications import Xception
        from keras.utils import multi_gpu_model
        import numpy as np
        num_samples = 1000
        height = 224
        width = 224
        num_classes = 1000
        # Instantiate the base model (or "template" model).
        # We recommend doing this with under a CPU device scope,
        # so that the model's weights are hosted on CPU memory.
        # Otherwise they may end up hosted on a GPU, which would
        # complicate weight sharing.
        with tf.device('/cpu:0'):
            model = Xception(weights=None,
                             input_shape=(height, width, 3),
                             classes=num_classes)
        # Replicates the model on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        parallel_model = multi_gpu_model(model, gpus=8)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop')
        # Generate dummy data.
        x = np.random.random((num_samples, height, width, 3))
        y = np.random.random((num_samples, num_classes))
        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit(x, y, epochs=20, batch_size=256)
        # Save model via the template model (which shares the same weights):
        model.save('my_model.h5')
    ```
    # On model saving
    To save the multi-gpu model, use `.save(fname)` or `.save_weights(fname)`
    with the template model (the argument you passed to `multi_gpu_model`),
    rather than the model returned by `multi_gpu_model`.
    """
    if K.backend() != 'tensorflow':
        raise ValueError('`multi_gpu_model` is only available '
                         'with the TensorFlow backend.')

    available_devices = _get_available_devices()
    available_devices = [_normalize_device_name(name) for name in available_devices]
    if not gpus:
        # Using all visible GPUs when not specifying `gpus`
        # e.g. CUDA_VISIBLE_DEVICES=0,2 python3 keras_mgpu.py
        gpus = len([x for x in available_devices if 'gpu' in x])

    if isinstance(gpus, (list, tuple)):
        if len(gpus) <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `len(gpus) >= 2`. '
                             'Received: `gpus=%s`' % gpus)
        num_gpus = len(gpus)
        target_gpu_ids = gpus
    else:
        if gpus <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `gpus >= 2`. '
                             'Received: `gpus=%d`' % gpus)
        num_gpus = gpus
        target_gpu_ids = range(num_gpus)

    import tensorflow as tf

    target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in target_gpu_ids]
    for device in target_devices:
        if device not in available_devices:
            raise ValueError(
                'To call `multi_gpu_model` with `gpus=%d`, '
                'we expect the following devices to be available: %s. '
                'However this machine only has: %s. '
                'Try reducing `gpus`.' % (gpus,
                                          target_devices,
                                          available_devices))

    def get_slice(data, i, parts):
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == num_gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i, gpu_id in enumerate(target_gpu_ids):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('replica_%d' % gpu_id):
                inputs = []
                # Retrieve a slice of the input.
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_i = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'i': i,
                                                'parts': num_gpus})(x)
                    inputs.append(slice_i)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for name, outputs in zip(model.output_names, all_outputs):
            merged.append(add(outputs, name=name))
        return Model(model.inputs, merged)


class ModelSaveCheckpoint(Callback):
    def __init__(self, model, folder_path):
        super(ModelSaveCheckpoint, self).__init__()
        self.model_to_save = model
        self.folder_path = folder_path

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(os.path.join(self.folder_path, 'model_%d.h5' % epoch))


if __name__ == '__main__':
    # memory of gpu allowcate
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)

    _main()
