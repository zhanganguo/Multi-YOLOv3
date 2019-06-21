"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import merge, Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D,Input, Dense
from keras.layers import MaxPool2D,AveragePooling2D, Lambda, DepthwiseConv2D

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def leakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    # bottleneck_channels = int(out_channels * bottleneck_ratio)
    bottleneck_channels = out_channels
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(
        inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1_conv_3'.format(prefix))(
            s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def shuffle_block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage - 1],
                     strides=2, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = shuffle_unit(x, out_channels=channel_map[stage - 1], strides=1,
                         bottleneck_ratio=bottleneck_ratio, stage=stage, block=(1 + i))

    return x


def ShuffleNetV2(input_shape=None,
                 include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 num_shuffle_units=[3, 7, 3],
                 bottleneck_ratio=1,
                 classes=1000):
    print('input_shape:{}'.format(input_tensor))
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    out_dim_stage_two = {0.5: 48, 1: 116, 1.2: 132, 1.5: 176, 2: 244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor) * 4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  # calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)
    print('out_channels_in_stage:', out_channels_in_stage)

    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = shuffle_block(x, out_channels_in_stage,
                          repeat=repeat,
                          bottleneck_ratio=bottleneck_ratio,
                          stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    # Create model.
    model = Model(inputs, x, name=name)

    return model


def yolo_body(inputs, num_anchors, num_classes):
    """
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            (None, 416, 416, 3)  0
    __________________________________________________________________________________________________
    conv1 (Conv2D)                  (None, 208, 208, 24) 648         input_1[0][0]
    __________________________________________________________________________________________________
    maxpool1 (MaxPooling2D)         (None, 104, 104, 24) 0           conv1[0][0]
    __________________________________________________________________________________________________
    stage2/block1/1x1conv_1 (Conv2D (None, 104, 104, 116 2900        maxpool1[0][0]
    __________________________________________________________________________________________________
    stage2/block1/bn_1x1conv_1 (Bat (None, 104, 104, 116 464         stage2/block1/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage2/block1/relu_1x1conv_1 (A (None, 104, 104, 116 0           stage2/block1/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage2/block1/3x3dwconv (Depthw (None, 52, 52, 116)  1160        stage2/block1/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage2/block1/3x3dwconv_2 (Dept (None, 52, 52, 24)   240         maxpool1[0][0]
    __________________________________________________________________________________________________
    stage2/block1/bn_3x3dwconv (Bat (None, 52, 52, 116)  464         stage2/block1/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage2/block1/bn_3x3dwconv_2 (B (None, 52, 52, 24)   96          stage2/block1/3x3dwconv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block1/1x1conv_2 (Conv2D (None, 52, 52, 116)  13572       stage2/block1/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage2/block1/1x1_conv_3 (Conv2 (None, 52, 52, 116)  2900        stage2/block1/bn_3x3dwconv_2[0][0
    __________________________________________________________________________________________________
    stage2/block1/bn_1x1conv_2 (Bat (None, 52, 52, 116)  464         stage2/block1/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block1/bn_1x1conv_3 (Bat (None, 52, 52, 116)  464         stage2/block1/1x1_conv_3[0][0]
    __________________________________________________________________________________________________
    stage2/block1/relu_1x1conv_2 (A (None, 52, 52, 116)  0           stage2/block1/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block1/relu_1x1conv_3 (A (None, 52, 52, 116)  0           stage2/block1/bn_1x1conv_3[0][0]
    __________________________________________________________________________________________________
    stage2/block1/concat_2 (Concate (None, 52, 52, 232)  0           stage2/block1/relu_1x1conv_2[0][0
                                                                     stage2/block1/relu_1x1conv_3[0][0
    __________________________________________________________________________________________________
    stage2/block1/channel_shuffle ( (None, 52, 52, 232)  0           stage2/block1/concat_2[0][0]
    __________________________________________________________________________________________________
    stage2/block2/spl/sp1_slice (La (None, 52, 52, 116)  0           stage2/block1/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage2/block2/1x1conv_1 (Conv2D (None, 52, 52, 116)  13572       stage2/block2/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage2/block2/bn_1x1conv_1 (Bat (None, 52, 52, 116)  464         stage2/block2/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage2/block2/relu_1x1conv_1 (A (None, 52, 52, 116)  0           stage2/block2/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage2/block2/3x3dwconv (Depthw (None, 52, 52, 116)  1160        stage2/block2/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage2/block2/bn_3x3dwconv (Bat (None, 52, 52, 116)  464         stage2/block2/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage2/block2/1x1conv_2 (Conv2D (None, 52, 52, 116)  13572       stage2/block2/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage2/block2/bn_1x1conv_2 (Bat (None, 52, 52, 116)  464         stage2/block2/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block2/relu_1x1conv_2 (A (None, 52, 52, 116)  0           stage2/block2/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block2/spl/sp0_slice (La (None, 52, 52, 116)  0           stage2/block1/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage2/block2/concat_1 (Concate (None, 52, 52, 232)  0           stage2/block2/relu_1x1conv_2[0][0
                                                                     stage2/block2/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage2/block2/channel_shuffle ( (None, 52, 52, 232)  0           stage2/block2/concat_1[0][0]
    __________________________________________________________________________________________________
    stage2/block3/spl/sp1_slice (La (None, 52, 52, 116)  0           stage2/block2/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage2/block3/1x1conv_1 (Conv2D (None, 52, 52, 116)  13572       stage2/block3/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage2/block3/bn_1x1conv_1 (Bat (None, 52, 52, 116)  464         stage2/block3/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage2/block3/relu_1x1conv_1 (A (None, 52, 52, 116)  0           stage2/block3/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage2/block3/3x3dwconv (Depthw (None, 52, 52, 116)  1160        stage2/block3/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage2/block3/bn_3x3dwconv (Bat (None, 52, 52, 116)  464         stage2/block3/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage2/block3/1x1conv_2 (Conv2D (None, 52, 52, 116)  13572       stage2/block3/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage2/block3/bn_1x1conv_2 (Bat (None, 52, 52, 116)  464         stage2/block3/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block3/relu_1x1conv_2 (A (None, 52, 52, 116)  0           stage2/block3/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block3/spl/sp0_slice (La (None, 52, 52, 116)  0           stage2/block2/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage2/block3/concat_1 (Concate (None, 52, 52, 232)  0           stage2/block3/relu_1x1conv_2[0][0
                                                                     stage2/block3/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage2/block3/channel_shuffle ( (None, 52, 52, 232)  0           stage2/block3/concat_1[0][0]
    __________________________________________________________________________________________________
    stage2/block4/spl/sp1_slice (La (None, 52, 52, 116)  0           stage2/block3/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage2/block4/1x1conv_1 (Conv2D (None, 52, 52, 116)  13572       stage2/block4/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage2/block4/bn_1x1conv_1 (Bat (None, 52, 52, 116)  464         stage2/block4/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage2/block4/relu_1x1conv_1 (A (None, 52, 52, 116)  0           stage2/block4/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage2/block4/3x3dwconv (Depthw (None, 52, 52, 116)  1160        stage2/block4/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage2/block4/bn_3x3dwconv (Bat (None, 52, 52, 116)  464         stage2/block4/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage2/block4/1x1conv_2 (Conv2D (None, 52, 52, 116)  13572       stage2/block4/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage2/block4/bn_1x1conv_2 (Bat (None, 52, 52, 116)  464         stage2/block4/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block4/relu_1x1conv_2 (A (None, 52, 52, 116)  0           stage2/block4/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage2/block4/spl/sp0_slice (La (None, 52, 52, 116)  0           stage2/block3/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage2/block4/concat_1 (Concate (None, 52, 52, 232)  0           stage2/block4/relu_1x1conv_2[0][0
                                                                     stage2/block4/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage2/block4/channel_shuffle ( (None, 52, 52, 232)  0           stage2/block4/concat_1[0][0]
    __________________________________________________________________________________________________
    stage3/block1/1x1conv_1 (Conv2D (None, 52, 52, 232)  54056       stage2/block4/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block1/bn_1x1conv_1 (Bat (None, 52, 52, 232)  928         stage3/block1/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block1/relu_1x1conv_1 (A (None, 52, 52, 232)  0           stage3/block1/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block1/3x3dwconv (Depthw (None, 26, 26, 232)  2320        stage3/block1/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage3/block1/3x3dwconv_2 (Dept (None, 26, 26, 232)  2320        stage2/block4/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block1/bn_3x3dwconv (Bat (None, 26, 26, 232)  928         stage3/block1/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block1/bn_3x3dwconv_2 (B (None, 26, 26, 232)  928         stage3/block1/3x3dwconv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block1/1x1conv_2 (Conv2D (None, 26, 26, 232)  54056       stage3/block1/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block1/1x1_conv_3 (Conv2 (None, 26, 26, 232)  54056       stage3/block1/bn_3x3dwconv_2[0][0
    __________________________________________________________________________________________________
    stage3/block1/bn_1x1conv_2 (Bat (None, 26, 26, 232)  928         stage3/block1/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block1/bn_1x1conv_3 (Bat (None, 26, 26, 232)  928         stage3/block1/1x1_conv_3[0][0]
    __________________________________________________________________________________________________
    stage3/block1/relu_1x1conv_2 (A (None, 26, 26, 232)  0           stage3/block1/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block1/relu_1x1conv_3 (A (None, 26, 26, 232)  0           stage3/block1/bn_1x1conv_3[0][0]
    __________________________________________________________________________________________________
    stage3/block1/concat_2 (Concate (None, 26, 26, 464)  0           stage3/block1/relu_1x1conv_2[0][0
                                                                     stage3/block1/relu_1x1conv_3[0][0
    __________________________________________________________________________________________________
    stage3/block1/channel_shuffle ( (None, 26, 26, 464)  0           stage3/block1/concat_2[0][0]
    __________________________________________________________________________________________________
    stage3/block2/spl/sp1_slice (La (None, 26, 26, 232)  0           stage3/block1/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block2/1x1conv_1 (Conv2D (None, 26, 26, 232)  54056       stage3/block2/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block2/bn_1x1conv_1 (Bat (None, 26, 26, 232)  928         stage3/block2/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block2/relu_1x1conv_1 (A (None, 26, 26, 232)  0           stage3/block2/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block2/3x3dwconv (Depthw (None, 26, 26, 232)  2320        stage3/block2/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage3/block2/bn_3x3dwconv (Bat (None, 26, 26, 232)  928         stage3/block2/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block2/1x1conv_2 (Conv2D (None, 26, 26, 232)  54056       stage3/block2/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block2/bn_1x1conv_2 (Bat (None, 26, 26, 232)  928         stage3/block2/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block2/relu_1x1conv_2 (A (None, 26, 26, 232)  0           stage3/block2/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block2/spl/sp0_slice (La (None, 26, 26, 232)  0           stage3/block1/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block2/concat_1 (Concate (None, 26, 26, 464)  0           stage3/block2/relu_1x1conv_2[0][0
                                                                     stage3/block2/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block2/channel_shuffle ( (None, 26, 26, 464)  0           stage3/block2/concat_1[0][0]
    __________________________________________________________________________________________________
    stage3/block3/spl/sp1_slice (La (None, 26, 26, 232)  0           stage3/block2/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block3/1x1conv_1 (Conv2D (None, 26, 26, 232)  54056       stage3/block3/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block3/bn_1x1conv_1 (Bat (None, 26, 26, 232)  928         stage3/block3/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block3/relu_1x1conv_1 (A (None, 26, 26, 232)  0           stage3/block3/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block3/3x3dwconv (Depthw (None, 26, 26, 232)  2320        stage3/block3/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage3/block3/bn_3x3dwconv (Bat (None, 26, 26, 232)  928         stage3/block3/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block3/1x1conv_2 (Conv2D (None, 26, 26, 232)  54056       stage3/block3/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block3/bn_1x1conv_2 (Bat (None, 26, 26, 232)  928         stage3/block3/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block3/relu_1x1conv_2 (A (None, 26, 26, 232)  0           stage3/block3/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block3/spl/sp0_slice (La (None, 26, 26, 232)  0           stage3/block2/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block3/concat_1 (Concate (None, 26, 26, 464)  0           stage3/block3/relu_1x1conv_2[0][0
                                                                     stage3/block3/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block3/channel_shuffle ( (None, 26, 26, 464)  0           stage3/block3/concat_1[0][0]
    __________________________________________________________________________________________________
    stage3/block4/spl/sp1_slice (La (None, 26, 26, 232)  0           stage3/block3/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block4/1x1conv_1 (Conv2D (None, 26, 26, 232)  54056       stage3/block4/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block4/bn_1x1conv_1 (Bat (None, 26, 26, 232)  928         stage3/block4/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block4/relu_1x1conv_1 (A (None, 26, 26, 232)  0           stage3/block4/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block4/3x3dwconv (Depthw (None, 26, 26, 232)  2320        stage3/block4/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage3/block4/bn_3x3dwconv (Bat (None, 26, 26, 232)  928         stage3/block4/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block4/1x1conv_2 (Conv2D (None, 26, 26, 232)  54056       stage3/block4/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block4/bn_1x1conv_2 (Bat (None, 26, 26, 232)  928         stage3/block4/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block4/relu_1x1conv_2 (A (None, 26, 26, 232)  0           stage3/block4/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block4/spl/sp0_slice (La (None, 26, 26, 232)  0           stage3/block3/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block4/concat_1 (Concate (None, 26, 26, 464)  0           stage3/block4/relu_1x1conv_2[0][0
                                                                     stage3/block4/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block4/channel_shuffle ( (None, 26, 26, 464)  0           stage3/block4/concat_1[0][0]
    __________________________________________________________________________________________________
    stage3/block5/spl/sp1_slice (La (None, 26, 26, 232)  0           stage3/block4/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block5/1x1conv_1 (Conv2D (None, 26, 26, 232)  54056       stage3/block5/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block5/bn_1x1conv_1 (Bat (None, 26, 26, 232)  928         stage3/block5/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block5/relu_1x1conv_1 (A (None, 26, 26, 232)  0           stage3/block5/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block5/3x3dwconv (Depthw (None, 26, 26, 232)  2320        stage3/block5/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage3/block5/bn_3x3dwconv (Bat (None, 26, 26, 232)  928         stage3/block5/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block5/1x1conv_2 (Conv2D (None, 26, 26, 232)  54056       stage3/block5/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block5/bn_1x1conv_2 (Bat (None, 26, 26, 232)  928         stage3/block5/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block5/relu_1x1conv_2 (A (None, 26, 26, 232)  0           stage3/block5/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block5/spl/sp0_slice (La (None, 26, 26, 232)  0           stage3/block4/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block5/concat_1 (Concate (None, 26, 26, 464)  0           stage3/block5/relu_1x1conv_2[0][0
                                                                     stage3/block5/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block5/channel_shuffle ( (None, 26, 26, 464)  0           stage3/block5/concat_1[0][0]
    __________________________________________________________________________________________________
    stage3/block6/spl/sp1_slice (La (None, 26, 26, 232)  0           stage3/block5/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block6/1x1conv_1 (Conv2D (None, 26, 26, 232)  54056       stage3/block6/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block6/bn_1x1conv_1 (Bat (None, 26, 26, 232)  928         stage3/block6/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block6/relu_1x1conv_1 (A (None, 26, 26, 232)  0           stage3/block6/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block6/3x3dwconv (Depthw (None, 26, 26, 232)  2320        stage3/block6/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage3/block6/bn_3x3dwconv (Bat (None, 26, 26, 232)  928         stage3/block6/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block6/1x1conv_2 (Conv2D (None, 26, 26, 232)  54056       stage3/block6/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block6/bn_1x1conv_2 (Bat (None, 26, 26, 232)  928         stage3/block6/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block6/relu_1x1conv_2 (A (None, 26, 26, 232)  0           stage3/block6/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block6/spl/sp0_slice (La (None, 26, 26, 232)  0           stage3/block5/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block6/concat_1 (Concate (None, 26, 26, 464)  0           stage3/block6/relu_1x1conv_2[0][0
                                                                     stage3/block6/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block6/channel_shuffle ( (None, 26, 26, 464)  0           stage3/block6/concat_1[0][0]
    __________________________________________________________________________________________________
    stage3/block7/spl/sp1_slice (La (None, 26, 26, 232)  0           stage3/block6/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block7/1x1conv_1 (Conv2D (None, 26, 26, 232)  54056       stage3/block7/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block7/bn_1x1conv_1 (Bat (None, 26, 26, 232)  928         stage3/block7/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block7/relu_1x1conv_1 (A (None, 26, 26, 232)  0           stage3/block7/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block7/3x3dwconv (Depthw (None, 26, 26, 232)  2320        stage3/block7/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage3/block7/bn_3x3dwconv (Bat (None, 26, 26, 232)  928         stage3/block7/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block7/1x1conv_2 (Conv2D (None, 26, 26, 232)  54056       stage3/block7/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block7/bn_1x1conv_2 (Bat (None, 26, 26, 232)  928         stage3/block7/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block7/relu_1x1conv_2 (A (None, 26, 26, 232)  0           stage3/block7/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block7/spl/sp0_slice (La (None, 26, 26, 232)  0           stage3/block6/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block7/concat_1 (Concate (None, 26, 26, 464)  0           stage3/block7/relu_1x1conv_2[0][0
                                                                     stage3/block7/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block7/channel_shuffle ( (None, 26, 26, 464)  0           stage3/block7/concat_1[0][0]
    __________________________________________________________________________________________________
    stage3/block8/spl/sp1_slice (La (None, 26, 26, 232)  0           stage3/block7/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block8/1x1conv_1 (Conv2D (None, 26, 26, 232)  54056       stage3/block8/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block8/bn_1x1conv_1 (Bat (None, 26, 26, 232)  928         stage3/block8/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block8/relu_1x1conv_1 (A (None, 26, 26, 232)  0           stage3/block8/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage3/block8/3x3dwconv (Depthw (None, 26, 26, 232)  2320        stage3/block8/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage3/block8/bn_3x3dwconv (Bat (None, 26, 26, 232)  928         stage3/block8/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block8/1x1conv_2 (Conv2D (None, 26, 26, 232)  54056       stage3/block8/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage3/block8/bn_1x1conv_2 (Bat (None, 26, 26, 232)  928         stage3/block8/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block8/relu_1x1conv_2 (A (None, 26, 26, 232)  0           stage3/block8/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage3/block8/spl/sp0_slice (La (None, 26, 26, 232)  0           stage3/block7/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage3/block8/concat_1 (Concate (None, 26, 26, 464)  0           stage3/block8/relu_1x1conv_2[0][0
                                                                     stage3/block8/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage3/block8/channel_shuffle ( (None, 26, 26, 464)  0           stage3/block8/concat_1[0][0]
    __________________________________________________________________________________________________
    stage4/block1/1x1conv_1 (Conv2D (None, 26, 26, 464)  215760      stage3/block8/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage4/block1/bn_1x1conv_1 (Bat (None, 26, 26, 464)  1856        stage4/block1/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage4/block1/relu_1x1conv_1 (A (None, 26, 26, 464)  0           stage4/block1/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage4/block1/3x3dwconv (Depthw (None, 13, 13, 464)  4640        stage4/block1/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage4/block1/3x3dwconv_2 (Dept (None, 13, 13, 464)  4640        stage3/block8/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage4/block1/bn_3x3dwconv (Bat (None, 13, 13, 464)  1856        stage4/block1/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage4/block1/bn_3x3dwconv_2 (B (None, 13, 13, 464)  1856        stage4/block1/3x3dwconv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block1/1x1conv_2 (Conv2D (None, 13, 13, 464)  215760      stage4/block1/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage4/block1/1x1_conv_3 (Conv2 (None, 13, 13, 464)  215760      stage4/block1/bn_3x3dwconv_2[0][0
    __________________________________________________________________________________________________
    stage4/block1/bn_1x1conv_2 (Bat (None, 13, 13, 464)  1856        stage4/block1/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block1/bn_1x1conv_3 (Bat (None, 13, 13, 464)  1856        stage4/block1/1x1_conv_3[0][0]
    __________________________________________________________________________________________________
    stage4/block1/relu_1x1conv_2 (A (None, 13, 13, 464)  0           stage4/block1/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block1/relu_1x1conv_3 (A (None, 13, 13, 464)  0           stage4/block1/bn_1x1conv_3[0][0]
    __________________________________________________________________________________________________
    stage4/block1/concat_2 (Concate (None, 13, 13, 928)  0           stage4/block1/relu_1x1conv_2[0][0
                                                                     stage4/block1/relu_1x1conv_3[0][0
    __________________________________________________________________________________________________
    stage4/block1/channel_shuffle ( (None, 13, 13, 928)  0           stage4/block1/concat_2[0][0]
    __________________________________________________________________________________________________
    stage4/block2/spl/sp1_slice (La (None, 13, 13, 464)  0           stage4/block1/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage4/block2/1x1conv_1 (Conv2D (None, 13, 13, 464)  215760      stage4/block2/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage4/block2/bn_1x1conv_1 (Bat (None, 13, 13, 464)  1856        stage4/block2/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage4/block2/relu_1x1conv_1 (A (None, 13, 13, 464)  0           stage4/block2/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage4/block2/3x3dwconv (Depthw (None, 13, 13, 464)  4640        stage4/block2/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage4/block2/bn_3x3dwconv (Bat (None, 13, 13, 464)  1856        stage4/block2/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage4/block2/1x1conv_2 (Conv2D (None, 13, 13, 464)  215760      stage4/block2/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage4/block2/bn_1x1conv_2 (Bat (None, 13, 13, 464)  1856        stage4/block2/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block2/relu_1x1conv_2 (A (None, 13, 13, 464)  0           stage4/block2/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block2/spl/sp0_slice (La (None, 13, 13, 464)  0           stage4/block1/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage4/block2/concat_1 (Concate (None, 13, 13, 928)  0           stage4/block2/relu_1x1conv_2[0][0
                                                                     stage4/block2/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage4/block2/channel_shuffle ( (None, 13, 13, 928)  0           stage4/block2/concat_1[0][0]
    __________________________________________________________________________________________________
    stage4/block3/spl/sp1_slice (La (None, 13, 13, 464)  0           stage4/block2/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage4/block3/1x1conv_1 (Conv2D (None, 13, 13, 464)  215760      stage4/block3/spl/sp1_slice[0][0]
    __________________________________________________________________________________________________
    stage4/block3/bn_1x1conv_1 (Bat (None, 13, 13, 464)  1856        stage4/block3/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage4/block3/relu_1x1conv_1 (A (None, 13, 13, 464)  0           stage4/block3/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage4/block3/3x3dwconv (Depthw (None, 13, 13, 464)  4640        stage4/block3/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage4/block3/bn_3x3dwconv (Bat (None, 13, 13, 464)  1856        stage4/block3/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage4/block3/1x1conv_2 (Conv2D (None, 13, 13, 464)  215760      stage4/block3/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage4/block3/bn_1x1conv_2 (Bat (None, 13, 13, 464)  1856        stage4/block3/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block3/relu_1x1conv_2 (A (None, 13, 13, 464)  0           stage4/block3/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block3/spl/sp0_slice (La (None, 13, 13, 464)  0           stage4/block2/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage4/block3/concat_1 (Concate (None, 13, 13, 928)  0           stage4/block3/relu_1x1conv_2[0][0
                                                                     stage4/block3/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage4/block3/channel_shuffle ( (None, 13, 13, 928)  0           stage4/block3/concat_1[0][0]
    __________________________________________________________________________________________________
    stage4/block4/spl/sp1_slice (La (None, 13, 13, 464)  0           stage4/block3/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage4/block4/1x1conv_1 (Conv2D (None, 13, 13, 464)  215760      stage4/block4/spl/sp1_slice[0][0]
    stage4/block4/bn_1x1conv_1 (Bat (None, 13, 13, 464)  1856        stage4/block4/1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage4/block4/relu_1x1conv_1 (A (None, 13, 13, 464)  0           stage4/block4/bn_1x1conv_1[0][0]
    __________________________________________________________________________________________________
    stage4/block4/3x3dwconv (Depthw (None, 13, 13, 464)  4640        stage4/block4/relu_1x1conv_1[0][0
    __________________________________________________________________________________________________
    stage4/block4/bn_3x3dwconv (Bat (None, 13, 13, 464)  1856        stage4/block4/3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage4/block4/1x1conv_2 (Conv2D (None, 13, 13, 464)  215760      stage4/block4/bn_3x3dwconv[0][0]
    __________________________________________________________________________________________________
    stage4/block4/bn_1x1conv_2 (Bat (None, 13, 13, 464)  1856        stage4/block4/1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block4/relu_1x1conv_2 (A (None, 13, 13, 464)  0           stage4/block4/bn_1x1conv_2[0][0]
    __________________________________________________________________________________________________
    stage4/block4/spl/sp0_slice (La (None, 13, 13, 464)  0           stage4/block3/channel_shuffle[0][
    __________________________________________________________________________________________________
    stage4/block4/concat_1 (Concate (None, 13, 13, 928)  0           stage4/block4/relu_1x1conv_2[0][0
                                                                     stage4/block4/spl/sp0_slice[0][0]
    __________________________________________________________________________________________________
    stage4/block4/channel_shuffle ( (None, 13, 13, 928)  0           stage4/block4/concat_1[0][0]
    __________________________________________________________________________________________________
    1x1conv5_out (Conv2D)           (None, 13, 13, 1024) 951296      stage4/block4/channel_shuffle[0][
    __________________________________________________________________________________________________
    global_max_pool (GlobalMaxPooli (None, 1024)         0           1x1conv5_out[0][0]
    """

    # net, endpoint = inception_v2.inception_v2(inputs)
    shufflenet = ShuffleNetV2(input_tensor=inputs,
                              input_shape=None,
                              scale_factor=1.0,
                              pooling='max',
                              num_shuffle_units=[3, 7, 3],
                              bottleneck_ratio=1.5,
                              classes=num_classes)

    # input: 416 x 416 x 3
    # stage3/block1/relu_1x1conv_1: 52 x 52 x [96/232/352/488]
    # stage4/block1/relu_1x1conv_1: 26 x 26 x [192/464/704/976]
    # 1x1conv5_out: 13 x 13 x [1024/1024/1024/2048]

    f1 = shufflenet.get_layer('1x1conv5_out').output
    # f1 :13 x 13 x [1024/1024/1024/2048]
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)

    f2 = shufflenet.get_layer('stage4/block1/relu_1x1conv_1').output
    # f2: 26 x 26 x [192/464/704/976]
    x = Concatenate()([x, f2])

    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)

    f3 = shufflenet.get_layer('stage3/block1/relu_1x1conv_1').output
    # f3 : 52 x 52 x [96/232/352/488]
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs=inputs, outputs=[y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""

    num_layers = len(yolo_outputs)

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    # print("yolo_outputs",yolo_outputs)
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    return loss
