import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.framework import arg_scope
import numpy as np
import os

# from Anatomy_Recon.fullyADMM import Input_brainReal_2_5D as inp
import Input_brainReal_2_5D as inp

import time

# Hyperparameter
growth_k = 16
nb_block = 4 # how many (dense block + Transition Layer) ?
init_learning_rate = 5e-5
epsilon = 1e-8 # AdamOptimizer epsilon
dropout_rate = 0.2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.flags.DEFINE_integer('batch_size_eval', 1,
                            """Number of images to process in a batch.""")
tf.flags.DEFINE_string('data_dir1', r'D:\SKK_DL\Unfold_MLAA\data\train_data_1130_2',
                           """Path to the data directory.""")


def inputs_eval(tdata_dir):
    [oinput1, oinput2, oinput3, oinput4, oinput5, oinput6, ok] = inp.inputs_eval(data_dir=tdata_dir, batch_size=FLAGS.batch_size_eval)
    return oinput1, oinput2, oinput3, oinput4, oinput5, oinput6, ok


def distorted_inputs():
    data_dir1 = os.path.join(FLAGS.data_dir1)
    [xlr1, xlr2, xlr3, xlr4, xlr5, xlr6] = inp.distorted_inputs(data_dir=[data_dir1], batch_size=FLAGS.batch_size)
    return xlr1, xlr2, xlr3, xlr4, xlr5, xlr6


def _activation_image(x, slice, name):
    x_out = x[0, :, :, slice]
    tf.summary.image(name,
                     tf.expand_dims(tf.expand_dims(x_out, -1), 0), max_outputs=1)

def _activation_image_2d(x, slice, name):
    x_out = x[0, :, :, slice]
    tf.summary.image(name,
                     tf.expand_dims(tf.expand_dims(x_out, -1), 0), max_outputs=1)


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(w, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.contrib.layers.xavier_initializer(), trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = l2_norm(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = l2_norm(u_)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
   w_norm = w / sigma

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm

def conv_layer(input, filter, kernel, stride=1, layer_name="conv", first=False, use_bias = True):
    with tf.variable_scope(layer_name):
        if use_bias is False:
            return tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME', kernel_initializer=tf.initializers.truncated_normal(stddev=0.01), use_bias=False)
        else:
            return  tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME', kernel_initializer=tf.initializers.truncated_normal(stddev=0.01))


def conv_layer_spectral(x, filter, kernel, stride=1, use_bias=True, padding='SAME', layer_name="spectral_conv"):
    with tf.variable_scope(layer_name):
        w = tf.get_variable("kernel", shape=[kernel[0], kernel[1], x.get_shape()[-1], filter], initializer=tf.contrib.layers.xavier_initializer(), regularizer=None)
        bias = tf.get_variable("bias", [filter], initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w),  strides=[1, stride, stride, 1], padding=padding)
        if use_bias:
            x = tf.nn.bias_add(x, bias)
    return x



def dense_layer_spectral(x, filter, use_bias=False, layer_name="spectral_dense"):
    with tf.variable_scope(layer_name):
        w = tf.get_variable("kernel", shape=[x.get_shape()[-1], filter], initializer=tf.contrib.layers.xavier_initializer(), regularizer=None)
        bias = tf.get_variable("bias", [filter], initializer=tf.constant_initializer(0.0))
        x = tf.matmul(x, w)
        if use_bias:
            x = tf.nn.bias_add(x, bias)
    return x


def conv_layer_up(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.variable_scope(layer_name):
        network = tf.layers.conv2d_transpose(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        return network

# def conv_layer_up_spectral(x, filter, kernel, stride=(1, 1), use_bias=True, padding='SAME', layer_name="spectral_conv", upsample=True):
#     xdim = x.get_shape().as_list()
#     with tf.variable_scope(layer_name):
#         w = tf.get_variable("kernel", shape=[kernel[0], kernel[1], filter, x.get_shape()[-1]], initializer=tf.contrib.layers.xavier_initializer(), regularizer=None)
#         bias = tf.get_variable("bias", [filter], initializer=tf.constant_initializer(0.0))
#         x = tf.nn.conv2d_transpose(value=x,
#                                    filter=spectral_norm(w),
#                                    strides=[1, 1, 1, 1],
#                                    padding=padding,
#                                    output_shape=[xdim[0], xdim[1], xdim[2], filter])
#         if use_bias:
#             x = tf.nn.bias_add(x, bias)
#         if upsample == True:
#             x = ops.up_sample(x, 2)
#
#         return x

def Batch_Normalization(x, scope, groups=32, reuse=False):
    # return x
    # return tf.layers.batch_normalization(x,
    #                              training=training,
    #                              reuse=reuse,
    #                              fused=True,
    #                              renorm=False,
    #                              name=scope)
    return tf.contrib.layers.group_norm(x, scope = scope, reuse = reuse, groups=groups)

def Drop_out(x, rate, training) :
    return x
    # return tf.layers.dropout(inputs=x, rate=rate, training=training)

def total_variation(images, name=None):

    # Image must be 3D!!!!!!!!!!!!!!
    # SKKang 1102
    # 4D input: [batch_size x, y, z]

    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif3 = images[:, :, :, 1:] - images[:, :, :, :-1]

    # Only sum for the last 3 axis.
    # This results in a 1-D tensor with the total variation for each image.
    sum_axis = [1, 2, 3]

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (
        # tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) +
        # tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis) +
        tf.reduce_sum(tf.abs(pixel_dif3), axis=sum_axis)
    )

    return tot_var


def Relu(x):
    return tf.nn.relu(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=-1)

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1,2,3))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

  return 1 - (numerator + 1) / (denominator + 1 + 1e-8)

def pls_loss(in1, in2):
    pixel_dif1_1 = in1[:, 1:, :, :] - in1[:, :-1, :, :]
    pixel_dif1_1 = pixel_dif1_1[:,0:119,0:119,:]
    pixel_dif2_1 = in1[:, :, 1:, :] - in1[:, :, :-1, :]
    pixel_dif2_1 = pixel_dif2_1[:, 0:119, 0:119, :]

    pixel_dif1_2 = in2[:, 1:, :, :] - in2[:, :-1, :, :]
    pixel_dif1_2 = pixel_dif1_2[:, 0:119, 0:119, :]
    pixel_dif2_2 = in2[:, :, 1:, :] - in2[:, :, :-1, :]
    pixel_dif2_2 = pixel_dif2_2[:, 0:119, 0:119, :]

    cos = tf.divide(tf.multiply(pixel_dif1_1, pixel_dif1_2) + tf.multiply(pixel_dif2_1, pixel_dif2_2),
                    1e-9 + tf.multiply(tf.sqrt(tf.square(pixel_dif1_1) + tf.square(pixel_dif1_2)),
                                       tf.sqrt(tf.square(pixel_dif2_1) + tf.square(pixel_dif2_2))))
    sin = tf.sqrt(1 - tf.square(cos))

    out = tf.reduce_mean(
        tf.reduce_sum(tf.multiply(tf.sqrt(tf.square(pixel_dif1_1) + tf.square(pixel_dif1_2)),
                                    tf.sqrt(tf.square(pixel_dif2_1) + tf.square(pixel_dif2_2))) * sin, axis=[1, 2, 3])
    )

    out = tf.where(tf.is_nan(out), tf.ones_like(out) * 0, out)

    return out, sin

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def attention(x, ch, layer_name):
    with tf.variable_scope(layer_name):
        batch_size, height, width, num_channels = x.get_shape().as_list()
        f = conv_layer_spectral(x, filter = ch // 8, kernel=[1,1], stride=2, layer_name='f_conv') # [bs, h, w, c']
        g = conv_layer_spectral(x, filter = ch // 8, kernel=[1,1], stride=1, layer_name='g_conv') # [bs, h, w, c']
        h = conv_layer_spectral(x, filter = ch // 2, kernel=[1,1], stride=2, layer_name='h_conv') # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
        o = conv_layer_spectral(o, ch, kernel=[1, 1], stride=1, layer_name='o_conv')  # [bs, h, w, c']

        x = gamma * o + x

        return x

def mish(x, name='mish'):
    return x * tf.nn.tanh(tf.nn.softplus(x))

def PReLU(_x, name):
    # _alpha = tf.get_variable(name, shape=_x.get_shape()[-1],
    #                          dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
    # return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
    return mish(_x, name=name)

def Channel_attention(name, x, filter=64):
    _res = x

    x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    x = conv_layer(x, kernel=1, filter=filter/16, layer_name=name + '_conv1', first=True)
    x = tf.contrib.layers.group_norm(x, scope=name+'GN1', groups=8)
    x = PReLU(x, name=name + 'PR1')

    x = conv_layer(x, kernel=1, filter=filter, layer_name=name + '_conv2', first=True)
    x=tf.contrib.layers.group_norm(x, scope=name+'GN2')
    x = tf.nn.sigmoid(x)
    x = tf.multiply(x, _res)

    return x

def rbloc(x, name, filter=64, keep_prob=0.8):
    with tf.variable_scope(name):
        _res = x

        x = conv_layer(x, kernel=3, filter=filter, layer_name='_conv1')
        x= Batch_Normalization(x, scope = 'bn1')
        x = PReLU(x, name='PR1')
        x = tf.nn.dropout(x, keep_prob = keep_prob)

        x = conv_layer(x, kernel=3, filter=filter, layer_name='_conv2')
        x = Batch_Normalization(x, scope = 'bn2')
        x = PReLU(x, name='PR2')
        x = tf.nn.dropout(x, keep_prob=keep_prob)
        # x = Channel_attention(name + '_CA', x, filter=filter)

        x = PReLU(x + Batch_Normalization(conv_layer(_res, kernel=3, filter=filter, layer_name='_conv3'), scope='bn3'), name='PR3')
        x = tf.nn.dropout(x, keep_prob=keep_prob)

    return x

def Up_scaling(x,name, n_feats):
    ## if scale is 2^n
    with tf.variable_scope(name):
        x = conv_layer(x, kernel=3, filter=2 * 2 * n_feats)
        x = Batch_Normalization(x, scope='bn1')
        x = PReLU(x, name='PR1')
        x = tf.depth_to_space(x, 2)
    return x


# def unfold_resnet(input, layer_name='unfold', scale=1, isfinal = False):
#     with tf.variable_scope(layer_name):
#         x = PReLU(conv_layer(input, filter=64, kernel=3, layer_name='conv1_1'), name='pr1')
#         x = conv_layer(x, filter=64, kernel=3, layer_name='conv2_1')
#         x *= scale
#
#         input2 = input + x
#
#         if not isfinal:
#             x = PReLU(conv_layer(input2, filter=64, kernel=3, layer_name='conv1_2'), name = 'pr2')
#             x = conv_layer(x, filter=64, kernel=3, layer_name='conv2_2')
#             x *= scale
#
#
#             x = input2 + x
#
#             x = conv_layer(x, filter=32, kernel=3, layer_name='conv_down')
#
#         else:
#             x = PReLU(conv_layer(input2, filter=64, kernel=3, layer_name='conv1_2'), name = 'pr3')
#             x = conv_layer(x, filter=64, kernel=3, layer_name='conv2_2')
#             x *= scale
#
#             x = tf.ninput2 + x
#
#             x = PReLU(conv_layer(x, filter = 64*4, kernel=3, layer_name='conv_up'), name = 'pr4')
#             x = tf.depth_to_space(x, 2)
#
#             x = conv_layer(x, filter=32, kernel=3, layer_name='conv2_out_f')
#
#         return tf.nn.relu(x)

# def unfold_resnet_grad(input, grad, layer_name='unfold', scale=1, isfinal = False, filter_size=128):
#     with tf.variable_scope(layer_name):
#         input = tf.concat([input, grad], axis=-1)
#         x = conv_layer(input, filter=filter_size, kernel=3, layer_name='conv1_1', first=True)
#
#         rbloc(x, name='rb1')
#         rbloc(x, name='rb2')
#
#         x = conv_layer(x, filter=1, kernel=3, layer_name='conv_down')
#
#         return x
#
# def unfold_resnet_img(input, layer_name='unfold', scale=1, isfinal = False, filter_size=128, keep_prob=0.5):
#     with tf.variable_scope(layer_name):
#         x = conv_layer(input, filter=filter_size, kernel=3, layer_name='conv1_1', first=True)
#
#         x = rbloc(x, name='rb1', keep_prob=keep_prob)
#         x = rbloc(x, name='rb2', keep_prob=keep_prob)
#         x = rbloc(x, name='rb3', keep_prob=keep_prob)
#         x = rbloc(x, name='rb4', keep_prob=keep_prob)
#
#         # if isfinal:
#         #     x = conv_layer(x, filter=3, kernel=3, layer_name='conv_down')
#         # else:
#         x = conv_layer(x, filter=1, kernel=3, layer_name='conv_down')
#         return x+input

def unfold_resnet_xup(input, input2, layer_name='unfold', scale=1, isfinal=False, filter_size=128, keep_prob=0.5):
    with tf.variable_scope(layer_name):
        x =  PReLU(Batch_Normalization(tf.concat([input, input2], axis=-1), scope='BNin', groups=1), name='PR1_2')
        x = conv_layer(x, filter=filter_size, kernel=3, layer_name='conv1_1', first=True)
        x = PReLU(Batch_Normalization(x, scope='BN1'), name='PR1')
        # x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)

        res = x

        x = conv_layer(x, filter=filter_size*2, kernel=3,stride=2, layer_name='conv1_2', first=True)
        x = PReLU(Batch_Normalization(x, scope='BN1_2'), name='PR1_2')

        xd1 = rbloc(x, name='rb1', keep_prob=keep_prob, filter=filter_size*2)
        xd2 = conv_layer(xd1, filter=filter_size * 2, kernel=3, stride=2, layer_name='convdown', first=True)
        xd3 = rbloc(xd2, name='rb2', keep_prob=keep_prob, filter=filter_size*4)
        xd4 = tf.image.resize_bilinear(xd3, [60, 60])
        x = rbloc(tf.concat([xd4, xd1], axis=-1), name='rb22', keep_prob=keep_prob, filter=filter_size * 2)
        # x = rbloc(x, name='rb3', keep_prob=keep_prob, filter=filter_size)
        # x = rbloc(x, name='rb4', keep_prob=keep_prob, filter=filter_size*2)

        x = tf.image.resize_bilinear(x, [120, 120])
        x = conv_layer(tf.concat([x, res], axis=-1), filter=filter_size, kernel=3, layer_name='conv12_2', first=True)
        xf = PReLU(Batch_Normalization(x, scope='BN12_2'), name='PR12_2')
        # Pixel Shuffler instead of bilinear upsampling
        # xf = Up_scaling(xf, name='up1', n_feats=filter_size)

        x1 = conv_layer(xf, filter=filter_size, kernel=3, layer_name='conv_down1_1', first=True)
        x1 = PReLU(Batch_Normalization(x1, scope='BN2'), name='PR2')
        x1 = conv_layer(x1, filter=3, kernel=3, layer_name='conv_down1_2', first=True)
        # x1 = tf.nn.relu(Batch_Normalization(x1, scope='BN2f', groups=1), name='PR2')
        # x2 = conv_layer(xf, filter=filter_size, kernel=3, layer_name='conv_down2_1', first=True)
        # x2 = PReLU(Batch_Normalization(x2, scope='BN2_2'), name='PR2_2')
        # x2 = conv_layer(x2, filter=3, kernel=3, layer_name='conv_down2_2', first=True)

        return x1


def unfold_resnet_zup(input, layer_name='unfold', scale=1, isfinal=False, filter_size=64, keep_prob=0.5):
    with tf.variable_scope(layer_name):
        x = conv_layer(input, filter=filter_size, kernel=3, layer_name='conv1_1', first=True)
        x = PReLU(Batch_Normalization(x, scope='BN1'), name='PR1')

        x = rbloc(x, name='rb1', keep_prob=keep_prob, filter=filter_size)
        x = rbloc(x, name='rb2', keep_prob=keep_prob, filter=filter_size)
        # x = rbloc(x, name='rb3', keep_prob=keep_prob, filter=filter_size)
        # x = rbloc(x, name='rb4', keep_prob=keep_prob, filter=filter_size)

        x = conv_layer(x, filter=3, kernel=3, layer_name='conv_down', first=True)

        return x

class Discrim_CNN():
    def __init__(self, x, y1, reuse):
        self.model = self.DiscriminatorCNN(x,y1, reuse=reuse)

    def DiscriminatorCNN(self, x, y1, reuse):
        with tf.variable_scope("dis_d2e", reuse=reuse) as vs:
            # Encoder
            x = tf.concat([x, y1], axis=-1)
            x0 = conv_layer_spectral(x, 16, [3, 3], 1, padding='SAME', layer_name='c1')
            x1 = tf.nn.leaky_relu(x0)
            x2 = conv_layer_spectral(x1, 32, [3, 3], 1, padding='SAME', layer_name='c2')
            # x2 = x2 + x0
            x3 = tf.nn.leaky_relu(x2)

            x4 = conv_layer_spectral(x3, 64, [3, 3], 2, padding='SAME', layer_name='c3')
            x5 = tf.nn.leaky_relu(x4)
            x6 = conv_layer_spectral(x5, 64, [3, 3], 1, padding='SAME', layer_name='c4')
            x6 = x6 + x4
            x7 = tf.nn.leaky_relu(x6)

            x8 = conv_layer_spectral(x7, 128, [3, 3], 2, padding='SAME', layer_name='c5')
            x9 = tf.nn.leaky_relu(x8)
            x10 = conv_layer_spectral(x9, 128, [3, 3], 1, padding='SAME', layer_name='c6')
            x10 = x10 + x8
            x11 = tf.nn.leaky_relu(x10)

            x12 = conv_layer_spectral(x11, 256, [3, 3], 1, padding='SAME', layer_name='c7')
            x13 = tf.nn.leaky_relu(x12)
            x14 = conv_layer_spectral(x13, 256, [3, 3], 1, padding='SAME', layer_name='c8')
            x14 = x14 + x12
            x15 = tf.nn.leaky_relu(x14)

            x16 = conv_layer_spectral(x15, 512, [3, 3], 1, padding='VALID', layer_name='c9')
            x17 = tf.nn.leaky_relu(x16)
            x18 = conv_layer_spectral(x17, 512, [3, 3], 1, padding='VALID', layer_name='c10')
            x18 = tf.nn.leaky_relu(x18)

            x = conv_layer_spectral(x18, 1, [3, 3], 1, padding='VALID', layer_name='cf')
            # x = tf.reduce_mean(x18, axis=[1, 2])
            # x = dense_layer_spectral(x, 1, use_bias=False)

            #
            return x