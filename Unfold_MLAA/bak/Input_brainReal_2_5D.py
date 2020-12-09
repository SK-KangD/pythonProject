
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import random

from matplotlib import pyplot as plt
import pylab


NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4993

##############################
#   DATA STRUCTURE
#   LR, HR, slice_number, maxi
#   150x208 for Pyton
##############################




# get filenaems with _i and _y
def get_Data_rnd(dirpath):
    fulln = []
    flist = os.listdir(dirpath)
    for filenames in flist:
        fulln.append(os.path.join(dirpath, filenames))
    # r = random.random()
    # random.shuffle(fulln, lambda : r)

    return fulln

#################################################################
# SKK_0428_READING FILES
# Data declaration by class
def read_2dSKK(iname_queue):
    class np2dSets(object):
        pass

    result = np2dSets()
    result.height = 120
    result.width = 120
    result.tall = 109

    ini1 = 128*128*120
    ini2 = 64*64*64

    ireader = tf.WholeFileReader()

    key, value = ireader.read(iname_queue)

    reading_bytes = tf.decode_raw(value, out_type=tf.float32)

    ## original implementation
    # result.lr = tf.reshape(tf.strided_slice(reading_bytes, [0], [ini]), [result.height, result.width, result.tall])
    # result.hr = tf.reshape(tf.slice(reading_bytes, [ini], [ini]), [result.height, result.width, result.tall])

    # Stocahstic training
    # result.input1 = tf.reshape(tf.slice(reading_bytes, [0], [172*252*3]), [3, 252, 172])
    # result.input2 = tf.reshape(tf.slice(reading_bytes, [172*252*3], [344*344*3]), [3, 344, 344])
    # result.input3 = tf.reshape(tf.slice(reading_bytes, [172*252*3 + 344*344*3], [344*344*3]), [3, 344, 344])
    result.input1 = tf.reshape(tf.slice(reading_bytes, [0], [120 * 168 * 3]), [3, 168, 120])
    result.input2 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 1], [120 * 168 * 3]), [3, 168, 120])
    result.input3 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 2], [120 * 120 * 3]), [3, 120, 120])
    result.input4 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 2 + 120 * 120 * 3 * 1], [120 * 120 * 3]), [3, 120, 120])
    result.input5 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 2 + 120 * 120 * 3 * 2], [120 * 120 * 3]), [3, 120, 120])
    result.input6 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 2 + 120 * 120 * 3 * 3], [120 * 120 * 3]), [3, 120, 120])

    # result.slc = tf.cast(result.slc, dtype=tf.float32)
    # result.slc = tf.divide(result.slc, 10)

    return result, key


def read_2dSKK_eval(iname_queue):
    class np2dSets(object):
        pass

    result = np2dSets()
    result.height = 128
    result.width = 128
    result.tall = 128

    ini1 = 128 * 128 * 120
    ini2 = 64 * 64 * 64

    ireader = tf.WholeFileReader()

    key, value = ireader.read(iname_queue)

    reading_bytes = tf.decode_raw(value, out_type=tf.float32)

    ## original implementation
    # result.lr = tf.reshape(tf.strided_slice(reading_bytes, [0], [ini]), [result.height, result.width, result.tall])
    # result.hr = tf.reshape(tf.slice(reading_bytes, [ini], [ini]), [result.height, result.width, result.tall])

    # Stocahstic training
    result.input1 = tf.reshape(tf.slice(reading_bytes, [0], [120 * 168 * 3]), [3, 168, 120])
    result.input2 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 1], [120 * 168 * 3]), [3, 168, 120])
    result.input3 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 2], [120 * 120 * 3]), [3, 120, 120])
    result.input4 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 2 + 120 * 120 * 3 * 1], [120 * 120 * 3]), [3, 120, 120])
    result.input5 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 2 + 120 * 120 * 3 * 2], [120 * 120 * 3]), [3, 120, 120])
    result.input6 = tf.reshape(tf.slice(reading_bytes, [120 * 168 * 3 * 2 + 120 * 120 * 3 * 3], [120 * 120 * 3]), [3, 120, 120])

    # result.slc = tf.cast(result.slc, dtype=tf.float32)
    # result.slc = tf.divide(result.slc, 10)

    return result, key

def _generate_image_and_label_batch(xlr1, xlr2, xlr3, xlr4, xlr5, xlr6,
                                    min_queue_examples, batch_size, shuffle, eval_test):
    if eval_test == 1:
        num_preprocess_threads = 1
    else:
        num_preprocess_threads = 16

    if shuffle:
        [out_xlr1, out_xlr2, out_xlr3, out_xlr4, out_xlr5, out_xlr6] = tf.train.shuffle_batch(
            [xlr1, xlr2, xlr3, xlr4, xlr5, xlr6],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
            )
    else:
        [out_xlr1, out_xlr2, out_xlr3, out_xlr4, out_xlr5, out_xlr6] = tf.train.batch(
            [xlr1, xlr2, xlr3, xlr4, xlr5, xlr6],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return out_xlr1, out_xlr2, out_xlr3, out_xlr4, out_xlr5, out_xlr6


def _generate_image_and_label_batch_eval(xinput1, xinput2, xinput3, xinput4, xinput5, xinpu6, outkeay,
                                         min_queue_examples, batch_size, shuffle, eval_test):
    if eval_test == 1:
        num_preprocess_threads = 1
    else:
        num_preprocess_threads = 8

    if shuffle:
        [oinput1, oinput2, oinput3, oinput4, oinput5, oinput6, ok] = tf.train.shuffle_batch(
            [xinput1, xinput2, xinput3, xinput4, xinput5, xinpu6, outkeay],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
            )
    else:
        [oinput1, oinput2, oinput3, oinput4, oinput5, oinput6, ok] = tf.train.batch(
            [xinput1, xinput2, xinput3, xinput4, xinput5, xinpu6, outkeay],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)


    return oinput1, oinput2, oinput3, oinput4, oinput5, oinput6, ok


def distorted_inputs(data_dir, batch_size):

    if len(data_dir) != 1:
        inames1 = get_Data_rnd(data_dir[0])
        inames2 = get_Data_rnd(data_dir[1])

        iname_que = tf.train.string_input_producer(inames1 + inames2)
    else:
        inames = get_Data_rnd(data_dir[0])
        iname_que = tf.train.string_input_producer(inames)

    read_input, _ = read_2dSKK(iname_que)

    x_input1 = tf.cast(read_input.input1, tf.float32)
    x_input2 = tf.cast(read_input.input2, tf.float32)
    x_input3 = tf.cast(read_input.input3, tf.float32)
    x_input4 = tf.cast(read_input.input4, tf.float32)
    x_input5 = tf.cast(read_input.input5, tf.float32)
    x_input6 = tf.cast(read_input.input6, tf.float32)


    x_input1.set_shape([3, 168, 120])
    x_input2.set_shape([3, 168, 120])
    x_input3.set_shape([3, 120, 120])
    x_input4.set_shape([3, 120, 120])
    x_input5.set_shape([3, 120, 120])
    x_input6.set_shape([3, 120, 120])


    x_input1 = tf.transpose(x_input1, [1, 2, 0])
    x_input2 = tf.transpose(x_input2, [1, 2, 0])
    x_input3 = tf.transpose(x_input3, [1, 2, 0])
    x_input4 = tf.transpose(x_input4, [1, 2, 0])
    x_input5 = tf.transpose(x_input5, [1, 2, 0])
    x_input6 = tf.transpose(x_input6, [1, 2, 0])

    # maxi = tf.reduce_max(x_input3)
    #
    # x_input1 = x_input1 / maxi
    # x_input2 = x_input2 / maxi
    # x_input3 = x_input3 / maxi
    # x_input4 = x_input4 / tf.reduce_max(x_input4)

    #### added

    min_fraction_of_examples_in_queue = 0.0003
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d distorted pib images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(x_input1, x_input2, x_input3, x_input4, x_input5, x_input6, min_queue_examples, batch_size, shuffle=True, eval_test=False)
    ###################################################################


def inputs_eval(data_dir, batch_size):
    inames = get_Data_rnd(data_dir)

    iname_que = tf.train.string_input_producer(inames, shuffle=False)

    read_input, outkey = read_2dSKK_eval(iname_que)

    x_input1 = tf.cast(read_input.input1, tf.float32)
    x_input2 = tf.cast(read_input.input2, tf.float32)
    x_input3 = tf.cast(read_input.input3, tf.float32)
    x_input4 = tf.cast(read_input.input4, tf.float32)
    x_input5 = tf.cast(read_input.input5, tf.float32)
    x_input6 = tf.cast(read_input.input6, tf.float32)

    x_input1.set_shape([3, 168, 120])
    x_input2.set_shape([3, 168, 120])
    x_input3.set_shape([3, 120, 120])
    x_input4.set_shape([3, 120, 120])
    x_input5.set_shape([3, 120, 120])
    x_input6.set_shape([3, 120, 120])

    x_input1 = tf.transpose(x_input1, [1, 2, 0])
    x_input2 = tf.transpose(x_input2, [1, 2, 0])
    x_input3 = tf.transpose(x_input3, [1, 2, 0])
    x_input4 = tf.transpose(x_input4, [1, 2, 0])
    x_input5 = tf.transpose(x_input5, [1, 2, 0])
    x_input6 = tf.transpose(x_input6, [1, 2, 0])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.3
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d distorted pib images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)


    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch_eval(x_input1, x_input2, x_input3, x_input4, x_input5, x_input6, outkey,
                                           min_queue_examples, batch_size,
                                           shuffle=False, eval_test=True)

