from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from datetime import datetime
import time
import os
import shutil
import numpy as np
import scipy.io


import random
from scipy.ndimage import zoom

import tensorflow as tf

# import tensorlayer as tl

import layers_revision_sngan as lays
# from adaBound import AdaBoundOptimizer
# from Anatomy_Recon.fullyADMM import layers_revision_sngan as lays
# import Anatomy_Recon.fullyADMM.layers_revision_sngan as lays

from matplotlib import pyplot as plt

from six.moves import xrange

from scipy import sparse

# CUDA_VISIBLE_DEVICES=""
# data_dir = 'D:\DNN_PiB_gan\in_out_crop'
# inames= input.get_Data_rnd(data_dir)
#
# learnin_rate = 0.0002
#
# epoch = 100

FLAGS = tf.flags.FLAGS

decay_steps = 120000000

def proj_mumap(umap, sysMat):
    imdim = umap.get_shape().as_list()
    umap_reshape = tf.transpose(umap, [1, 2, 0, 3])
    umap_reshape = tf.reshape(umap_reshape, [120 * 120, FLAGS.batch_size * imdim[3]])

    umapPrjImg = tf.sparse_tensor_dense_matmul(sysMat, umap_reshape)

    umapPrjImg = tf.reshape(umapPrjImg, [168, 120, FLAGS.batch_size, imdim[3]])
    umapPrjImg = tf.transpose(umapPrjImg, [2, 0, 1, 3])

    return umapPrjImg

def EMiter(sino_reshape, img, umap, sysMat):
    imdim = img.get_shape().as_list()

    img_reshape = tf.transpose(img, [1, 2, 0, 3])
    img_reshape = tf.reshape(img_reshape, [120 * 120, FLAGS.batch_size * imdim[3]])

    umap_reshape = tf.transpose(umap, [1, 2, 0, 3])
    umap_reshape = tf.reshape(umap_reshape, [120 * 120, FLAGS.batch_size * imdim[3]])

    umapPrjImg = tf.sparse_tensor_dense_matmul(sysMat, umap_reshape)

    prjImg = tf.sparse_tensor_dense_matmul(sysMat, img_reshape)
    prjImg = tf.multiply(prjImg, tf.exp(-umapPrjImg*0.1))

    prjImg = tf.divide(sino_reshape, prjImg + 1e-8)

    imgEst = tf.multiply(
        tf.divide(tf.sparse_tensor_dense_matmul(sysMat, prjImg, adjoint_a=True),
                  tf.sparse_tensor_dense_matmul(sysMat, tf.exp(-umapPrjImg*0.1), adjoint_a=True) + 1e-8),
        img_reshape
    )

    imgEst = tf.reshape(imgEst, [120, 120, FLAGS.batch_size, imdim[3]])
    imgEst = tf.transpose(imgEst, [2, 0, 1, 3])

    imgEst = tf.where(tf.is_nan(imgEst), tf.zeros_like(imgEst), imgEst)

    return imgEst

def TRiter(sino_reshape, umap, img, sysMat):
    imdim = img.get_shape().as_list()

    img_reshape = tf.transpose(img, [1, 2, 0, 3])
    img_reshape = tf.reshape(img_reshape, [120 * 120, FLAGS.batch_size * imdim[3]])

    umap_reshape = tf.transpose(umap, [1, 2, 0, 3])
    umap_reshape = tf.reshape(umap_reshape, [120 * 120, FLAGS.batch_size * imdim[3]])

    prjImg = tf.sparse_tensor_dense_matmul(sysMat, img_reshape)
    umapPrjImg = tf.sparse_tensor_dense_matmul(sysMat, umap_reshape)
    umapPrjImg = tf.multiply(prjImg, tf.exp(-0.1*umapPrjImg))

    deno = tf.sparse_tensor_dense_matmul(sysMat, tf.multiply(umapPrjImg,  tf.sparse_tensor_dense_matmul(sysMat, tf.ones_like(img_reshape))), adjoint_a=True)
    numo = tf.sparse_tensor_dense_matmul(sysMat, tf.multiply(umapPrjImg, (1-tf.divide(sino_reshape, umapPrjImg + 1e-8))), adjoint_a=True)

    tmp = tf.divide(numo, deno+1e-8)

    tmp = tf.reshape(tmp, [120, 120, FLAGS.batch_size, imdim[3]])
    tmp = tf.transpose(tmp, [2, 0, 1, 3])
    tmp = tf.where(tf.is_nan(tmp), tf.zeros_like(tmp), tmp)

    umapEst = umap + tmp

    return umapEst


def EMiter_grad(sino, sino_norm, img, sysMat):
    imdim = img.get_shape().as_list()

    img_reshape = tf.transpose(img, [1, 2, 0, 3])
    img_reshape = tf.reshape(img_reshape, [200 * 200, FLAGS.batch_size * imdim[3]])

    sino_reshape = tf.squeeze(sino, axis=-1)
    sino_reshape = tf.transpose(sino_reshape, [1, 2, 0])
    sino_reshape = tf.reshape(sino_reshape, [168 * 200, FLAGS.batch_size])
    sino_reshape = tf.tile(sino_reshape, [1, imdim[3]])

    prjImg = tf.sparse_tensor_dense_matmul(sysMat, img_reshape)
    prjImg = tf.divide(sino_reshape - prjImg, prjImg + 1e-9)

    sino_norm = tf.expand_dims(sino_norm, axis=-1)
    sino_norm = tf.tile(sino_norm, [1, 1, FLAGS.batch_size])
    sino_norm = tf.reshape(sino_norm, [200 * 200, FLAGS.batch_size])
    sino_norm = tf.tile(sino_norm, [1, imdim[3]])

    grad = tf.divide(tf.sparse_tensor_dense_matmul(sysMat, prjImg, adjoint_a=True), sino_norm + 1e-9)
    # grad = np.multiply(img_reshape, grad - 1)
    grad = grad

    grad = tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad)

    grad = tf.reshape(grad, [200, 200, FLAGS.batch_size, imdim[3]])
    grad = tf.transpose(grad, [2, 0, 1, 3])

    return grad


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

OUT_ITER = 5
TRAIN_EPOCH = 1000


def train(train_dir):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Sino, Label, MLEM, MRI
        xlr1, xlr2, xlr3, xlr4, xlr5, xlr6 = lays.distorted_inputs()

        import h5py
        with h5py.File('projMat_MLAA_120_7.3.mat', 'r') as f:
            data = f['W']['data']
            ir = f['W']['ir']
            jc = f['W']['jc']
            loaded_sparse = scipy.sparse.csc_matrix((data, ir, jc), shape=(20160,14400), dtype=np.float32)
        # tmp = scipy.io.loadmat('C:\\Users\SKKang\PycharmProjects\DeepLearnings\Anatomy_Recon\\forADMM\\sysmat240.mat')
        loaded_sparse_tf = convert_sparse_matrix_to_sparse_tensor(loaded_sparse)

        do_flip2 = tf.random_uniform([]) > 0.5
        xlr1 = tf.cond(do_flip2, lambda: tf.image.flip_up_down(xlr1), lambda: xlr1)
        xlr2 = tf.cond(do_flip2, lambda: tf.image.flip_up_down(xlr2), lambda: xlr2)
        xlr3 = tf.cond(do_flip2, lambda: tf.image.flip_left_right(xlr3), lambda: xlr3)
        xlr4 = tf.cond(do_flip2, lambda: tf.image.flip_left_right(xlr4), lambda: xlr4)
        xlr5 = tf.cond(do_flip2, lambda: tf.image.flip_left_right(xlr5), lambda: xlr5)
        xlr6 = tf.cond(do_flip2, lambda: tf.image.flip_left_right(xlr6), lambda: xlr6)

        # do_flip3 = tf.random_uniform([]) > 0.5
        # xlr1 = tf.cond(do_flip3, lambda: tf.reverse(xlr1, axis=[-1]), lambda: xlr1)
        # xlr2 = tf.cond(do_flip3, lambda: tf.reverse(xlr2, axis=[-1]), lambda: xlr2)
        # xlr3 = tf.cond(do_flip3, lambda: tf.reverse(xlr3, axis=[-1]), lambda: xlr3)
        # xlr4 = tf.cond(do_flip3, lambda: tf.reverse(xlr4, axis=[-1]), lambda: xlr4)
        # xlr5 = tf.cond(do_flip3, lambda: tf.reverse(xlr5, axis=[-1]), lambda: xlr5)
        # xlr6 = tf.cond(do_flip3, lambda: tf.reverse(xlr6, axis=[-1]), lambda: xlr6)

        sino_reshape = tf.transpose(xlr1, [1, 2, 0, 3])
        sino_reshape = tf.reshape(sino_reshape, [168 * 120, FLAGS.batch_size * 3])

        x = xlr3
        u = xlr5
        xout = []
        uout = []
        upout = []
        for iter in range(0, OUT_ITER):
            # EM update
            x = EMiter(sino_reshape, tf.nn.relu(x), tf.nn.relu(u), loaded_sparse_tf)
            x = lays.unfold_resnet_xup(x, u, layer_name='gen_x_up' + str(iter), scale=1, keep_prob=0.9)
            u = TRiter(sino_reshape, tf.nn.relu(u), tf.nn.relu(x), loaded_sparse_tf)
            u = lays.unfold_resnet_xup(u, x, layer_name='gen_u_up' + str(iter), scale=1, keep_prob=0.9)

            sino_att = tf.multiply(proj_mumap(tf.nn.relu(x), loaded_sparse_tf), tf.exp(-0.1*proj_mumap(tf.nn.relu(u), loaded_sparse_tf)))

            xout.append(x)
            uout.append(u)
            upout.append(sino_att)
        #
            if iter % 2 == 1:
                lays._activation_image_2d(tf.nn.relu(u), 0, name='ITER: ' + str(iter) + 'mu_Output')
                lays._activation_image_2d(tf.nn.relu(x), 0, name='ITER: ' + str(iter) + 'im_Output')
                # lays._activation_image_2d(z, 0,
                #                           name='ITER: ' + str(iter) + 'z_Output')
        #
        lays._activation_image_2d(tf.nn.relu(u), 0, name='ITER: ' + 'final_' + '_u_Output')
        lays._activation_image_2d(tf.nn.relu(x), 0, name='ITER: ' + 'final_' + '_x_Output')

        # loss1 = tf.reduce_mean(tf.losses.absolute_difference(x, xlr4))
        # loss2 = tf.reduce_mean(tf.losses.absolute_difference(u, xlr6))

        loss1=0
        loss2=0
        loss_tv_u=0
        for xs in xout:
            loss1 = loss1 + tf.reduce_mean(tf.losses.absolute_difference(xs, xlr4))

        for us in uout:
            loss2 = loss2 + tf.reduce_mean(tf.losses.absolute_difference(us, xlr6))
            loss_tv_u = loss_tv_u + 1e-6 * tf.reduce_mean(lays.total_variation(us))
        # loss2 = loss2/OUT_ITER

        tf.summary.scalar('TV z slice loss', loss_tv_u)
        tf.summary.scalar('x loss', loss1)
        tf.summary.scalar('u loss', loss2)

        loss2_p = 0
        for up in upout:
            # loss2_p = loss2_p + tf.reduce_mean(tf.losses.absolute_difference(up, xlr2))/200
            loss2_p = loss2_p + tf.reduce_mean(up - tf.multiply(xlr1, tf.log(up + 1e-6)))
        # loss2_p = loss2_p/OUT_ITER

        tf.summary.scalar('u proj loss', loss2_p)

        loss2 = loss2 + 0.5*loss2_p + loss_tv_u


        full_loss = loss1 + loss2

        # tf.summary.scalar('prj loss', loss2_p)
        #
        # tf.summary.scalar('L1_loss', lossz)
        # tf.summary.scalar('ssim_loss', lossz_ssim)
        # tf.summary.scalar('MR_loss', lossmr)
        #
        #
        lr = tf.train.exponential_decay(0.00005,
                                        global_step,
                                        200000,
                                        0.5,
                                        staircase=True)
        # lr = 1e-5
        tf.summary.scalar('learning_rate', lr)
        # #
        #
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'gen_x_up' in var.name]
        u_vars = [var for var in t_vars if 'gen_u_up' in var.name]

        # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in t_vars
        #                    if 'bias' not in v.name]) * 0.001
        # tf.summary.scalar('L2 decay loss', lossL2)
        # full_loss += lossL2 * 0.001

        # optm = AdaBoundOptimizer(learning_rate=lr, final_lr=0.1, beta1=0.9, beta2=0.999, amsbound=True)

        # ## accummulation
        # accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in t_vars]
        # zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        optm = tf.train.AdamOptimizer(lr, epsilon=1e-4)
        optm2 = tf.train.AdamOptimizer(lr, epsilon=1e-4)
        # optm_mr = tf.train.AdamOptimizer(lr)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            # g_grads = optm.compute_gradients(full_loss, var_list=t_vars)
            g_grads = optm.compute_gradients(loss1, var_list=g_vars)
            u_grads = optm2.compute_gradients(loss2, var_list=u_vars)
            # g_grads_mr = optm_mr.compute_gradients(lossmr, var_list=t_vars)

        # accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(g_grads)]

        # train_step = optm.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(g_grads)])
        update_G = optm.apply_gradients(g_grads, global_step=global_step)
        update_u = optm2.apply_gradients(u_grads)

        saver_sa = tf.train.Saver()
        saver_re = tf.train.Saver()
        #
        sum_writer = tf.summary.FileWriter(os.path.join(train_dir, 'summs'), graph=tf.get_default_graph())
        summaries = tf.summary.merge_all()
        #
        ckpt_path = r'D:\SKK_DL\Unfold_MLAA\TrainResults\12_09_03_58test'
        ckpt = tf.train.get_checkpoint_state(ckpt_path)


        # total number of parameters: 10796656
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            saver_re.restore(sess, ckpt.model_checkpoint_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(0, 8000600):
                _, lossr, summs = sess.run(
                    [update_G, full_loss, summaries])
                for ii in range(0,2):
                    _ = sess.run(update_u)

                if i % 10 == 0:
                    if i % 200 == 0 and i != 0:
                        sum_writer.add_summary(summs, global_step=i)
                    if i % 10000 == 0 and i != 0:
                        saver_sa.save(sess, os.path.join(train_dir, 'model'), global_step=i)
                    # sum_writer.add_summary(summ, global_step=i)
                    print(
                        "{}th epoch imloss {}".format(
                            i, lossr))
                    # print("{}th iter {}".format(i, imll))
                    f = open(os.path.join(train_dir, 'losses.txt'), 'a')
                    f.write(
                        "{}th epoch  imloss {}\n".format(
                            i, lossr))
                    # f.write("{}th iter {}".format(i, imll))
                    f.close()

            coord.request_stop()
            coord.join(threads)

def main(argv=None):  # pylint: disable=unused-argument
    fname = time.strftime("%m_%d_%H_%M")
    fname = fname + '_unfold'
    tfi = os.path.join(r'D:\SKK_DL\Unfold_MLAA\TrainResults', fname)
    if tf.gfile.Exists(tfi):
        tf.gfile.DeleteRecursively(tfi)
    tf.gfile.MakeDirs(tfi)

    # Copy current source code
    source_dir = os.path.join(tfi, 'Source_code')
    tf.gfile.MakeDirs(source_dir)

    source_split = os.path.split(os.path.realpath(__file__))
    source_list = os.listdir(os.path.join(source_split[0]))

    for files in source_list:
        if files.endswith('.py'):
            shutil.copyfile(os.path.join(source_split[0], files), os.path.join(source_dir, files))

    train(tfi)

if __name__ == '__main__':
    tf.app.run()
