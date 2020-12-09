
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
import scipy
import random
from scipy.ndimage import zoom
import tensorflow as tf
import layers_revision_sngan as lays
from matplotlib import pyplot
import pylab

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

# SINO_DIR = 'F:\DNN_anatomy_recon\Train_brainSimul_admmnet\sin\\testset'
# REF_DIR = 'F:\DNN_anatomy_recon\\bowsher_results\ADMMNET_prepare\TrainSet'
# ML_DIR = 'F:\DNN_anatomy_recon\\bowsher_results\ADMMNET_prepare\MLEM\\Testset'
# SINO_list = os.listdir(SINO_DIR)
# REF_list = os.listdir(REF_DIR)
# ML_list = os.listdir(ML_DIR)

# sinolist = []
# for file in os.listdir(SINO_DIR):
#     if file.endswith(".img"):
#         sinolist.append(file)
#
# reflist = []
# for file in os.listdir(REF_DIR):
#     if file.endswith(".img"):
#         reflist.append(file)
#
# mllist = []
# for file in os.listdir(ML_DIR):
#     if file.endswith(".img"):
#         mllist.append(file)


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

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

OUT_ITER = 5
TRAIN_EPOCH = 1000

def train():
    checkpoint_dir = r'D:\SKK_DL\Unfold_MLAA\TrainResults\12_09_06_12_unfold'

    testdir = r'D:\SKK_DL\Unfold_MLAA\data\test_data_1130_2'

    numofset = len(os.listdir(testdir))

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        xlr1, xlr2, xlr3, xlr4, xlr5, xlr6, ok = lays.inputs_eval(testdir)

        import h5py
        with h5py.File(r'projMat_MLAA_120_7.3.mat', 'r') as f:
            data = f['W']['data']
            ir = f['W']['ir']
            jc = f['W']['jc']
            loaded_sparse = scipy.sparse.csc_matrix((data, ir, jc), shape=(20160, 14400), dtype=np.float32)
        # tmp = scipy.io.loadmat('C:\\Users\SKKang\PycharmProjects\DeepLearnings\Anatomy_Recon\\forADMM\\sysmat240.mat')
        loaded_sparse_tf = convert_sparse_matrix_to_sparse_tensor(loaded_sparse)

        sino_reshape = tf.transpose(xlr1, [1, 2, 0, 3])
        sino_reshape = tf.reshape(sino_reshape, [168 * 120, FLAGS.batch_size * 3])

        # tmp = tf.squeeze(no_feed, axis=[0, -1])
        # sino_norm = tf.expand_dims(tmp, axis=-1)
        # sino_norm = tf.tile(sino_norm, [1, 1, FLAGS.batch_size])
        # sino_norm = tf.reshape(sino_norm, [120 * 120, FLAGS.batch_size])
        # sino_norm = tf.tile(sino_norm, [1, 3])

        # x = EMiter(sino_reshape, sino_norm, tf.nn.relu(xlr3), tf.nn.relu(xlr5), loaded_sparse_tf)
        # x = TRiter(sino_reshape, sino_norm, tf.nn.relu(xlr5), tf.nn.relu(xlr3), loaded_sparse_tf)

        x = xlr3
        u = xlr5
        for iter in range(0, OUT_ITER):
            # EM update
            x = EMiter(sino_reshape, tf.nn.relu(x), tf.nn.relu(u), loaded_sparse_tf)
            x = lays.unfold_resnet_xup(x, u, layer_name='gen_x_up' + str(iter), scale=1, keep_prob=1)
            u = TRiter(sino_reshape, tf.nn.relu(u), tf.nn.relu(x), loaded_sparse_tf)
            u = lays.unfold_resnet_xup(u, x, layer_name='gen_u_up' + str(iter), scale=1, keep_prob=1)
            # u_prj = tf.nn.relu(proj_mumap(tf.nn.relu(u), loaded_sparse_tf))


        saver_re = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver_re.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            fname = time.strftime("%m_%d_%H_%M")
            fname = fname + 'test'
            tfi = os.path.join(r'D:\SKK_DL\Unfold_MLAA\TestResults', fname)
            if tf.gfile.Exists(tfi):
                tf.gfile.DeleteRecursively(tfi)
            tf.gfile.MakeDirs(tfi)


            for i in range(0, round(numofset / FLAGS.batch_size_eval) + 1):

                [xr, outkeys]= sess.run(
                    [u, ok])

                for ii, in_path in enumerate(outkeys):
                    name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
                    gimg_tmp = xr[ii, :, :, :].flatten()
                    gimg_tmp.tofile(os.path.join(tfi, 'eval_gimg' + name))

                    # gxhr_tmp = xr[ii, :, :, :].flatten()
                    # gxhr_tmp.tofile(os.path.join(tfi, 'eval_labl' + name))

                print(i, name)

            coord.request_stop()
            coord.join(threads)






def main(argv=None):  # pylint: disable=unused-argument
    train()

if __name__ == '__main__':
    tf.app.run()
