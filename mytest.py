import tensorflow as tf
import numpy as np
import math
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join('dump', 'test_log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 10
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/bu3d_frontal_600/test_shape_names.txt'))]

MODEL_PATH = 'log/model.ckpt'
HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/bu3d_frontal_600/test_gallery_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/bu3d_frontal_600/test_probe_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def cos(vector1,vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    return dot_product / ((normA*normB)**0.5)

def evaluate(num_votes):
    probeList = list()
    pgDis = list()
    galleryList = list()
    probeNum = 37
    galleryNUm = 5
    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        pos, ftr = MODEL.get_model(pointclouds_pl,tf.constant(False))

    SAVER = tf.train.Saver()
    with tf.Session() as sess:
        SAVER.restore(sess, MODEL_PATH)
        for fn in range(len(TEST_FILES)):
            current_data, test_label = provider.loadDataFile(TEST_FILES[fn])
            current_data = current_data[:, 0:NUM_POINT, :]
            test_label = np.squeeze(test_label)

            file_size = current_data.shape[0]
            for n in range(file_size):
                feed_dict = {pointclouds_pl: current_data[n:n+1,:,:], labels_pl: test_label[n:n+1]}
                res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
                probeList.append(res1[0])
                # print res1[0]

        #########################
        for fn in range(len(TRAIN_FILES)):
            current_data, train_label = provider.loadDataFile(TRAIN_FILES[fn])
            current_data = current_data[:, 0:NUM_POINT, :]
            train_label = np.squeeze(train_label)

            file_size = current_data.shape[0]
            for n in range(file_size):
                feed_dict = {pointclouds_pl: current_data[n:n+1,:,:], labels_pl: train_label[n:n+1]}
                res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
                galleryList.append(res1[0])
        # print train_label
        ###############################
        for i in range(len(probeList)):
            probeDis = list()
            for j in range(len(galleryList)):
                dis = cos(probeList[i],galleryList[j])
                # dis = np.sqrt(np.sum(np.square(probeList[i] - galleryList[j])))
                probeDis.append(dis)
            minnum  = min(probeDis)
            predict = probeDis.index(minnum)

            pgDis.append(predict // galleryNUm)


        accuNum = 0
        classNum = 0
        for i in range(len(pgDis)):
            if i %probeNum == 0:
                classNum = classNum+1
            if pgDis[i] in range(classNum,classNum+galleryNUm):
                accuNum  = accuNum + 1
        print accuNum



if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=0)
    LOG_FOUT.close()
