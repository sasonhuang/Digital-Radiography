#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Author:Shansong Huang
# Date:2017-11-22
# Predict the test dataset with DenseNet169 framework
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import densenet169
import pandas
import fun_dense

batch_size = 20
gpu_num = 4
FC_SIZE = 2048
num_classes = 14

config = tf.ConfigProto()
#不全部占满显存, 按需分配
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

label_path = "/home/sshuang/change_batch/label_test_normal"
model_path = '/home/sshuang/result/2018_01_06_weights_epoch35_des.h5'

image_and_label = fun_dense.get_image_and_label(label_path)

test_generator = fun_dense.generate_from_source_test(
    image_and_label[2], image_and_label[5], batch_size)

image_numbers_test = image_and_label[2].shape[0]
with tf.device('/cpu:0'):
    model = densenet169.DenseNet(
        classes=num_classes)
model.load_weights(model_path)
model.compile(optimizer='Adam', loss='binary_crossentropy')

print('Number of test :', image_numbers_test)
predict_test = model.predict_generator(
    test_generator, steps=image_numbers_test//batch_size)
pandas.DataFrame(
    predict_test).to_csv(
    '/home/sshuang/result/y_pred_35.csv', header=None, index=None)
pandas.DataFrame(
    image_and_label[5]).to_csv(
    '/home/sshuang/result/y_true_35.csv', header=None, index=None)
