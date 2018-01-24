#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Author:Shansong Huang
# Date:2017-11-22
# Predict the test dataset with ResNet50 framework
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
import keras.backend.tensorflow_backend as KTF
import pandas
import fun

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
model_path = '/home/sshuang/result/2017_01_17_weights_epoch1_ft.h5'

image_and_label = fun.get_image_and_label(label_path)

test_generator = fun.generate_from_source_test(
    image_and_label[2], image_and_label[5], batch_size)

image_numbers_test = image_and_label[2].shape[0]
with tf.device('/cpu:0'):
    base_model = ResNet50(
        weights=None, include_top=False, input_shape=(512, 512, 3),
        pooling=None)
    model = fun.add_new_last_layer(base_model, num_classes, FC_SIZE)
model.load_weights(model_path, by_name=True, skip_mismatch=True)
fun.setup_to_transfer_learn(model, base_model)

print('Number of test :', image_numbers_test)
predict_test = model.predict_generator(
    test_generator, steps=image_numbers_test//batch_size)
pandas.DataFrame(
    predict_test).to_csv(
    '/home/sshuang/result/y_pred_tf.csv', header=None, index=None)
pandas.DataFrame(
    image_and_label[5]).to_csv(
    '/home/sshuang/result/y_true_tf.csv', header=None, index=None)
