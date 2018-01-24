#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Author:Shansong Huang
# Date:2017-11-22
# The main program with DenseNet169 framework
import tensorflow as tf
from keras.callbacks import Callback
import keras.backend.tensorflow_backend as KTF
import densenet169
import pandas
import fun_dense

config = tf.ConfigProto()
#不全部占满显存, 按需分配
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

# model_path = '/home/sshuang/result/2017_12_19_weights_epoch25_aug.h5'
label_path = "/home/sshuang/change_batch/label"
# 训练的batch_size: batch size × iter size = 80(CVPR2017)   20
batch_size = 20
epochs = 35
nb_dense_block = 4
num_classes = 14

# 生成训练和测试数据
# return X_train——0, X_validation——1, X_test——2
#        Y_train——3, Y_validation——4, Y_test——5
image_and_label = fun_dense.get_image_and_label(label_path)


train_generator = fun_dense.generate_from_source(
    image_and_label[0], image_and_label[3], batch_size)
validation_generator = fun_dense.generate_from_source(
    image_and_label[1], image_and_label[4], batch_size)
test_generator = fun_dense.generate_from_source_test(
    image_and_label[2], image_and_label[5], batch_size)

image_numbers_train = image_and_label[0].shape[0]
image_numbers_validation = image_and_label[1].shape[0]
image_numbers_test = image_and_label[2].shape[0]

with tf.device('/cpu:0'):
    model = densenet169.DenseNet(
        classes=num_classes)
model.compile(optimizer='Adam', loss='binary_crossentropy')
# model.load_weights(model_path)


# recording loss history by Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
loss_history = LossHistory()

history = model.fit_generator(
    train_generator, steps_per_epoch=image_numbers_train//batch_size,
    epochs=epochs, validation_data=validation_generator,
    validation_steps=image_numbers_validation//batch_size,
    callbacks=[loss_history])

pandas.DataFrame(
    loss_history.losses).to_csv(
    '/home/sshuang/result/train_loss.csv', header=None, index=None)

# plot_model(
# model, show_shapes=True, to_file='/home/sshuang/result/densenet_model.png')

print('++++++++++++++++++++++++++++++++++++++++++++')
print('Number of train shape:', image_and_label[0].shape)
print('Number of validation shape:', image_and_label[1].shape)
print('Number of test shape:', image_and_label[2].shape)
print('++++++++++++++++++++++++++++++++++++++++++++')

# 保存训练得到的模型权值
model.save_weights('/home/sshuang/result/2018_01_06_weights_epoch35_des.h5')
