#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Author:Shansong Huang
# Date:2017-11-22
# The main program with ResNet50 framework
import tensorflow as tf
# from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
import keras.backend.tensorflow_backend as KTF
# from keras.callbacks import ModelCheckpoint
import pandas
import fun

config = tf.ConfigProto()
#不全部占满显存, 按需分配
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

# model_path = '/home/sshuang/result/2018_01_18_weights_epoch10_ft.h5'
label_path = "/home/sshuang/change_batch/label"
# 训练的batch_size: batch size × iter size = 80(CVPR2017)   20
batch_size = 20
epochs = 1
num_classes = 14

# 生成训练和测试数据
# return X_train——0, X_validation——1, X_test——2
#        Y_train——3, Y_validation——4, Y_test——5
image_and_label = fun.get_image_and_label(label_path)


train_generator = fun.generate_from_source_test(
    image_and_label[0], image_and_label[3], batch_size)
validation_generator = fun.generate_from_source_test(
    image_and_label[1], image_and_label[4], batch_size)
# test_generator = fun.generate_from_source_test(
#     image_and_label[2], image_and_label[5], batch_size)

image_numbers_train = image_and_label[0].shape[0]
image_numbers_validation = image_and_label[1].shape[0]
image_numbers_test = image_and_label[2].shape[0]

# 使用ResNet的结构，不包括最后一层，且加载ImageNet的预训练参数 imagenet
with tf.device('/cpu:0'):
    base_model = ResNet50(
        weights=None, include_top=False, input_shape=(512, 512, 1),
        pooling=None)
    model = fun.add_new_last_layer(base_model, num_classes)
# model.load_weights(model_path)
fun.setup_to_transfer_learn(model, base_model)


# recording loss history by Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
loss_history = LossHistory()
# filepath = "/home/sshuang/result/2018_01_17_weights_epoch1_ft.h5"
tb = TensorBoard(
    log_dir='./logs1', histogram_freq=0, write_graph=True, write_images=True)
# checkpoint = ModelCheckpoint(
#     filepath, monitor='val_loss', verbose=1, save_best_only=True,
#     save_weights_only=True, mode='min')
callbacks_list = [loss_history, tb]

history = model.fit_generator(
    train_generator, steps_per_epoch=image_numbers_train//batch_size,
    epochs=epochs, validation_data=validation_generator,
    validation_steps=image_numbers_validation//batch_size,
    callbacks=callbacks_list)

pandas.DataFrame(
    loss_history.losses).to_csv(
    '/home/sshuang/result/train_loss30.csv', header=None, index=None)

#plot_model(
#    model, show_shapes=True, to_file='/home/sshuang/result/model.png')

print('++++++++++++++++++++++++++++++++++++++++++++')
print('Number of train shape:', image_and_label[0].shape)
print('Number of validation shape:', image_and_label[1].shape)
print('Number of test shape:', image_and_label[2].shape)
print('++++++++++++++++++++++++++++++++++++++++++++')

# 保存训练得到的模型
model.save_weights('/home/sshuang/result/2018_01_18_weights_epoch_test_ft.h5')
