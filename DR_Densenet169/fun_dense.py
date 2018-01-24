#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Author:Shansong Huang
# Date:2017-11-22
# The function of main.py
from keras.preprocessing.image import random_zoom, random_shift
from keras.preprocessing.image import random_rotation, load_img
from keras.preprocessing.image import img_to_array
from itertools import izip
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas
import os

gray = True


# get all CSV file paths
# path: all label's path
# return: each CSV files path
def get_csv_path(path):
    files_1 = os.listdir(path)
    files_1.sort()
    path_list = []
    for file_1 in files_1:
        path_list.append(path+"/"+file_1)
    path_list.sort()
    path_list = np.array(path_list)
    return path_list


# get index array from each CSV file
# label_path: each CSV files path
# return: CSV's information
def get_index_array(label_path):
    dataframe = pandas.read_csv(label_path, header=None)
    dataset = dataframe.values
    label = np.array(dataset)
    return label


# split dataset to train validation and test by 7:1:2
def random_split(label):
    train, test = train_test_split(
        label, test_size=0.2, random_state=1)
    train, validation = train_test_split(
        train, test_size=0.125, random_state=1)
    return train, validation, test


# get image and label paths
# label_path: all label's path
# return: each sample's path and 14 dim labels
def get_image_and_label(label_path):
    # Get every csv file path
    label_files_path_list = get_csv_path(label_path)
    train = []
    validation = []
    test = []
    for file in label_files_path_list:
        index_array = get_index_array(file)
    # Randomly(seed = 0) split to train, validation and test 70%,10%,20%
        data = random_split(index_array)
        train.extend(data[0])
        validation.extend(data[1])
        test.extend(data[2])

    train = np.random.permutation(train)
    validation = np.random.permutation(validation)
    test = np.random.permutation(test)
    X_train = train[:, :1]
    X_validation = validation[:, :1]
    X_test = test[:, :1]
    Y_train = train[:, 1:]
    Y_validation = validation[:, 1:]
    Y_test = test[:, 1:]
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


# Data Augmentation:
# Randomly translated in 4 directions by 25 pixels
# Randomly rotated from -15 to 15 degrees
# Randomly scaled between 80% and 120%
def distorted_image(image):
    image1 = random_rotation(
        image, 15, row_axis=0, col_axis=1, channel_axis=2)
    image2 = random_zoom(
        image1, (0.8, 1.2), row_axis=0, col_axis=1, channel_axis=2)
    image3 = random_shift(
        image2, 0.05, 0.05, row_axis=0, col_axis=1, channel_axis=2)
    return image3


# change array to list
def array_to_list(target_array):
    target_list = []
    for i in target_array:
        for a in i:
            target_list.append(a)
    return target_list


# Non-stop generate sample's path and labels
# until one epoch end for train and validation
def generate_from_source(image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    y = []
    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    while True:
        for image_path, label_path in izip(image_path_list, label_path_list):
            image_array = img_to_array(
                load_img(image_path, grayscale=gray, target_size=(512, 512)))
            label = label_path
            x.append(distorted_image(image_array))
            y.append(label)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x), np.array(y))
                x = []
                y = []


# non-stop generate sample's path and labels
# until one epoch end for test(without data-augmentation)
def generate_from_source_test(image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    y = []
    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    while True:
        for image_path, label_path in izip(image_path_list, label_path_list):
            image_array = img_to_array(
                load_img(image_path, grayscale=gray, target_size=(512, 512)))
            label = label_path
            x.append(image_array)
            y.append(label)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x), np.array(y))
                x = []
                y = []
