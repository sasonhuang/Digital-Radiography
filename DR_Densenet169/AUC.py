#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Author:Shansong Huang
# Date:2017-11-22
# Calculate the AUC value and get ROC curve
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

y_pred_path = 'y_pred_35.csv'
y_true_path = 'y_true_35.csv'
# num_without_normal : with hernia*3 and without normal
# num_normal : with hernia*3 and with normal
# num_hernia : with hernia and without normal
# num_hernia_normal : with hernia and with normal
num_without_normal = 10520
num_noraml = 22600
num_hernia = 10340
num_hernia_normal = 22420
final_num = num_hernia


# get labels from each CSV file
# label_path: each CSV files path
# return: CSV's information
def get_label_array(label_path):
    dataframe = pandas.read_csv(label_path, header=None)
    dataset = dataframe.values
    label = np.array(dataset)
    return label
y_true = get_label_array(y_true_path)
y_pred = get_label_array(y_pred_path)
num = y_pred.shape[1]
auc = []
roc_fpr = []
roc_tpr = []
for i in range(num):
    auc.append(roc_auc_score(y_true[:final_num, i], y_pred[:final_num, i]))
    fpr, tpr, _ = roc_curve(y_true[:final_num, i], y_pred[:final_num, i])
    roc_fpr.append(fpr)
    roc_tpr.append(tpr)
# for i in range(num):
pandas.DataFrame(auc).to_csv(
    'AUC.csv', header=None, index=None)
print('Value of AUC:', auc)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(roc_fpr[0], roc_tpr[0], label='Atelectasis')
plt.plot(roc_fpr[1], roc_tpr[1], label='Cardiomegaly')
plt.plot(roc_fpr[2], roc_tpr[2], label='Effusion')
plt.plot(roc_fpr[3], roc_tpr[3], label='Infiltration')
plt.plot(roc_fpr[4], roc_tpr[4], label='Mass')
plt.plot(roc_fpr[5], roc_tpr[5], label='Nodule')
plt.plot(roc_fpr[6], roc_tpr[6], label='Pneumonia')
plt.plot(roc_fpr[7], roc_tpr[7], label='Pneumothorax')
plt.plot(roc_fpr[8], roc_tpr[8], label='Consolidation')
plt.plot(roc_fpr[9], roc_tpr[9], label='Edema')
plt.plot(roc_fpr[10], roc_tpr[10], label='Emphysema')
plt.plot(roc_fpr[11], roc_tpr[11], label='Fibrosis')
plt.plot(roc_fpr[12], roc_tpr[12], label='Pleural_Thickening')
plt.plot(roc_fpr[13], roc_tpr[13], label='Hernia')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('roc.png')
plt.show()
