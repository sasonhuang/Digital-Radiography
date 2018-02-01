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
final_num = y_pred.shape[0]
num = y_pred.shape[1]
auc = []
roc_fpr = []
roc_tpr = []
roc_thresholds = []
for i in range(num):
    auc.append(roc_auc_score(y_true[:final_num, i], y_pred[:final_num, i]))
    fpr, tpr, thresholds = roc_curve(
        y_true[:final_num, i], y_pred[:final_num, i])
    roc_fpr.append(fpr)
    roc_tpr.append(tpr)
    roc_thresholds.append(thresholds)
# roc_fpr: 1-Specificity
# roc_tpr: Sensitivity
# youden_index = Sensitivity + Specificity -1
#              = roc_tpr - roc_fpr
cut_poit = {}
all_cut_point = {}
disease_name = (
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia')
dict_thresholds_x = {}
dict_thresholds_y = {}
all_dict_thresholds_x = {}
all_dict_thresholds_y = {}
for n in range(14):
    for i in range(len(roc_fpr[n])):
        youden_index = roc_tpr[n][i] - roc_fpr[n][i]
        cut_poit[youden_index] = roc_thresholds[n][i]
        dict_thresholds_x[youden_index] = roc_fpr[n][i]
        dict_thresholds_y[youden_index] = roc_tpr[n][i]
    Max = max(cut_poit.keys())
    all_cut_point[disease_name[n]] = cut_poit[Max]
    all_dict_thresholds_x[disease_name[n]] = dict_thresholds_x[Max]
    all_dict_thresholds_y[disease_name[n]] = dict_thresholds_y[Max]
    cut_poit = {}
    dict_thresholds_x = {}
    dict_thresholds_y = {}
# print('Value of Youden index:', Max)
print('Num of test:', final_num)
print('10520 : with hernia*3 and without normal')
print('22600 : with hernia*3 and with normal')
print('10340 : with hernia and without normal')
print('22420 : with hernia and with normal')

print('Value of AUC:')
for o, a in zip(disease_name, auc):
    print(o, a)
print('--------------------')
print('Value of all cut point:')
for key, value in sorted(all_cut_point.items()):
    print(key, value)
pandas.DataFrame(auc).to_csv(
    'AUC.csv', header=None, index=None)
outfile = open('dict.txt', 'w')
for key, value in sorted(all_cut_point.items()):
    outfile.write(str(key) + ':' + str(value) + '\n')
plt.rcParams['savefig.dpi'] = 600

plt.figure(0)
plt.plot([0, 1], [0, 1], 'k--')
for i in range(num):
    plt.plot(roc_fpr[i], roc_tpr[i], label=disease_name[i])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('roc.png')
plt.show()

for i in range(num):
    plt.figure(num+1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(roc_fpr[i], roc_tpr[i])
    label = str('True Positive Rate: ' + str(
        all_dict_thresholds_y[disease_name[i]]) + '\n' +
        'False Positice Rate: ' + str(all_dict_thresholds_x[disease_name[i]])
        + '\n' + 'Cut point: ' + str(all_cut_point[disease_name[i]]))
    plt.scatter(
        all_dict_thresholds_x[disease_name[i]],
        all_dict_thresholds_y[disease_name[i]],
        c='r', marker='o', label=label)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(disease_name[i])
    plt.legend(loc='best')
    plt.savefig(str(i+1) + '.png')
    plt.clf()
