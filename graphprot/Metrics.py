import sys
import os
import torch
import numpy as np

from sklearn.metrics import confusion_matrix


def get_boolean(vals, threshold, target):

    if target == 'fnat' or target == 'bin':
        vals_bool = [1 if x > threshold else 0 for x in vals]
    else:
        vals_bool = [1 if x < threshold else 0 for x in vals]

    return vals_bool


def get_comparison(prediction, ground_truth, binary=True, classes=[0, 1]):

    CM = confusion_matrix(ground_truth, prediction, labels=classes)

    FP = CM.sum(axis=0) - np.diag(CM)
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)

    if binary == True:
        return FP[1], FN[1], TP[1], TN[1]

    else:
        return FP, FN, TP, TN


class Metrics(object):

    def __init__(self, y_pred, y_hat, target, threshold=4, binary=True):
        '''Master class from which all the other metrics are computed
        Arguments
        y_pred:      predicted values
        y_hat:       ground truth
        target:      irmsd, fnat, class, bin
        threshold:   threshold used to split the data into a binary vector
        binary:      transform the data in binary vectors
        '''

        self.y_pred = y_pred
        self.y_hat = y_hat
        self.binary = binary
        self.target = target
        self.threshold = threshold

        print('Threshold set to {}'.format(self.threshold))

        if self.binary == True:

            self.y_pred = get_boolean(
                self.y_pred, self.threshold, self.target)
            self.y_hat = get_boolean(
                self.y_hat, self.threshold, self.target)
            classes = [0, 1]

        else:
            if self.target == 'class':
                classes = [1, 2, 3, 4, 5]
            else:
                classes = [0, 1]

        FP, FN, TP, TN = get_comparison(
            self.y_pred, self.y_hat, self.binary, classes=classes)

        try:
            # Sensitivity, hit rate, recall, or true positive rate
            self.TPR = TP/(TP+FN)
        except:
            self.TPR = None

        try:
            # Specificity or true negative rate
            self.TNR = TN/(TN+FP)
        except:
            self.TNR = None

        try:
            # Precision or positive predictive value
            self.PPV = TP/(TP+FP)
        except:
            self.PPV = None

        try:
            # Negative predictive value
            self.NPV = TN/(TN+FN)
        except:
            self.NPV = None

        try:
            # Fall out or false positive rate
            self.FPR = FP/(FP+TN)
        except:
            self.FPR = None

        try:
            # False negative rate
            self.FNR = FN/(TP+FN)
        except:
            self.FNR = None

        try:
            # False discovery rate
            self.FDR = FP/(TP+FP)
        except:
            self.FDR = None

        self.ACC = (TP+TN)/(TP+FP+FN+TN)

    def HitRate(self):

        idx = np.argsort(self.y_pred)

        if self.target == 'fnat' or self.target == 'bin':
            idx = idx[::-1]

        ground_truth_bool = get_boolean(
            self.y_hat, self.threshold, self.target)
        ground_truth_bool = np.array(ground_truth_bool)

        hitrate = np.cumsum(ground_truth_bool[idx])

        return hitrate
