import sys
import os
import torch
import numpy as np

from sklearn.metrics import confusion_matrix


def get_boolean(values, threshold, target):

    inverse = ['fnat', 'bin'] 
    if target in inverse:
        values_bool = [1 if x > threshold else 0 for x in values]
    else:
        values_bool = [1 if x < threshold else 0 for x in values]

    return values_bool


def get_comparison(prediction, ground_truth, binary=True, classes=[0, 1]):

    CM = confusion_matrix(ground_truth, prediction, labels=classes)

    false_positive = CM.sum(axis=0) - np.diag(CM)
    false_negative = CM.sum(axis=1) - np.diag(CM)
    true_positive = np.diag(CM)
    true_negative = CM.sum() - (false_positive + false_negative + true_positive)

    if binary == True:
        return false_positive[1], false_negative[1], true_positive[1], true_negative[1]

    else:
        return false_positive, false_negative, true_positive, true_negative


class Metrics(object):

    def __init__(self, prediction, y, target, threshold=4, binary=True):
        '''Master class from which all the other metrics are computed
        Arguments
        prediction:  predicted values
        y:           ground truth
        target:      irmsd, fnat, class, bin
        threshold:   threshold used to split the data into a binary vector
        binary:      transform the data in binary vectors
        '''

        self.prediction = prediction
        self.y = y
        self.binary = binary
        self.target = target
        self.threshold = threshold

        print('Threshold set to {}'.format(self.threshold))

        if self.binary == True:

            prediction_bool = get_boolean(
                self.prediction, self.threshold, self.target)
            y_bool = get_boolean(
                self.y, self.threshold, self.target)
            classes = [0, 1]

            false_positive, false_negative, true_positive, true_negative = get_comparison(
                prediction_bool, y_bool, self.binary, classes=classes)
            
        else:
            if self.target == 'class':
                classes = [1, 2, 3, 4, 5]
            else:
                classes = [0, 1]

            false_positive, false_negative, true_positive, true_negative = get_comparison(
                self.prediction, self.y, self.binary, classes=classes)

        try:
            # Sensitivity, hit rate, recall, or true positive rate
            self.sensitivity = true_positive/(true_positive+false_negative)
        except:
            self.sensitivity = None

        try:
            # Specificity or true negative rate
            self.specificity = true_negative/(true_negative+false_positive)
        except:
            self.specificity = None

        try:
            # Precision or positive predictive value
            self.precision = true_positive/(true_positive+false_positive)
        except:
            self.precision = None

        try:
            # Negative predictive value
            self.NPV = true_negative/(true_negative+false_negative)
        except:
            self.NPV = None

        try:
            # Fall out or false positive rate
            self.FPR = false_positive/(false_positive+true_negative)
        except:
            self.FPR = None

        try:
            # False negative rate
            self.FNR = false_negative/(true_positive+false_negative)
        except:
            self.FNR = None

        try:
            # False discovery rate
            self.FDR = false_positive/(true_positive+false_positive)
        except:
            self.FDR = None

        self.accuracy = (true_positive+true_negative)/(true_positive+false_positive+false_negative+true_negative)

    def hitrate(self):

        idx = np.argsort(self.prediction)

        inverse = ['fnat', 'bin']
        if self.target in inverse:   
            idx = idx[::-1]

        ground_truth_bool = get_boolean(
            self.y, self.threshold, self.target)
        ground_truth_bool = np.array(ground_truth_bool)

        hitrate = np.cumsum(ground_truth_bool[idx])
        return hitrate
