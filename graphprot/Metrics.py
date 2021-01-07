import sys
import os
import torch
import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def get_boolean(values, threshold, target):

    if target == 'fnat' or target == 'bin':
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

        # regression metrics
        self.explained_variance = None
        self.max_error =  None
        self.mean_abolute_error = None
        self.mean_squared_error = None
        self.root_mean_squared_error = None
        self.mean_squared_log_error = None
        self.median_squared_log_error = None
        self.r2_score = None
        
        if target in ['fnat', 'irmsd', 'lrmsd']:

            # Explained variance regression score function
            self.explained_variance = metrics.explained_variance_score(self.y, self.prediction)
            
            # Max_error metric calculates the maximum residual error
            self.max_error = metrics.max_error(self.y, self.prediction)

            # Mean absolute error regression loss
            self.mean_absolute_error = metrics.mean_absolute_error(self.y, self.prediction)
            
            # Mean squared error regression loss
            self.mean_squared_error = metrics.mean_squared_error(self.y, self.prediction, squared = True)

            # Root mean squared error regression loss
            self.root_mean_squared_error = metrics.mean_squared_error(self.y, self.prediction, squared = False)

            try:
                # Mean squared logarithmic error regression loss
                self.mean_squared_log_error = metrics.mean_squared_log_error(self.y, self.prediction)
            raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                            "targets contain negative values.")  
            
            # Median absolute error regression loss
            self.median_squared_log_error = metrics.median_absolute_error(self.y, self.prediction)

            # R^2 (coefficient of determination) regression score function
            self.r2_score = metrics.r2_score(self.y, self.prediction)
            
            
    def format_score(self):
        
        '''
        output : 
        - idx: value rank 
        - ground_truth_bool: binary y values
        '''

        idx = np.argsort(self.prediction)

        if self.target == 'fnat' or self.target == 'bin':
            idx = idx[::-1]
                
        ground_truth_bool = get_boolean(
            self.y, self.threshold, self.target)
        ground_truth_bool = np.array(ground_truth_bool)

        return idx, ground_truth_bool

    def hitrate(self):

        idx, ground_truth_bool = self.format_score()
        
        return np.cumsum(ground_truth_bool[idx])

    def auc(self):
        
        idx, ground_truth_bool = self.format_score()
        return roc_auc_score(ground_truth_bool, idx)

