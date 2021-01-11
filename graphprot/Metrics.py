import sys
import os
import torch
import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def get_boolean(values, threshold, target):
    '''
    Transform continuous or multiclass values into boolean values (0/1)
    
    Arguments
    values: vector of the target values
    threshold: threshold used to assign a boolean value. 
               0 is assigned to 'bad' values
               1 is assigned to 'good' values
    target: target (y). 
               if target is fnat or bin: target value > threshold = 1 
               else:                     target value > threshold = 0 
    '''
    inverse = ['fnat', 'bin'] 
    if target in inverse:
        values_bool = [1 if x > threshold else 0 for x in values]
    else:
        values_bool = [1 if x < threshold else 0 for x in values]

    return values_bool


def get_comparison(prediction, ground_truth, binary=True, classes=[0, 1]):
    '''
    Compute the confusion matrix to compute the number of: 
    - false positive (FP)
    - false negative (FN)
    - true positive (TP)
    - true negative (TN)
    
    Arguments
    prediction: value predicted
    ground truth: target value
    binary: If binary is True, the function will return a single value for each FP/FN/TP/TN variable.
            If binary is False, the function will return a vector of n values for each FP/FN/TP/TN
            variable, n being the total number of classes
            Defaults is set to True
    '''
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
        
        Computed metrics:
        
        Classification metrics: 
        - self.sensitivity: Sensitivity, hit rate, recall, or true positive rate
        - self.specificity: Specificity or true negative rate
        - self.precision: Precision or positive predictive value
        - self.NPV: Negative predictive value
        - self.FPR: Fall out or false positive rate
        - self.FNR: False negative rate
        - self.FDR: False discovery rate
        - self.accuracy: Accuracy
        
        - self.auc(): AUC
        - self.hitrate(): Hit rate
       
        Regression metrics:
        - self.explained_variance: Explained variance regression score function
        - self.max_error: Max_error metric calculates the maximum residual error
        - self.mean_abolute_error: Mean absolute error regression loss
        - self.mean_squared_error: Mean squared error regression loss
        - self.root_mean_squared_error: Root mean squared error regression loss
        - self.mean_squared_log_error: Mean squared logarithmic error regression loss
        - self.median_squared_log_error: Median absolute error regression loss
        - self.r2_score: R^2 (coefficient of determination) regression score function

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
            except ValueError:
                print ("WARNING: Mean Squared Logarithmic Error cannot be used when "
                            "targets contain negative values.")  
            
            # Median absolute error regression loss
            self.median_squared_log_error = metrics.median_absolute_error(self.y, self.prediction)

            # R^2 (coefficient of determination) regression score function
            self.r2_score = metrics.r2_score(self.y, self.prediction)
            
            
    def format_score(self):
        '''
        Sorts the predicted values depending on the target:
        - if target is fnat or bin, the highest the better ranked
        - else: the lowest the better ranked    
        
        Returns the ranks of the predicted values and 
        the corresponding boolean (0/1) target values
        
        output : 
        - idx: ranks of the predicted values
        - ground_truth_bool: boolean y values
        '''

        idx = np.argsort(self.prediction)

        inverse = ['fnat', 'bin']
        if self.target in inverse:   
            idx = idx[::-1]
                
        ground_truth_bool = get_boolean(
            self.y, self.threshold, self.target)
        ground_truth_bool = np.array(ground_truth_bool)
        return idx, ground_truth_bool

    def hitrate(self):
        '''
        Sorts the target boolean values (0/1) according to the ranks of predicted values
        Returns the cumulative sum of hits (1) 
        '''
        
        idx, ground_truth_bool = self.format_score()
        return np.cumsum(ground_truth_bool[idx])

    def auc(self):
        '''
        Returns the Receiver Operating Characteristic (ROC) area under the curve (AUC)
        '''
        idx, ground_truth_bool = self.format_score()
        return roc_auc_score(ground_truth_bool, idx)
