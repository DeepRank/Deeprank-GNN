import sys
import os
import torch
import numpy as np

from sklearn.metrics import confusion_matrix

def get_boolean(vals, threshold, target):
        
        if target == 'fnat' or target == 'bin':
                vals_bool = [1 if x >= threshold else 0 for x in vals]
        else: 
                vals_bool = [1 if x < threshold else 0 for x in vals]
        
        return vals_bool

def get_comparison(prediction, ground_truth):
        
    CM = confusion_matrix(ground_truth, prediction)
    
    FP = CM.sum(axis=0) - np.diag(CM)
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    	
    return FP, FN, TP, TN
        
class Metrics(object):
        
        def __init__(self, y_pred, y_hat, task, target, threshold = 4):
                '''Master class from which all the other metrics are computed
                Arguments
                y_pred : predicted values
                y_hat : ground truth
                task : classification ('class') or regression ('reg')
                target : irmsd, fnat, class, bin
                threshold 
                '''
                
                self.task = task
                self.target = target
                self.y_pred = y_pred
                self.y_hat = y_hat
                
                self.threshold = threshold
                print ('Threshold set to {}'.format(self.threshold))
                
                if self.task == 'reg':
                        
                        self.y_pred = get_boolean(self.y_pred, self.threshold, self.target)
                        self.y_hat = get_boolean(self.y_hat, self.threshold, self.target)
                        
        def get_metrics(self):            
                
                FP, FN, TP, TN = get_comparison(self.y_pred, self.y_hat)
                
                # Sensitivity, hit rate, recall, or true positive rate
                self.TPR = TP/(TP+FN)
                # Specificity or true negative rate
                self.TNR = TN/(TN+FP) 
                # Precision or positive predictive value
                self.PPV = TP/(TP+FP)
                # Negative predictive value
                self.NPV = TN/(TN+FN)
                # Fall out or false positive rate
                self.FPR = FP/(FP+TN)
                # False negative rate
                self.FNR = FN/(TP+FN)
                # False discovery rate
                self.FDR = FP/(TP+FP)
                # Overall accuracy
                self.ACC = (TP+TN)/(TP+FP+FN+TN)
                
        def HitRate(self):
                
                idx = np.argsort(self.y_pred)
                
                if self.target == 'fnat' or self.target == 'bin':
                        idx = idx[::-1]
                        
                ground_truth_bool = get_boolean(self.y_hat, self.threshold, self.target)
                ground_truth_bool = np.array(ground_truth_bool)
                
                hitrate = np.cumsum(ground_truth_bool[idx])
                
                return hitrate
