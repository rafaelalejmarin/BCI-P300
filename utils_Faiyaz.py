# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def acc(prediction, actual, tolerance):
    """ True Positve and False Positive calculation"""
    N = len(actual)
    actual_positives = len(np.where(actual == 1)[0])
    actual_negatives = N - actual_positives
    
    actual_pos_inds = np.where(actual == 1)[0]
    
    #print('\n\n sum of prediction \n\n',sum(prediction))
    
    predictedPositiveIndeces = np.where(prediction == 1)[0]
    print('\n\nactual_positives = ',actual_positives)
    print('\n\nnum_pred_pos_inds = ', len(predictedPositiveIndeces))


    truePositive = 0
    falsePositive = 0
    for i in range(len(actual_pos_inds)):
        lower = actual_pos_inds[i] - tolerance
        higher = actual_pos_inds[i] + tolerance
        if lower < 0:
            lower = 0
        if higher > N - 1:
            higher = N - 1
        if sum(prediction[lower:higher]) > 0:
            truePositive += 1
        else:
            falsePositive += 1
    print ('\n\ntruePositives = ', truePositive)
    print ('Size Actual = ', N)
    
#    predictedNegativeIndeces = np.where(prediction != 1)[0]
    
    falseNegative = actual_positives - truePositive
    trueNegative = actual_negatives - falsePositive


#    for i in enumerate(predictedNegativeIndeces):
#        lower = i[0]-tolerance
#        higher = i[0]+tolerance
#        if lower < 0:
#            lower = 0
#        if higher > N - 1:
#            higher = N - 1
#        if sum(prediction[lower:higher]) == 0:
#            trueNegative += 1
#        else:
#            falseNegative += 1
            
    precision = truePositive/(truePositive + falsePositive)
    usefulness = np.size(predictedPositiveIndeces)
    sensitivity = truePositive/actual_positives #Positive Accuracy
    specificity = trueNegative/actual_negatives #Negative Accuracy
    
    #return [precision, truePositive, np.size(predictedPositiveIndeces)]
    return [precision, usefulness, sensitivity, specificity, truePositive, falsePositive, trueNegative, falseNegative, actual_positives, predictedPositiveIndeces]

#a1 = [1, 0, 0, 1, 0, 0]
#a2 = [0, 0, 0, 1, 0, 0]
#result = acc(a1, a2, 3)
