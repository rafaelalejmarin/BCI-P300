# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy

def acc(prediction, actual, tolerance):
    """ True Positve and False Positive calculation"""
    N = len(actual)
    positives = len(numpy.where(prediction == 1))
    negatives = N - positives
    
    predictedPositiveIndeces = numpy.where(prediction)
    truePositive = 0
    falsePositive = 0
    for i in predictedPositiveIndeces:
        lower = i[0]-tolerance
        higher = i[0]+tolerance
        if lower < 0:
            lower = 0
        if higher > N - 1:
            higher = N - 1
        if sum(actual[lower:higher]) > 0 :
            truePositive += 1
        else:
            falsePositive += 1
            
    predictedNegativeIndeces = numpy.where(prediction != 1)
    trueNegative = 0
    falseNegative = 0
    for i in predictedNegativeIndeces:
        lower = i[0]-tolerance
        higher = i[0]+tolerance
        if lower < 0:
            lower = 0
        if higher > N - 1:
            higher = N - 1
        if sum(actual[lower:higher]) == 0:
            trueNegative += 1
        else:
            falseNegative += 1
            
    precision = truePositive/len(predictedPositiveIndeces)
    usefulness = len(predictedPositiveIndeces)
    sensitivity = truePositive/positives #Positive Accuracy
    specificity = trueNegative/negatives #Negative Accuracy
    
    
    return [precision, usefulness, sensitivity, specificity, truePositive, falsePositive, trueNegative, falseNegative]

a1 = [0,0,0,1,0,0,0]
a2 = [0,0,0,1,1,0,0]
temp = acc(a1,a2)