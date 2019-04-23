# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
 """ Do training Before """
def getPrcesion(testingInput, threshold, tolerance):
    [specificity,precision] = alhadFunction(testingInput, threshold, tolerance)
    while(0.45 <= specificity <= 0.55)
        if specificity > 0.55:
            threshold += 0.01
        if specificity < 0.45:
            threshold -+ 0.01
        [specificity,precision] = alhadFunction(threshold,tolerance)
    return [specificity, precision]