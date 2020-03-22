# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import cv2
import argparse
import logging
import sys
from scipy.interpolate import UnivariateSpline

class Filter(object):
    def __init__(self):
        self._processfunc = {'_brightcontrast': {'alpha':1.2, 'beta':1.2}, 
                             '_white_balance':{'ratio':1.3}, 
                             '_sharpen':{'blurratio':1.5, 'originalratio':-0.5},
							 '_saturation':{},
                             }
    def _saturation(self, img, param):
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
        [0, 75, 145, 215    , 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
        [0, 30,  80, 120, 192])
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), 
            cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(xrange(256))

    def _brightcontrast(self, img, param):
        return cv2.convertScaleAbs(img, param['alpha'], param['beta'])
    
    def _sharpen(self, img, param):
        output = cv2.GaussianBlur(img, (0, 0), 25)
        return cv2.addWeighted(img, param['blurratio'], output, param['originalratio'], 0)        
    
    def _white_balance(self, img, param):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] + ((avg_a - 128) * (result[:, :, 0] / 255.0) * param['ratio'])
        result[:, :, 2] = result[:, :, 2] + ((avg_b - 128) * (result[:, :, 0] / 255.0) * param['ratio'])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
        
    def process(self, img):
        processimg = img.copy()
        final = img.copy()
        for funcname in self._processfunc:
            param = self._processfunc[funcname]
            method_to_call = getattr(self, funcname)            
            processimg = method_to_call(processimg, param)
            final = np.hstack((final, processimg))
        if final is not None:
            cv2.imwrite('result.jpg', final)
        

    
def testprocess(imagefilename):
    img = cv2.imread(imagefilename)
    if len(img.shape) > 2 and img.shape[2] == 4:
        #convert the image from RGBA2RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    filter = Filter()
    filter.process(img)
    
if __name__ == '__main__':
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(level=LOG_LEVEL,
                format="%(levelname)s:[%(lineno)d]%(name)s:%(funcName)s->%(message)s",  #logger.BASIC_FORMAT,
                datefmt='%a, %d %b %Y %H:%M:%S')    
    if len(sys.argv) == 1:
        test_list('../input.list', '../')
    elif len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr,'debugsearch.py command'    