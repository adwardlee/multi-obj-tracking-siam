from __future__ import division
import argparse
import logging
import numpy as np
import cv2

def SelectRoi(region, object, frame, number = 1):
    object_box = []
    region_box = []
    if region != None:
        for idx in range(number):
            temp = region[idx].split(',')
            x, y, w, h = map(int, temp)
            region_box.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    else:
        for idx in range(number):
            try:
                init_rect = cv2.selectROI('SiamRPN_Tracking', frame, False, False)
                x, y, w, h = init_rect
                region_box.append([[x, y],[x + w, y],[x + w, y + h],[x, y + h]])
            except:
                exit()
    if object != None:
        for idx in range(number):
            temp = region[idx].split(',')
            x, y, w, h = map(int, temp)
            object_box.append([x,y,w,h])
    else:
        for idx in range(number):
            try:
                init_rect = cv2.selectROI('SiamRPN_Tracking', frame, False, False)
                x, y, w, h = init_rect
                object_box.append([x,y,w,h])
                print('initial position {}, {}, {}, {}'.format(x, y, w, h))
            except:
                exit()
    region_box = np.array(region_box)
    object_box = np.array(object_box)
    return region_box, object_box
