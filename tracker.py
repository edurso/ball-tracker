#!/usr/bin/env python3

# packages
import cv2
import numpy as np

# constants
CAMERA = 1
HSV_LOWER_THRESHOLD = np.array([0, 0, 115], np.uint8)
HSV_UPPER_THRESHOLD = np.array([180, 255, 255], np.uint8)
MIN_WIDTH = 20
MAX_WIDTH = 100
MIN_HEIGHT = 30
MAX_HEIGHT = 100
MIN_AREA = 350
MIN_PERIMETER = 100
PINK = (255, 0, 255)
NOT_PINK = (0, 255, 0)
LINE_SEPARATOR_THRESHOLD = 150

# set up camera
cam = cv2.VideoCapture(CAMERA)
stream = True

# list of (x, y) positions of scored balls
scored_balls = []

# main loop
while stream:

    # list of (x, y, w, h) positions of visible balls
    visible_balls = []

    # read frame
    ret, frame = cam.read()

    # quit stream if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        stream = False

    # find balls
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) 
    frame_threshed = cv2.inRange(frame_hsv, HSV_LOWER_THRESHOLD, HSV_UPPER_THRESHOLD) 
    contours, _ = cv2.findContours(frame_threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    for contour in contours:
        bb = cv2.boundingRect(contour) 
        (x, y, w, h) = bb 
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True) 
        if not ((w < MIN_WIDTH or w > MAX_WIDTH) or 
                (h < MIN_HEIGHT or h > MAX_HEIGHT) or 
                (area < MIN_AREA) or (perimeter < MIN_PERIMETER)):
            visible_balls.append(bb)
            cv2.rectangle(frame, (x, y), (x+w, y+h), PINK, 2) 

    # TODO determine if a ball is scored by watching it get smaller and then bigger again (bounces off wall)

    # TODO place a red dot on the screen by scored ball locations

    # TODO find some way to figure out the distance from the center of a ball

    # display frame
    cv2.imshow('Scoring', frame)

# TODO go over scored balls and calculate score
