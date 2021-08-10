#!/usr/bin/env python3

import cv2
import time
import numpy as np
from itertools import groupby

# FILTER CONSTANTS
HSV_LOWER_THRESHOLD = np.array([0, 0, 115], np.uint8)
HSV_UPPER_THRESHOLD = np.array([180, 255, 255], np.uint8)
MIN_WIDTH = 20
MAX_WIDTH = 100
MIN_HEIGHT = 30
MAX_HEIGHT = 100
MIN_AREA = 350
MIN_PERIMETER = 100

# BOUNDING BOX CONSTANT
PINK = (255, 0, 255)
NOT_PINK = (0, 255, 0)
LINE_SEPARATOR_THRESHOLD = 150

def read_frame(video_stream) -> tuple:
    """
    Reads a single frame from the given video stream.
    """

    ret, frame = video_stream.read() # Read Frame from Video Stream

    # Convert to RGB and Return Frame
    if video_stream.isOpened() and ret:
        return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return (ret, None)

def find_balls(frame) -> list:

    balls = [] # List for balls

    # Convert to HSV Image for HSV Filter
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) 

    # HSV Filter
    frame_threshed = cv2.inRange(frame_hsv, HSV_LOWER_THRESHOLD, HSV_UPPER_THRESHOLD) 

    # Find Contours in Filtered Image
    contours, _ = cv2.findContours(frame_threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

    # Filter Found Contours
    for contour in contours:
        bb = cv2.boundingRect(contour) # Get Bounding Box from Contour
        (x, y, w, h) = bb # Get Position and Dimensions of Letter from Bounding Box
        area = cv2.contourArea(contour) # Find Area of the Contour
        perimeter = cv2.arcLength(contour, True) # Find Perimeter of Contour
        # Do Not Add Excessively Small/Large Contours To List of balls
        if not ((w < MIN_WIDTH or w > MAX_WIDTH) or 
                (h < MIN_HEIGHT or h > MAX_HEIGHT) or 
                (area < MIN_AREA) or (perimeter < MIN_PERIMETER)):
            balls.append(bb)

    return balls # Return Final List of balls

def bound_balls(image, balls, save_letter_image: bool) -> None:

    # BOUND AND CLASSIFY LETTER IMAGES
    for letter, i in zip(balls, range(len(balls))): # For Each Letter w/ Index
        (x, y, w, h) = letter # Extract Position and Size
        letter_img = image[y:y+h, x:x+w] # Crop Letter Out of Image

        # Draw a Bounding Box Around the Letter on the Video Frame
        cv2.rectangle(image, (x, y), (x+w, y+h), PINK, 2) 

        # SAVE CROPPED IMAGES OF LETTER IF REQUESTED
        if save_letter_image:
            write_frame(letter_img, 'raw-training/letter-{}'.format(i))
        
        # DISPLAY CLASSIFIED LETTER ON VIDEO FRAME IF LETTER DOES NOT NEED TO BE SAVED
        else:
            cv2.putText(image, letter, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, NOT_PINK, 1)
