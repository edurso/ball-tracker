#!/usr/bin/env python3

# packages
import cv2
import math

# constants
CAMERA = 'data.mov' # use 1 for usb webcam, 0 for internal webcam, or just pass a file
MILLISECONDS_BEFORE_NEXT_FRAME = 1
MIN_AREA_THRESHOLD = 100
GREEN = (0, 255, 0)
RED = (0, 0, 255)
SIZE = (640, 480)
EUCLIDEAN_DIST_THRESHOLD = 100

# class to track objects by comparing the euclidean distances between frames
class EuclideanDistTracker:
    
    def __init__(self):
        # store the center positions of the objects
        self.center_points = {}
        # keep the count of the ids
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # objects boxes and ids
        objects_bbs_ids = []

        # get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < EUCLIDEAN_DIST_THRESHOLD:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # new object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # clean the dictionary by center points to remove ids not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # update dictionary with ids not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# set up stream 
stream = cv2.VideoCapture(CAMERA)
stream_active = True

# object detector (from stable camera)
obj_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# object tracker
obj_tracker = EuclideanDistTracker()

# placeholders for current, past, and scored balls with ids
current_balls_ids = []
past_balls_ids = []
scored_balls_ids = []

# stream loop
while stream_active:

    # read frame
    ret, frame = stream.read()

    # if frame is read correctly
    if ret:

        # frame shape
        h, w, _ = frame.shape
        #print('Height: {} | Width: {}'.format(h,w))

        # crop/resize frame
        frame = cv2.resize(frame, SIZE, interpolation = cv2.INTER_LINEAR)
        frame = frame[100:h, 0:w]

        # object detection
        masked_frame = obj_detector.apply(frame)

        # clean frame, remove shadows
        _, masked_frame = cv2.threshold(masked_frame, 254, 255, cv2.THRESH_BINARY)

        # find contours from mask
        cnts, _ = cv2.findContours(masked_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # TODO might need to add more filters for balls

        # array for visible balls
        visible_balls = []

        for cnt in cnts:

            # filter small contours
            if cv2.contourArea(cnt) > MIN_AREA_THRESHOLD:
                
                # bound detections
                x, y, w, h = cv2.boundingRect(cnt)
                
                # add detected ball to visible ball list
                visible_balls.append([x, y, w, h])

        # print visible balls
        #print(visible_balls)

        # object tracking
        current_balls_ids = obj_tracker.update(visible_balls)
        #print(current_balls_ids)

        # draw & label current balls
        for ball_id in current_balls_ids:
            x, y, w, h, id = ball_id
            cv2.putText(frame, str(id), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 3)

        # check if ball hit wall - might work ig
        if len(past_balls_ids) > 0:
            for past_balls_id in past_balls_ids:
                for current_balls_id in current_balls_ids:
                    px, py, pw, ph, pid = past_balls_id
                    cx, cy, cw, ch, cid = current_balls_id
                    
                    # consider ball scored if past area is smaller than current area
                    if pid == cid:
                        pa = pw * ph
                        ca = cw * ch
                        if pa < ca:

                            # do not add scored ball if it has already been scored
                            scored_ids_only = []
                            for scored_balls_id in scored_balls_ids:
                                sid, _, _ = scored_balls_id
                                scored_ids_only.append(sid)

                            if pid not in scored_ids_only:
                                scored_balls_ids.append((pid, px, py))
                            # for scored_balls_id in scored_balls_ids:
                            #     sid, _, _ = scored_balls_id
                            #     if not sid == pid:
                            

        # draw & label scored balls
        for scored_balls_id in scored_balls_ids:
            id, x, y = scored_balls_id
            cv2.circle(frame, (x, y), radius=0, color=RED, thickness=5)

        # show the frame
        cv2.imshow('frame', frame)

        # copy current ball ids to past ball ids for next iteration
        past_balls_ids = current_balls_ids.copy()

    # exit on 'q' press
    if cv2.waitKey(MILLISECONDS_BEFORE_NEXT_FRAME) & 0xFF == ord('q') : stream = False

# TODO calculate scores

# post stream clean up
stream.release()
cv2.destroyAllWindows()
