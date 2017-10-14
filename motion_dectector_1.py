# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import itertools
import numpy as np
from sklearn import mixture
from collections import deque
from picamera.array import PiRGBArray
from picamera import PiCamera

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
pwm = GPIO.PWM(18,50)
pwm.start(5)

angle = 50
coefMotor = .02



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
timeLoopIndex = 0
n_comp = 1
historySize = 10
historyAverage = [ 1.0/(i+1) for i in range(historySize) ]

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    print('ok')
    firstFrame = None
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 16
    rawCapture = PiRGBArray(camera, size=(640,480))
    # camera = cv2.VideoCapture(0)
    time.sleep(0.25)

# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

    # initialize the first frame in the video stream
    firstFrame = None
    # loop over the frames of the video

center = None
gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type='diag')
#gmm = mixture.BayesianGaussianMixture(n_components=n_comp, covariance_type='diag',
#        weight_concentration_prior_type='dirichlet_distribution', 
#        weight_concentration_prior=0.01, warm_start=True)

while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"
    
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
    
#for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#    frame = f.array
#    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.4)
    text = "Unoccupied"
    
    # if the first frame is None, initialize it
    if firstFrame is None:
        # maskSamples = [ [i,j] for i,j in itertools.product(range(gray.shape[0]), range(gray.shape[1])) ]
        #rawCapture.truncate(0)
        maskSamples = np.array([ [(i,j) for j in range(gray.shape[1])] for i in range(gray.shape[0])])
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    # thresh = cv2.dilate(thresh, np.ones((9,9),np.uint8), iterations=2)
    #print(np.max(gray))
    ##print(thresh.shape[1])
    maskS = thresh != 0
    samples = maskSamples[maskS]
    #samples = [ p for p in maskSamples if thresh[p[0], p[1]] != 0]
    ##print(len(samples))
    #print(len(samples))
    if len(samples) > 5:
        #print('ok')
        #print(samples[0])
        samples = samples.reshape(-1, 2)
        #print(samples.shape)
        gmm.fit(samples)
        #print("res")
        #print(gmm.means_)
        #print(gmm.covariances_)
        means = gmm.means_
        covariances = gmm.covariances_
        rects = []
        for i in range(len(means)):
            point1 = tuple(np.flip(means[i].astype(int), 0)-5)
            point2 = tuple(np.flip(means[i].astype(int), 0)+5)
            point0 = tuple(np.flip(means[i].astype(int), 0))
            if center is None:
                center = means[0]
                centerHistory = deque([center for i in range(historySize)])
            centerHistory.appendleft(point0)
            centerHistory.pop()
            print('ok')
            print(np.mean(centerHistory,0))
            smoothedpoint0 = tuple(np.average(centerHistory,0, historyAverage).astype(int))
            smoothedpoint1 = tuple(np.average(centerHistory,0, historyAverage).astype(int) -5)
            smoothedpoint2 = tuple(np.average(centerHistory,0, historyAverage).astype(int) +5)
            print(smoothedpoint1)
            #print(point1)
            #rects.append(point)
            cv2.rectangle(frame, smoothedpoint1, smoothedpoint2, (0,255,0))
            cv2.rectangle(frame, point1, point2, (255,0,0))

            print("angle")
            # angle = angle + coefMotor * ( smoothedpoint0 - np.array(frame.shape[:2]) / 2.0)[1]
            angle = 50 + 0.2 * ( smoothedpoint0 - np.flip(np.array(frame.shape[:2]), 0) / 2.0)[0]
            print(smoothedpoint0)
            print(np.flip(np.array(frame.shape[:2]), 0))
            print(angle)
            print( (smoothedpoint0 - np.flip(np.array(frame.shape[:2]),0) / 2.0)[0])
            pwm.ChangeDutyCycle(5.0 * float(angle) / 100.0 + 5.0)
    #(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # loop over the contours
    #for c in cnts:
    #    # if the contour is too small, ignore it
    #    if cv2.contourArea(c) < args["min_area"]:
    #        continue
    #    
    #    # compute the bounding box for the contour, draw it on the frame,
    #    # and update the text
    #    (x, y, w, h) = cv2.boundingRect(c)
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #    text = "Occupied"
    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    firstFrame = gray
    # rawCapture.truncate(0)
    # if timeLoopIndex == 1:
    #     timeLoopindex = -1
    #     firstFrame = gray
    # timeLoopIndex += 1

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
