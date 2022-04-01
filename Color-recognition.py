# from turtle import width
import cv2    # importing cv2 library from openCv
import numpy as np # importing np library from numpy


vid = cv2.VideoCapture(0)  # VideoCapture() -> It is a function that is used to record real time video from webcam and we place the frame obtained 
# in a variable called cap

while True:               # For the implemention of function for the infinite loop
    ret, frame = vid.read() # read() -> It is a function that reads the input which return tuples as output. 
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # cvtColor() -> It is a builtin function of openCv which converts input into different format.
    lower_blue = np.array([90,50,50]) # It is lower bound of required blue color
    upper_blue = np.array([130, 255, 255]) # It is the upper bound of required blue color
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # we created a mask in hsv format with defined color.
    
    result = cv2.bitwise_and(frame, frame, mask=mask)  # it will perform and operator between frame and mask.
    
    cv2.imshow("Original", frame)  # return the original frame
    
    cv2.imshow('frame', result)     # return the frame after implementing bitwise operator 
    cv2.imshow('mask', mask)       # return the inRange frame.
    
    if cv2.waitKey(1) == 13:  # An interrupt is required to break out the function -> 13 -> ASCII CODE for Enter
        break
    
vid.release()            # the camera port got released, no longer in work.
cv2.waitKey(0)           # it helps to retrain the output until an interrupt is performed.
cv2.destroyAllWindows()  # it kills all the window