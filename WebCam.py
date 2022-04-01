# ********************OpenCV Webcam*********************** #

# import cv2

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow("Live Video", frame)
#     if cv2.waitKey(1) == 13:
#         break
    
# cap.release()
# cv2.destroyAllWindows()






# ********************Capture Image Using Webcam*********************** #

# import cv2
# import matplotlib.pyplot as plt

# cap = cv2.VideoCapture(0)

# if cap.isOpened():
#     ret, frame = cap.read()
#     print(ret)
#     print(frame)
    
# else: 
#     ret = False

# img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# plt.imshow(img1)
# plt.title("Camera Image")
# plt.xticks([])
# plt.yticks([])
# plt.show()
# cap.release()





# ********************Edge Detection using WebCam*********************** #

# import cv2

# def sketch(image):
    
#     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     img_gray_blur = cv2.GaussianBlur(img_gray,(5,5), 0)
    
#     canny_edge = cv2.Canny(img_gray_blur, 10, 70)
    
#     ret, mask = cv2.threshold(canny_edge, 70, 255, cv2.THRESH_BINARY)
    
#     return mask

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow("Live Sketch Video", sketch(frame))
#     if cv2.waitKey(1) == 13:
#         break
    
# cap.release()
# cv2.destroyAllWindows()



# ********************Color Filtering using WebCam*********************** #

# import cv2
# import numpy as np

# device = cv2.VideoCapture(0)

# while True:
#     ret, frame = device.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     lower_range = np.array([110, 50, 50])  
#     upper_range = np.array([130, 255, 255])
    
#     mask = cv2.inRange(hsv, lower_range, upper_range) # mask is a filter which filter out only blue color and rest color will be hold.
    
#     cv2.imshow("Show", mask)
#     cv2.imshow("show1", frame)
    
#     if cv2.waitKey(1) == 13:
#         break
    
# device.release()
# cv2.destroyAllWindows()


# ********************Color Filtering with Bitwise Operator using WebCam*********************** #

# import cv2
# import numpy as np

# device = cv2.VideoCapture(0)


# while True:
#     ret, frame = device.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     lower_range = np.array([110, 50, 50])  
#     upper_range = np.array([130, 255, 255])
    
#     mask = cv2.inRange(hsv, lower_range, upper_range)
    
#     result = cv2.bitwise_and(frame, frame, mask=mask)
    
#     cv2.imshow("Original", frame)
#     cv2.imshow("Result", result)
#     cv2.imshow("Mask", mask)
    
#     if cv2.waitKey(1) == 13:
#          break
    
# device.release()
# cv2.destroyAllWindows()