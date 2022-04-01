#*****************How to read an image******************#

# import cv2

# img = cv2.imread('color.png')

# cv2.imshow('Output', img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()






#*****************How to write an image******************#

# import cv2

# img = cv2.imread('color.png')

# cv2.imshow('Output', img)

# cv2.imwrite('Output1.jpg', img)
# cv2.imwrite('Output2.png', img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()






#*****************How to get Image info******************#

# import cv2

# img = cv2.imread('Bird.jpg')

# cv2.imshow('Output', img)

# print(img.shape)

# print("Height :", img.shape[0])
# print("Width :", img.shape[1])
# print("Layers :", img.shape[2])

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************How to convert RGB to Gray Scale Image******************#

# Method 1

# import cv2

# img = cv2.imread('color.png')

# cv2.imshow('Output', img)

# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('GrayScale', gray_img)

# print(img.shape)

# print("Height :", img.shape[0])
# print("Width :", img.shape[1])
# print("Layers :", img.shape[2])

# print(gray_img.shape)

# print("Height :", gray_img.shape[0])
# print("Width :", gray_img.shape[1])
# print("Layers :", gray_img.shape[2])


# cv2.waitKey(0)

# cv2.destroyAllWindows()



# Method 2

# import cv2

# img = cv2.imread('color.png',0)

# cv2.imshow('GrayScale', img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************How to convert RGB to Binary Image******************#

# import cv2

# img = cv2.imread('color.png',0)

# cv2.imshow('GrayScale', img)

# ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# cv2.imshow('Binary', bw)

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************How to convert RGB to HSV Color Space******************#


# import cv2

# img = cv2.imread('color.png')

# img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# cv2.imshow('HSV Image', img_HSV)

# cv2.imshow('Hue-Channel', img_HSV[:, :, 0])

# cv2.imshow('Saturation-Channel', img_HSV[:, :, 1])

# cv2.imshow('Value-Channel', img_HSV[:, :, 2])

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************How to Extract RGB Color Space******************#

# import cv2
# import numpy as np

# img = cv2.imread('color.png')

# cv2.imshow('Output', img)

# B,G,R = cv2.split(img)

# zeros = np.zeros(img.shape[:2], dtype = "uint8")

# cv2.imshow("Red", cv2.merge([zeros,zeros,R]))
# cv2.imshow("Green", cv2.merge([zeros,G,zeros]))
# cv2.imshow("Blue", cv2.merge([B,zeros,zeros]))

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************How to Image Translate******************#

# import cv2
# import numpy as np

# img = cv2.imread('color.png')

# height, width = img.shape[:2]
# print(height)
# print(width)

# quartar_height, quartar_width = height/4, width/4

# print(quartar_height)
# print(quartar_width)

# T = np.float32([[1, 0, quartar_width],[0, 1, quartar_height]])

# print(T)

# img_translation = cv2.warpAffine(img, T, (width, height))

# cv2.imshow('Original image', img)
# cv2.imshow('Translation', img_translation)


# cv2.waitKey(0)

# cv2.destroyAllWindows()






#*****************How to rotate an image******************#

# import cv2
# import numpy as np

# img = cv2.imread('color.png')

# height, width = img.shape[:2]

# rotated_matrix = cv2.getRotationMatrix2D((width/2 , height/2), 180, 0.5)

# rotated_image = cv2.warpAffine(img, rotated_matrix, (width, height))

# cv2.imshow('Rotated image', rotated_image)
# cv2.imshow('Original', img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************How to Transpose an image******************#


# import cv2
# import numpy as np

# img = cv2.imread('color.png')

# rotated_img = cv2.transpose(img)

# cv2.imshow('Rotated image', rotated_img)
# cv2.imshow('Original', img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************How to resize an image******************#

# import cv2
# import numpy as np

# img = cv2.imread('color.png')

# img_linear_scaled = cv2.resize(img, None, fx= 0.75, fy =0.75)
# img_cubic_scaled = cv2.resize(img, None, fx= 1.25, fy=1.25, interpolation= cv2.INTER_CUBIC)

# img_area_scaled = cv2.resize(img, (300,200), interpolation= cv2.INTER_AREA )

# cv2.imshow('Linear Interplotation', img_linear_scaled)
# cv2.imshow('Cubic Interplotation', img_cubic_scaled)
# cv2.imshow('Area Interplotation', img_area_scaled)
# cv2.imshow('Original', img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************Image Pyramid******************#

# import cv2
# import numpy as np

# img = cv2.imread('color.png')

# smaller_img = cv2.pyrDown(img)
# larger_img = cv2.pyrUp(img)

# cv2.imshow("Original",img)
# cv2.imshow("Smaller",smaller_img)
# cv2.imshow("Larger", larger_img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()






#*****************Image Cropping******************#

# import cv2
# import numpy as np

# img = cv2.imread('color.png')

# height, width = img.shape[:2]

# start_row , start_column = int(height * .25) , int (width * .25)

# end_row, end_column = int(height * .75), int(width * .75)

# cropped = img[start_row:end_row, start_column:end_column]


# cv2.imshow("Cropped img", cropped)
# cv2.imshow("Original", img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()







#*****************Image Arithmetics******************#


# import cv2
# import numpy as np

# img = cv2.imread('color.png')

# M = np.ones(img.shape, dtype="uint8") * 150

# N = np.zeros(img.shape, dtype="uint8") + 150

 
# added = cv2.add(img, M)

# subtracted = cv2.subtract(img, M)

# multiplied = cv2.multiply(img, M)

# divided = cv2.divide(img, M)


# added1 = cv2.add(img, N)

# subtracted1 = cv2.subtract(img, N)

# multiplied1 = cv2.multiply(img, N)

# divided1 = cv2.divide(img, N)


# cv2.imshow("Original", img)
# cv2.imshow("Addition", added)
# cv2.imshow("Subtraction", subtracted)
# cv2.imshow("Multiplication", multiplied)
# cv2.imshow("Division", divided)

# cv2.imshow("Addition1", added1)
# cv2.imshow("Subtraction1", subtracted1)
# cv2.imshow("Multiplication1", multiplied1)
# cv2.imshow("Division1", divided1)

# cv2.waitKey(0)

# cv2.destroyAllWindows()






#*****************Image Bitwise Operation******************#


# import cv2
# import numpy as np

# square = np.zeros((300, 300), np.uint8)



# cv2.rectangle(square, (50,50), (250,250), 255, -1)
# cv2.imshow("Square", square)



# ellipse = np.zeros((300, 300), np.uint8)


# cv2.ellipse(ellipse, (150,150), (150,150), 30, 0, 180, 255, -1)
# cv2.imshow("Ellipse", ellipse)



# And = cv2.bitwise_and(square, ellipse)

# Or = cv2.bitwise_or(square, ellipse)

# Nor = cv2.bitwise_or(square, ellipse)

# Xor = cv2.bitwise_xor(square, ellipse)

# Not = cv2.bitwise_not(sqaure)

# cv2.imshow("AND-Operation", And)
# cv2.imshow("OR-Operation", Or)
# cv2.imshow("NOR-Operation", Nor)
# cv2.imshow("XOR-Operation", Xor)
# cv2.imshow("NOT-Operation", Not)
# cv2.waitKey(0)


# cv2.destroyAllWindows()







#*****************Image Blurring******************#

# import cv2

# import numpy as np

# img = cv2.imread('color.png')

# kernel_3x3 = np.ones((3,3), np.float32)/9 

# blurred = cv2.filter2D(img, -1, kernel_3x3)

# kernel_7x7 = np.ones((7,7), np.float32)/49 

# blurred1 = cv2.filter2D(img, -1, kernel_7x7)

# cv2.imshow("Original", img)
# cv2.imshow("Blurred", blurred)
# cv2.imshow("Blurred1", blurred1)

# cv2.waitKey(0)

# cv2.destroyAllWindows() 






#*****************Image Smoothing******************#


# import cv2

# img = cv2.imread('color.png')

# cv2.imshow('Output', img)


# blur = cv2.blur(img, (3,3))

# cv2.imshow("Blur Image", blur)

# Gaussian = cv2.GaussianBlur(img, (7,7), 0)

# cv2.imshow("Gaussian Blur", Gaussian)

# median = cv2.medianBlur(img, 5)

# cv2.imshow("Median Blur", median)


# bilateral = cv2.bilateralFilter(img, 9, 75, 75)
# cv2.imshow("bilateral Blur", bilateral )

# cv2.waitKey(0)

# cv2.destroyAllWindows()




#*****************Image Edge Detection******************#


# import cv2
# from cv2 import Laplacian

# img = cv2.imread('color.png',0)

# height, width = img.shape

# sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# sobel_or = cv2.bitwise_or(sobel_x, sobel_y)

# cv2.imshow('Output', img)
# cv2.imshow("Sobel_X_img", sobel_x)
# cv2.imshow("Sobel_Y_img", sobel_y)

# cv2.imshow("Sobel_Or", sobel_or)

# Laplacian = cv2.Laplacian(img, cv2.CV_64F)

# cv2.imshow("Laplacian", Laplacian)

# canny = cv2.Canny(img, 20, 170)

# cv2.imshow("Canny Image", canny)

# cv2.waitKey(0)

# cv2.destroyAllWindows()

