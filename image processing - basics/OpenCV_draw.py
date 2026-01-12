import cv2 as cv
import numpy as np


blank = np.zeros((500, 500, 3), dtype='uint8')
# cv.imshow('Blank', blank)
img = cv.imread('../resource/pic3.jpg')
# cv.imshow('Cat', img)



# 1. Paint the image a certain colour
blank[200:300, 300:400] = 0, 255, 0
cv.imshow('Paint', blank)

# 2. Draw a Rectangle
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=-1)
cv.imshow('Rectangle', blank)

# 3. Draw a Circle
cv.circle(blank, (255, 255), 100, (255, 0, 0), thickness=-1)
cv.imshow('Circle', blank)


# 4. Draw a line
cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=3)
cv.imshow('Line', blank)


# 5. Write a text
cv.putText(blank, 'Hello', (0, 255), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), thickness=2)
cv.imshow('Text', blank)
cv.waitKey(0)
