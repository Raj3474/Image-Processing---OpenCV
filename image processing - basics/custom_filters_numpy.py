import cv2 as cv
import numpy as np

img = cv.imread('../resource/pic1.jpg')

print(img.shape, img[1:][2:4])
img[1:50, 20:40 ] = 255, 0, 0

print(img[1:50, 20:40])

cv.imshow('modidy', img)

kernel = np.array([[2, 4, 2],
                   [4, 8, 4],
                   [2, 4, 2]
                   ], dtype=np.float32)

kernel *= 1/32
filtered_img = cv.filter2D(img, ddepth=-1, kernel=kernel)

cv.imshow('Before', img)
cv.imshow('After', filtered_img)
cv.waitKey(10000)

cv.destroyAllWindows()
