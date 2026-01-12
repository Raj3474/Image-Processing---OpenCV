import cv2 as cv
import matplotlib.pyplot as plt


# read images in BGR format
image = cv.imread('../resource/shapes.jpg')

plt.figure()


# matplotlib expects images in RGB format
plt.subplot(3, 3, 1)
plt.title('Orignial Img')
plt.imshow(image)


image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.subplot(3, 3, 2)
plt.title('RGB Img')
plt.imshow(image_rgb)

image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
plt.subplot(3, 3, 3)
plt.title('HSV Img')
plt.imshow(image_hsv)

image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
plt.subplot(3, 3, 4)
plt.title('Gray Img')
plt.imshow(image_gray, cmap='gray')

image_lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
plt.subplot(3, 3, 5)
plt.title('Lab Img')
plt.imshow(image_lab)




# Color detecting using HSV color space

import numpy as np
lower_red = np.array([0, 120, 70])
upper_red = np.array([25, 255, 255])

mask = cv.inRange(image_hsv, lower_red, upper_red)

result = cv.bitwise_and(image_rgb, image_rgb, mask = mask)
plt.subplot(3, 3, 6)
plt.title('Only red')
plt.imshow(result)



# splitting RBG color channel

image = cv.imread('../resource/shapes.jpg')

B, G, R = cv.split(image)
plt.subplot(3, 3, 7)
plt.title('Blue channel')
plt.imshow(B, cmap='gray')

plt.subplot(3, 3, 8)
plt.title('Green channel')
plt.imshow(G, cmap='gray')

plt.subplot(3, 3, 9)
plt.title('Red Channel')
plt.imshow(R, cmap='gray')


plt.show()
