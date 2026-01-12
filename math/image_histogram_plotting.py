import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('../resource/pic1.jpg', cv.IMREAD_GRAYSCALE)

print(img.shape)
print(img.flatten().shape)
plt.figure()

# plt.imshow(img, cmap='gray')
# plt.hist(img.ravel(), 256, [0, 256])

histr = cv.calcHist([img], [0], None, [256], [0, 256])
plt.plot(histr)
plt.show()
