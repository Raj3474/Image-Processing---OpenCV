import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def morphTrans():

  imgGray = cv.imread('../resource/pic2.jpg', cv.IMREAD_GRAYSCALE)

  imgGaus = cv.GaussianBlur(imgGray, (3, 3), 0)
  imgGaus = cv.adaptiveThreshold(imgGaus, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, 7, 3)


  plt.subplot(241)
  plt.imshow(imgGaus, cmap='gray')



  '''
  Morphological Operations
  '''
  kernel = np.ones((3, 3), np.uint8)
  erosion = cv.erode(imgGaus, kernel, iterations=1)
  plt.subplot(242)
  plt.imshow(erosion, cmap='gray')

  dilate = cv.dilate(imgGaus, kernel, iterations=1)
  plt.subplot(243)
  plt.imshow(dilate, cmap='gray')


  morph_types = [
    cv.MORPH_OPEN,
    cv.MORPH_CLOSE,
    cv.MORPH_GRADIENT,
    cv.MORPH_TOPHAT,
    cv.MORPH_BLACKHAT
  ]

  morph_titles = ['open', 'close', 'gradient', 'tophat', 'blackhat']

  for i in range(len(morph_types)):
    plt.subplot(2, 4, i+4)
    plt.imshow(cv.morphologyEx(imgGaus, morph_types[i], kernel), cmap='gray')
    plt.title(morph_titles[i])


if __name__ == '__main__':
  plt.figure()
  morphTrans()
  plt.show()
