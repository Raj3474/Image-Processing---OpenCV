import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os



def contours ():
  root = os.getcwd()
  imgPath = os.path.join(root, '../resource/pic4.jpg')

  img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)


  plt.figure()
  plt.subplot(231)
  plt.axis("off")
  plt.title('original image')
  plt.imshow(img, cmap='gray')

  img = cv.GaussianBlur(img, (7, 7), 0)
  plt.subplot(232)
  plt.axis("off")
  plt.title('original image after blur')
  plt.imshow(img, cmap='gray')

  _, thresh = cv.threshold(img, 250, 255, cv.THRESH_BINARY_INV)
  plt.subplot(233)
  plt.axis("off")
  plt.title('Binary image')
  plt.imshow(thresh, cmap='gray')



  kernel = np.ones((3, 3), np.uint8)
  thresh = cv.dilate(thresh, kernel)
  plt.subplot(234)
  plt.axis("off")
  plt.title('Binary image after dilation')
  plt.imshow(thresh, cmap='gray')

  contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  print(len(contours), contours)
  # print(f'number of contours found = {len(contours)}')



  # drawing coutours

  # print(contours[0])
  img_copy = cv.imread(imgPath)
  cv.drawContours(img_copy, contours, -1, (0, 200, 200), 3)

  plt.subplot(235)
  plt.axis("off")
  plt.title('contours')
  plt.imshow(img_copy, cmap='gray')


  x, y, w, h = cv.boundingRect(contours[1])
  cv.rectangle(img_copy, (x,y), (x+w, y+h), (0, 0, 255), 3)
  plt.subplot(236)
  plt.axis("off")
  plt.title('bounding box')
  plt.imshow(img_copy, cmap='gray')




  plt.show()
contours()
