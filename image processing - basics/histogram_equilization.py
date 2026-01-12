import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main():



  img = cv.imread('../resource/pic5.jpg', cv.IMREAD_GRAYSCALE)



  hist = cv.calcHist([img], [0], None, [256], [0, 256])
  cdf = hist.cumsum()
  cdfNorm = cdf * float(hist.max()) / cdf.max()
  plt.subplot(241)
  plt.imshow(img, cmap='gray')
  plt.subplot(245)
  plt.plot(hist)
  plt.plot(cdfNorm, color='b')

  # Histogram Equalization
  equimg = cv.equalizeHist(img)
  equhist = cv.calcHist([equimg], [0], None, [256], [0, 256])
  equcdf = equhist.cumsum()
  equcdfNorm = equcdf * float(equhist.max()) / equcdf.max()
  plt.subplot(242)
  plt.imshow(equimg, cmap='gray')
  plt.subplot(246)
  plt.plot(equhist)
  plt.plot(equcdfNorm, color='b')


  # equlization using CLAHE

  claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))
  claheImg = claheObj.apply(img)
  clahehist = cv.calcHist([claheImg], [0], None, [256], [0, 256])
  clahecdf = clahehist.cumsum()
  clahecdfNorm = equcdf * float(clahehist.max()) / clahecdf.max()
  plt.subplot(243)
  plt.imshow(claheImg, cmap='gray')
  plt.subplot(247)
  plt.plot(clahehist)
  plt.plot(clahecdfNorm, color='b')


  # orignial Image
  orig_img = cv.imread('../resource/pic1.jpg', cv.IMREAD_GRAYSCALE)



  hist = cv.calcHist([orig_img], [0], None, [256], [0, 256])
  cdf = hist.cumsum()
  cdfNorm = cdf * float(hist.max()) / cdf.max()
  plt.subplot(244)
  plt.imshow(img, cmap='gray')
  plt.subplot(248)
  plt.plot(hist)
  plt.plot(cdfNorm, color='b')



if __name__ == '__main__':
  plt.figure()
  main()
  plt.show()
