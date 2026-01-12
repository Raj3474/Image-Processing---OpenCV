import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def main():

  img = cv.imread('../resource/doc_input.jpg', cv.IMREAD_COLOR_RGB)


  rect = np.array([
    [173, 144],
    [508, 49],
    [515, 428],
    [153, 374]
  ], dtype=np.float32)

  plt.subplot(121)
  plt.axis('off')
  plt.scatter(rect[:, 0], rect[:, 1])
  plt.imshow(img, cmap='gray')

  dest = np.array([
    [0, 0],
    [400, 0],
    [400, 250],
    [0, 250]], dtype=np.float32)

  M = cv.getPerspectiveTransform(rect, dest)
  print(M, M.dtype)
  warped = cv.warpPerspective(img, M, (400, 250))

  plt.subplot(122)
  plt.axis('off')
  plt.imshow(warped, cmap='gray')


if __name__ == '__main__':
  plt.figure()
  main()
  plt.show()
