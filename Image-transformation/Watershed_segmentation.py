import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def plot(img, position, title):

  plt.subplot(3, 4, position)
  plt.axis("off")
  plt.title(title)
  plt.imshow(img, cmap='gray')


def readImage(img):
  root = os.getcwd()
  imgPath = os.path.join(root, f'../resource/{img}')
  return cv.imread(imgPath, cv.IMREAD_GRAYSCALE)


def main():
  gray_img = readImage(img='coins.webp')
  plot(gray_img, 1, 'original Image')

  # apply ostu's binarization

  gray_img = cv.GaussianBlur(gray_img, (7, 7), 0)

  ret, thresh = cv.threshold(gray_img, 220, 255, cv.THRESH_BINARY_INV)
  plot(thresh, 2, 'threshold image')


  #noise removal
  kernel = np.ones((3, 3), np.uint8)
  opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

  plot(opening, 3, 'opening')


  # sure background area
  sure_bg = cv.dilate(opening, kernel, iterations=3)
  plot(sure_bg, 4, 'sure_bg')


  #finding sure foreground area
  dist_transform = cv.distanceTransform(sure_bg, cv.DIST_L2,5)
  plot(dist_transform, 5, 'dist_transform')
  ret, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv.THRESH_BINARY)
  plot(sure_fg, 6, 'sure fg')


  # sure unknown region
  sure_fg = np.uint8(sure_fg)
  unknown = cv.subtract(sure_bg, sure_fg)
  plot(unknown, 7, 'sure unknown')


  # Marker labelling
  ret, markers = cv.connectedComponents(sure_fg)


  objects = set(map(int, markers.flatten()))
  print('from connected component ', len(objects), objects)

  # Add one to all labels so that sure background is not 0, but 1
  markers  = markers + 1

  # Now, mark the region of unknown with zero
  markers[unknown == 255] = 0

  plot(markers, 8, 'markers')
  objects = set(map(int, markers.flatten()))
  print('from component after adding the boundary', len(objects), objects)


  markers = np.int32(markers)

  img = cv.imread('../resource/coins.webp', cv.IMREAD_COLOR)
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  plot(img, 10, 'input image')
  markers = cv.watershed(img, markers)



  img[markers == -1] = [255, 0, 0]

  plot(markers, 9, 'Final valley created from watershed')
  plot(img, 11, 'Final image output')

  print(set(map(int, markers.flatten())))


if __name__ == '__main__':
  plt.figure()
  main()
  plt.show()
