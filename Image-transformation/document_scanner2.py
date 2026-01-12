import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


num_plot = 0

def plot(img, title='', isGray=True):
  global num_plot
  plt.subplot(2, 4, num_plot := num_plot + 1)
  plt.axis('off')
  plt.title(title)

  if isGray:
    plt.imshow(img, cmap= 'gray')
  else:
    plt.imshow(img)

def orderPoints(pts):

  plt.scatter(pts[:, 0], pts[:, 1])
  rect = np.zeros((4, 2), dtype=np.float32)
  C = np.mean(pts, axis=0)

  plt.scatter(C[0], C[1])
  angles = np.arctan2( pts[:, 1] - C[1], pts[:, 0] -C[0])


  clockwise_pts = np.argsort(angles)
  angles = np.sort(angles)

  rect = pts[clockwise_pts]

  return rect.astype(np.float32)

def main():
  global num_plot
  imgGray = cv.imread('../resource/receipt.webp', cv.IMREAD_GRAYSCALE)
  plot(imgGray)

  imgGaus = cv.GaussianBlur(imgGray, (5, 5), 0)

  edges = cv.Canny(imgGaus, 70, 200)
  plot(edges, 'edges')

  contours, _= cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
  # print(len(contours), contours)

  img = cv.imread('../resource/receipt.webp')
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


  contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

  for c in contours:

    peri = cv.arcLength(c, True)
    # print(peri)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)
    # print(approx)

    if len(approx) == 4:
      screencnt = approx
      break

  cv.drawContours(img, [screencnt], -1, (0, 200, 200), 3)
  plot(img, 'contours', False)

  src  = orderPoints(screencnt.reshape(4, 2))
  tl, tr, br, bl = src
  # print(src)
  plt.imshow(img, cmap='gray')

  widthA = np.sqrt((tr[1] - tl[1]) ** 2 + (tr[0] - tl[0]) ** 2)
  widthB = np.sqrt((br[1] - bl[1]) ** 2 + (br[0] - bl[0]) ** 2)
  maxWidth = int(max(widthA, widthB))

  heightA = np.sqrt((tl[1] - bl[1]) ** 2 + (tl[0] - bl[0]) ** 2)
  heightB = np.sqrt((tr[1] - br[1]) ** 2 + (tr[0] - br[0]) ** 2)
  maxHeight = int(max(heightA, heightB))

  dest = np.array([
    [0, 0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight],
    [0, maxHeight-1]], dtype=np.float32)

  M = cv.getPerspectiveTransform(src, dest)
  warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
  plot(warped, 'Warped')

  warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
  plot(warped, 'warped gray')
  # warpedGaus = cv.GaussianBlur(warped, (2, 2), 0)
  # plot(warpedGaus)
  warped_BW = cv.adaptiveThreshold(warped, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 9)
  plot(warped_BW, 'Final output')


if __name__ == '__main__':
  plt.figure()
  main()
  plt.show()
