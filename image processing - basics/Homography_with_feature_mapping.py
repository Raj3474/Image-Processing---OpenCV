import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

num_plots = 0
def plot(img, title):

  global num_plots
  # plt.subplot(2, 3, num_plots:=num_plots + 1)
  plt.axis('off')
  plt.title(title)
  plt.imshow(img)


def main():
  img1 = cv.imread('../resource/passport_1.jpg', cv.IMREAD_GRAYSCALE)
  img2 = cv.imread('../resource/passport_2.jpg', cv.IMREAD_GRAYSCALE)

  sift = cv.SIFT_create()
  keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
  keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

  FLANN_INDEX_KDTREE = 1
  nKDtrees = 5
  nLeafChecks = 50
  nNeighbors = 2

  indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=nKDtrees)
  searchParams = dict(checks=nLeafChecks)
  flann = cv.FlannBasedMatcher(indexParams, searchParams)

  matches = flann.knnMatch(descriptors1, descriptors2, k=nNeighbors)

  testRatio = 0.5
  goodMatches = []

  for i, (m,n) in enumerate(matches):

    if m.distance < testRatio*n.distance:
      goodMatches.append(m)


  minGoodMatches = 20

  print(goodMatches[0].queryIdx)

  if len(goodMatches) > minGoodMatches:

    srcPts = np.float32([keypoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    dstPts = np.float32([keypoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

    print(srcPts)
    errorThreshold = 5

    # RANSAC method
    M, mask = cv.findHomography(srcPts, dstPts, cv.RANSAC, errorThreshold)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape

    imgBorder = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

    print('img border shape ', imgBorder.shape)
    warpedImgBorder = cv.perspectiveTransform(imgBorder, M)

    img2 = cv.polylines(img2, [np.int32(warpedImgBorder)], True, 255, 3, cv.LINE_AA)

  else:
    print('not enough mathces')

  print(type(matchesMask))
  green = (0, 255, 0)

  drawParams = dict(matchColor=green, singlePointColor=green,
                     matchesMask=matchesMask, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS )

  print(type(goodMatches))
  print(type(matches))
  print(matchesMask)
  imgMatch = cv.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, None, **drawParams)

  plot(imgMatch, 'matches')

if __name__ == '__main__':
  plt.figure()
  main()
  plt.show()
