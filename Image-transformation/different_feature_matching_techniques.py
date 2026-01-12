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


def bruteForce():

  img1 = cv.imread('../resource/passport_1.jpg', cv.IMREAD_GRAYSCALE)
  img2 = cv.imread('../resource/passport_2.jpg', cv.IMREAD_GRAYSCALE)

  orb = cv.ORB_create()
  keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

  # print(type(keypoints1))
  # plt.scatter(keypoints1[1])
  bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
  matches = bf.match(descriptors1, descriptors2)
  matches = sorted(matches, key=lambda x:x.distance)
  nMatches = 8
  imgMatch = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:nMatches],
                            None, cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

  plot(imgMatch, 'mathces')


def knnBruteForce():
  img1 = cv.imread('../resource/passport_1.jpg', cv.IMREAD_GRAYSCALE)
  img2 = cv.imread('../resource/passport_2.jpg', cv.IMREAD_GRAYSCALE)

  sift = cv.SIFT_create()
  keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
  keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

  # print(type(keypoints1))
  # plt.scatter(keypoints1[1])
  bf = cv.BFMatcher()
  matches = bf.knnMatch(descriptors1, descriptors2, k = 2)

  goodMatches = []
  testRatio = 0.4

  for m,n in matches:
    if m.distance < testRatio*n.distance:
      goodMatches.append([m])


  # print(goodMatches)
  imgMatch = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, goodMatches, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

  plot(imgMatch, 'matches')


def FLANN():
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
  matchesMask = [[0, 0] for i in range(len(matches))]

  testRatio = 0.4

  for i, (m,n) in enumerate(matches):

    if m.distance < testRatio*n.distance:
      matchesMask[i] = [1, 0]

  drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                     matchesMask=matchesMask, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

  # print(goodMatches)
  imgMatch = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **drawParams)
  print(matchesMask)
  plot(imgMatch, 'matches')

if __name__ == '__main__':
  plt.figure()
  # bruteForce()
  # knnBruteForce()
  FLANN()
  plt.show()
