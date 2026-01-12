import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


num_plots = 0
def plot(img, title):

  global num_plots
  plt.subplot(3, 3, num_plots:=num_plots + 1)
  plt.axis('off')
  plt.title(title)
  plt.imshow(img, cmap='gray')


def rank_keypoints_by_harris(img, keypoints, top_k=20):

  # print(type(keypoints[0]), dir(keypoints[0]), keypoints[0].response)
  # compute harris corner response
  harris_response = cv.cornerHarris(img, blockSize=2, ksize=3, k=0.04)


  # Normalize Harris response map (optional bu useful for visualization or debuggin)
  harris_response = cv.normalize(harris_response, None, 0, 255, cv.NORM_MINMAX)


  # get Harris score for each keypoint
  for kp in keypoints:
    x, y = int(kp.pt[0]), (int(kp.pt[1]))

    if 0 <= y < harris_response.shape[0] and 0 <= x < harris_response.shape[1]:
      kp.response = harris_response[y, x]
    else:
      kp.response = 0
  # print(keypoints[0].response)

  keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)
  return keypoints_sorted[:top_k]


def visualize_matches(matches):
  #After matching the keypoints,we want to understand how good those matches were.
  #That's where this histogram comes in!
  distances = [m.distance for m in matches]
  plt.hist(distances, bins=50)
  plt.title("ORB Match Distances")
  plt.xlabel("Hamming Distance")
  plt.ylabel("Frequency")
  plt.show()


def main():

  img1 = cv.imread('../resource/passport_1.jpg', cv.IMREAD_GRAYSCALE)
  img2 = cv.imread('../resource/passport_2.jpg', cv.IMREAD_GRAYSCALE)


  '''
  # 1.....FAST keypoint detection
  '''
  fast = cv.FastFeatureDetector_create()
  keypoints1 = fast.detect(img1, None)
  keypoints2 = fast.detect(img2, None)

  # Draw keypoints
  img1_with_kp = cv.drawKeypoints(img1, keypoints1, None, color=(255, 0, 0))
  img2_with_kp = cv.drawKeypoints(img2, keypoints2, None, color=(255, 0, 0))



  # show the results
  plot(img1_with_kp, 'Image 1 with FAST Keypoints')
  plot(img2_with_kp, 'Image 2 with FAST Keypoints')


  '''
  # 2.... Rank keypoints with Harris score (Harris corner measure)
  '''

  top_k = 100
  top_kp1 = rank_keypoints_by_harris(img1, keypoints1, top_k)
  top_kp2 = rank_keypoints_by_harris(img2, keypoints2, top_k)

  img1_top = cv.drawKeypoints(img1, top_kp1, None, color=(255, 0, 0))
  img2_top = cv.drawKeypoints(img2, top_kp2, None, color=(255, 0, 0))
  plot(img1_top, 'image1 top 50 keypoints')
  plot(img2_top, 'image2 top 50 keypoints')



  '''
  # 3..... Orient the keypoints
  '''


  # use ORB to compute orientation only (not descriptors)
  orb = cv.ORB_create()
  keypoints1 = orb.compute(img1, keypoints1)[0]
  keypoints2 = orb.compute(img2, keypoints2)[0]

  img1_with_oriented_kp = cv.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
  img2_with_oriented_kp = cv.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

  plot(img1_with_oriented_kp, 'Image 1 roated keypoints')
  plot(img2_with_oriented_kp, 'Image 1 roated keypoints')



  '''
  # 4 compute descriptos using ORB (Rotated BRIEF)
  '''
  # use ORB's compute() to compute Rotated BRIEF descriptors
  descriptors1 = orb.compute(img1, keypoints1)[1]
  descriptors2 = orb.compute(img2, keypoints2)[1]

  img1_with_desc = cv.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=0)
  img2_with_desc = cv.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)


  '''
  # 5 Match keypoints between images
  '''
  # Match descriptors using Brute-Force Matcher with Hamming distnace
  bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
  matches = bf.match(descriptors1, descriptors2)


  matches = sorted(matches, key=lambda x:x.distance)

  matched_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:100], None, flags=2)
  # plot(matched_img, 'Final result')

  # Show the matches
  plt.figure(figsize=(16, 8))
  plt.title("Top 50 Feature Matches (ORB + BFMatcher)")
  plt.imshow(matched_img)
  plt.axis('off')
  plt.show()


  # visualize the matches
  visualize_matches(matches)


if __name__ == '__main__':
  plt.figure()
  main()
  plt.show()
