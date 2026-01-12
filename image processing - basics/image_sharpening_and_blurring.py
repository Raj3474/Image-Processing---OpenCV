import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

num_plots = 0
def plot(img, title):

  global num_plots
  plt.subplot(2, 3, num_plots := num_plots + 1)
  plt.title(title)
  plt.axis('off')
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  plt.imshow(img)

plt.figure()

img = cv.imread('../resource/pic1.jpg')
# plot(img, 'original Image')


Gaus_kernel = 1/16 * np.array([
  [1, 2, 1],
  [2, 4, 2],
  [1, 2, 1]
])
imgGaus = cv.filter2D(img, ddepth=-1, kernel=Gaus_kernel)
plot(imgGaus, 'Gausssion blurring')

sharp_kernel_basic = np.array([
  [0, 1, 0],
  [1, -4, 1],
  [0, 1, 0]
])

img_fil = cv.filter2D(img, ddepth=-1, kernel = sharp_kernel_basic)
img_basic_laplace = cv.absdiff(img, img_fil)
plot(img_basic_laplace, 'Basic Laplace sharpening')


sharp_kernel_strong = np.array([
  [1, 1, 1],
  [1, -8, 1],
  [1, 1, 1]
])

img_fil = cv.filter2D(img, ddepth=-1, kernel = sharp_kernel_strong)
img_strong_laplace = cv.add(img, img_fil)
plot(img_strong_laplace, 'strong Laplace sharpening')

imgGaus = cv.GaussianBlur(img, (7, 7), 0)
mask = cv.absdiff(img, imgGaus)
unsharp = img + mask
plot(unsharp, 'unsharp sharpening')


sharp_kernel_basic = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])

img_fil = cv.filter2D(img, ddepth=-1, kernel = sharp_kernel_basic)
plot(img_fil, 'High boost filter')

high_boost_kernel = np.array([
  [-1, -1, -1],
  [-1, 9, -1],
  [-1, 1, -1]
])

img_fil = cv.filter2D(img, ddepth=-1, kernel = high_boost_kernel)
img_strong_laplace = cv.absdiff(img, img_fil)
plot(img_strong_laplace, 'strong high boost filter sharpenings')

plt.show()
