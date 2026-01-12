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


def convolution(image, kernel):

  channels = cv.split(image)
  num_channels = len(channels)

  image_h, image_w = image.shape[:2]
  kernel_h, kernel_w = kernel.shape

  pad_h = kernel_h // 2
  pad_w = kernel_w //2

  # padding using cv.copyMakeBorder
  # image_padded = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REFLECT_101, None, 0)
  # print(f'image padded using cv: \n: {image_padded}')

  # padding using numpy pad
  # padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)) + ((0, 0),) * (image.ndim == 3), mode='reflect')
  # print(padded_image.shape)


  # Border padding as BORDER_REFLECT_101
  padded_image = np.zeros((image_h + 2 * pad_h, image_w + 2 * pad_w, num_channels), dtype=np.float32)


  for c in range(num_channels):

    # padding left and right
    if image.ndim == 3:
      A = np.concatenate((image[:, pad_w:0:-1, c], image[:, :, c], image[:,-2:-(pad_w + 2):-1, c]), axis=1)
    else:
      A = np.concatenate((image[:, pad_w:0:-1], image[:, :], image[:,-2:-(pad_w + 2):-1]), axis=1)

    #padding top and bottom
    A = np.concatenate((A[pad_h:0:-1, :], A[:, :], A[-2:-(pad_h + 2):-1, :]), axis=0)
    # print(padded_image)

    if np.ndim == 3:
      padded_image[:, :, c] = A
    else:
      padded_image[:, :, c] = A

    # print('diff padded channel', padded_image[:3, :3, c].ravel())


  output = np.zeros_like(image, dtype=np.float32)
  for c in range(num_channels):
    for i in range(image_h):
      for j in range(image_w):

        # print('inside for loop ', i, i+kernel_h , j, j+kernel_w, c)
        A = padded_image[i:i+kernel_h , j:j+kernel_w, c]
        # print(1111, A)
        if image.ndim == 3:
          output[i, j, c] = np.sum(A * kernel)
        else:
          output[i, j] = np.sum(A * kernel)

  output = np.clip(output, 0, 255)
  return np.round(output).astype(np.uint8)



def GaussianFilter(img):

  # gaussian kernel 5x5
  GausKernel = 1/273 * np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7,],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
  ])

  imgGaus = convolution(img, GausKernel)

  return imgGaus


def Gradient(image):

  # using Sobel filter
  Kx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]], dtype=np.float32)

  Ky = np.array([
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]], dtype=np.float32)

  Ix = cv.filter2D(image, ddepth=-1, kernel=Kx)
  Iy = cv.filter2D(image, ddepth=-1, kernel=Ky)

  plot(Ix, 'a-ax')
  plot(Iy, 'y axis')

  G = np.hypot(Ix, Iy)
  theta = np.atan2(Iy, Iy)


  return (G, theta)


def non_max_suppression(img, D):
  Y, X = img.shape
  Z = np.zeros((Y, X), dtype=np.int32)


  angle = D * 180. / np.pi
  angle[angle < 0] += 180


  for i in range(1, Y-1):
    for j in range(1, X-1):

      q = 255
      r = 255

      # angle 0
      if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
        q = img[i][j-1]
        r = img[i][j+1]

      # angle 45
      elif (22.5 <= angle[i, j] < 67.5):
        q = img[i+1][j-1]
        r = img[i-1][j+1]

      # angle 90
      elif (67.5 <= angle[i, j] < 112.5):
        q = img[i+1][j]
        r = img[i-1][j]

      # angle 135
      elif (112.5 <= angle[i, j] < 157.5):
        q = img[i+1][j+1]
        r = img[i-1][j-1]


      if (img[i,j] >= q) and (img[i,j] >=r):
        Z[i, j] = img[i,j]
      else:
        Z[i, j] = 0

  return Z


def double_thresholding(img):

  STRONG_PIXEL = 255
  WEAK_PIXEL = 100
  # 0.09
  # 0.17
  hiThresh = 43
  loThresh = 7

  Y, X = img.shape
  res = np.zeros((Y,X), dtype=np.int32)

  res[img > hiThresh] = STRONG_PIXEL
  res[(img > loThresh) & (img <= hiThresh)] = WEAK_PIXEL

  return res


def hysteresis(img):
  STRONG_PIXEL = 255
  WEAK_PIXEL = 100

  M, N = img.shape

  for i in range(1, M-1):
    for j in range(1, N-1):

      if (img[i, j] == WEAK_PIXEL):
        dir = [-1, 0, 1]

        is_strong_neighbour = False
        for dx in dir:
          for dy in dir:


            if dx == 0 and dy == 0:
              continue

            if img[i+dx, j+dy] == 255:
              is_strong_neighbour = True
              break

        if is_strong_neighbour:
          img[i, j] = STRONG_PIXEL
        else:
          img[i,j] = 0


  return img



"""
Canny Edge detector from Scratch
"""

def main():
  img = cv.imread('../resource/madhu.png', cv.IMREAD_GRAYSCALE)
  plot(img, 'Orignal')


  # Step1: Smooth the image
  imgGaus = GaussianFilter(img.copy())
  plot(imgGaus, 'Gaussian Image')

  # Step2: calculate Gradient
  GradientMat, thetaMat = Gradient(imgGaus.copy())
  plot(GradientMat, 'sobel gradient')


  # Step3: Non Maximum suppresion
  non_max_img = non_max_suppression(GradientMat, thetaMat)
  plot(non_max_img, 'non max suppression')

  # Step4: Double Thresholding
  img_double_thres = double_thresholding(non_max_img)
  plot(img_double_thres, 'Double Thresholding')


  # Step5: Edge tracking by Hysteresis
  img_hysteresis = hysteresis(img_double_thres)
  plot(img_hysteresis, 'Hysteresis Image')



if __name__ == '__main__':
  plt.figure()
  main()
  plt.show()
