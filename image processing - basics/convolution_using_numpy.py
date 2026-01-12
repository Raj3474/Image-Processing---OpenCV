import numpy as np
import cv2 as cv
# Define the input signal and kernel
signal = np.array([
  [1, 2, 3, 2, 3],
  [4, 5, 6, 3, 4],
  [7, 8, 9, 1, 1],
  [7, 8, 9, 1, 1],
  [7, 8, 9, 1, 1]], dtype=np.uint8)

# kernel = np.array([
#   [1, 1, 1],
#   [1, 1, 1],
#   [1, 1, 1]], dtype=np.uint8)

kernel = 1/9 * np.array([
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]],  dtype=np.float32)


def convolution(image, kernel):

  image_h, image_w = image.shape
  kernel_h, kernel_w = kernel.shape

  pad_h = kernel_h // 2
  pad_w = kernel_w //2


  image_padded = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REFLECT_101, None, 0)
  print(f'signal padded using cv: \n: {image_padded}')

  # Border padding as BORDER_REFLECT_101
  # padding left and right
  padded_image = np.concatenate((image[:, pad_w:0:-1], image, image[:,-2:-(pad_w + 2):-1]), axis=1)

  # padding top and bottom
  padded_image = np.concatenate((padded_image[pad_h:0:-1, :], padded_image, padded_image[-2:-(pad_h + 2):-1, :]), axis=0)
  print('manual padding: \n', padded_image)


  output = np.zeros((image_h, image_w))

  for i in range(image_h):
    for j in range(image_w):


      A = padded_image[i:i+kernel_h , j:j+kernel_w]
      output[i][j] = np.sum(A * kernel)

  output = np.clip(output, 0, 255)
  return np.round(output).astype(np.uint8)


result = convolution(signal, kernel)

result.astype(np.float64)
print(result.dtype)

print(f'output from manual convolution: \n {result}')

print(f'output from cv filter2D - \n {cv.filter2D(signal, ddepth=-1, kernel=kernel)}')
