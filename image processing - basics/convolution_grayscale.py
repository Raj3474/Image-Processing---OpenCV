import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('../resource/pic1.jpg', cv.IMREAD_GRAYSCALE)
# print(img.dtype)


# kernel = 1/9 * np.array([
#   [1, 1, 1],
#   [1, 1, 1],
#   [1, 1, 1]], dtype=np.float32)

kernel = 1/9 * np.array([
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]],  dtype=np.float32)

def convolution(image, kernel):

  image_h, image_w = image.shape
  kernel_h, kernel_w = kernel.shape

  pad_h = kernel_h // 2
  pad_w = kernel_w //2

  # Border padding as BORDER_REFLECT_101
  # padding left and right
  padded_image = np.concatenate((image[:, pad_w:0:-1], image, image[:,-2:-(pad_w + 2):-1]), axis=1, dtype=np.float32)

  # padding top and bottom
  padded_image = np.concatenate((padded_image[pad_h:0:-1, :], padded_image, padded_image[-2:-(pad_h + 2):-1, :]), axis=0, dtype=np.float32)
  print(padded_image[:4, :4])
  output = np.zeros((image_h, image_w), dtype=np.float32)

  for i in range(image_h):
    for j in range(image_w):


      A = padded_image[i:i+kernel_h , j:j+kernel_w]
      output[i][j] = np.sum(A * kernel)

  output = np.clip(output, 0, 255)
  return np.round(output).astype(np.uint8)


plt.figure()

print(111, img[0:3, 0:3])
plt.subplot(1, 3, 1)
plt.title('Orignal Pic')
plt.imshow(img, cmap="gray")

img1 = img.copy()
img1 = convolution(img1, kernel)
print(222, img1[0:3, 0:3])
print(333, img[0:3, 0:3])
plt.subplot(1, 3, 2)
plt.title('manual convolution')
plt.imshow(img1, cmap="gray")



img2 = cv.filter2D(img, ddepth=-1, kernel=kernel)
print(444, img2[0:3, 0:3])
plt.subplot(1, 3, 3)
plt.title('Open CV convolution')
plt.imshow(img2, cmap="gray")
print(img.dtype, img1.dtype, img2.dtype)

plt.show()
