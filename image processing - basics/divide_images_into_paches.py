import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np



img = cv.imread('../resource/pic1.jpg')

height, width, _ = img.shape
print(height, width)


dp_x = width // 3
dp_y = height // 3


plt.figure()
figure_count = 1
for i in range(3):
  y_start = i * dp_y
  y_end = y_start + dp_y
  for j in range(3):

    x_start = j * dp_x
    x_end = x_start + dp_x


    print(y_start, y_end, x_start, x_end)
    patch_img = img[y_start:y_end, x_start:x_end]
    patch_img = cv.cvtColor(patch_img, cv.COLOR_BGR2RGB)

    plt.subplot(3, 3, figure_count)
    plt.axis('off')

    plt.imshow(patch_img)
    figure_count += 1

plt.show()
