import cv2 as cv
import matplotlib.pyplot as plt


plt.figure()


img = cv.imread('../resource/pic1.jpg', cv.IMREAD_COLOR_RGB)


plt.subplot(3, 3, 1)
plt.title('Original Image')
plt.imshow(img)



'''
Rotating image
'''

plt.subplot(3, 3, 2)
plt.axis('off')
plt.title('roated image')
height, width,_ = img.shape
T = cv.getRotationMatrix2D((0, height), 20, 1)
imgRotated = cv.warpAffine(img, T, (width, height))
plt.imshow(imgRotated)



'''
Resizing Image
'''
img = img[77:113, 49:100, :]
height, width, _ = img.shape
print(img.shape)
scale = 1/2

interMethods = [
  cv.INTER_AREA,
  cv.INTER_LINEAR,
  cv.INTER_NEAREST,
  cv.INTER_CUBIC,
  cv.INTER_LANCZOS4
]

interTitle = ['area', 'linear', 'nearest', 'cubic', 'lanczos']

plt.subplot(3, 3, 3)
plt.title('original')
plt.imshow(img)
for i in range(len(interMethods)):
  plt.subplot(3, 3, i + 4)
  plt.title(interTitle[i])


  img = cv.resize(img, (int(scale * width), int(scale * height)), interpolation=interMethods[i])
  print(img.shape)
  plt.imshow(img)



plt.show()
