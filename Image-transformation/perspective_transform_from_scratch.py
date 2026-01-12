import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def manual_getPerspectiveTransform(src, dst):


    if len(src) != 4 or len(dst) != 4:
        raise ValueError("DLT requires exactly 4 corresponding points.")

    # Build the A matrix (8x9)
    A = np.zeros((8, 9))
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        A[2*i, :] = [x, y, 1, 0, 0, 0, -x*u, -y*u, -u]
        A[2*i+1, :] = [0, 0, 0, x, y, 1, -x*v, -y*v, -v]


    # Solve A * h = 0 using SVD
    # SVD of A = U * S * Vh
    U, S, Vh = np.linalg.svd(A)

    # The homography vector 'h' is the last column of V (or last row of Vh)
    h = Vh[-1, :]

    # Reshape the vector into a 3x3 matrix and normalize so H[2,2] is 1
    H = h.reshape(3, 3)
    H = H / H[2, 2]

    return H


def manual_warpPerspectiv(img, H, output_size):


  # 1. Calculate the homography matrix (H)


  # 2. Invert the homography matrix (H_inv)
  # To avoid gaps in the destination image, iterate over
  # destination pixels and map them back to the source image
  # (inverse mapping or gathering)
  H_inv = np.linalg.inv(H)


  # 3. Create an empty destination image
  dst_w, dst_h = output_size

  warped_img = np.zeros((dst_h, dst_w), dtype=np.float32)

  # 4. itenate over every pixel in the destination image
  for y_dst in range(dst_h):
    for x_dst in range(dst_w):


      # use homogeneour coordintes (x, y, 1) for matrix multiplication
      dst_point_h = np.array([x_dst, y_dst, 1])


      # Transfrom back to source coordinate using the inverse homograpy
      # result = H_inv * dest_point_h
      src_point_h = H_inv.dot(dst_point_h)


      # convert back from homogeneous coordinates (divide by the third component 'w')
      w = src_point_h[2]
      x_src = src_point_h[0] / w
      y_src = src_point_h[1] / w


      # perform nearest neighbor interpolation
      x_src_int = int(round(x_src))
      y_src_int = int(round(y_src))


      # check if the calculated source coordinnates are within the bound of the image
      if 0 <= x_src_int < img.shape[1] and 0 <= y_src_int < img.shape[0]:
         warped_img[y_dst, x_dst] = img[y_src_int, x_src_int]

  return warped_img

def main():

  img = cv.imread('../resource/passport_1.jpg', cv.IMREAD_GRAYSCALE)

  rect = np.array([
    [173, 144],
    [508, 49],
    [515, 428],
    [153, 374]
  ], dtype=np.float32)

  plt.subplot(121)
  plt.axis('off')
  plt.scatter(rect[:, 0], rect[:, 1])
  plt.imshow(img, cmap='gray')

  dest = np.array([
    [0, 0],
    [400, 0],
    [400, 250],
    [0, 250]], dtype=np.float32)

  H = manual_getPerspectiveTransform(rect, dest)
  warped = manual_warpPerspectiv(img, H, (400, 250))

  plt.subplot(122)
  plt.axis('off')
  plt.gca().invert_yaxis()
  plt.imshow(warped, cmap='gray')



if __name__ == '__main__':
  plt.figure()
  main()
  plt.show()
