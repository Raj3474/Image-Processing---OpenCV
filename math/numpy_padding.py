import numpy as np

# Create a sample 3D array (2x3x4)
arr_3d = np.arange(18).reshape((3, 3, 2))
print("Original array shape:", arr_3d.shape)
print(arr_3d)

# Pad with 1 unit of zero on all sides of all axes
padded_arr_3d = np.pad(arr_3d, ((0, 1), (1,1), (0, 0)), mode='constant')

print("Padded array shape:", padded_arr_3d)
# The new shape is (2+1+1, 3+1+1, 4+1+1) -> (4, 5, 6)
