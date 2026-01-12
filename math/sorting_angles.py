import numpy as np

P = np.array([
  [0, 0],
  [0, 10],
  [10, 10],
  [10, 0]
])


C = np.mean(P, axis=0)
print(C)

a = np.atan2(P[:, 1], P[:, 0])
b = np.atan2(P[:, 1] - C[1], P[:, 0] - C[0])

print((180 / np.pi) * a, (180 / np.pi) * b)


print(np.argsort(-b))
print(np.argsort(-a))
