import cv2
import numpy as np

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Read the first frame and convert to float
_, img = cap.read()
averageValue1 = np.float32(img)

while True:
    # Capture next frame
    _, img = cap.read()

    # Update background model
    cv2.accumulateWeighted(img, averageValue1, 0.2)
    print(averageValue1[:, :, :] - [100, 100, 0])

    # Convert back to 8-bit for display
    resultingFrames1 = cv2.convertScaleAbs(averageValue1)
    print(averageValue1.shape)

    # Show both original and background model
    cv2.imshow('Original Frame', img)
    # cv2.imshow('Original Frame2', averageValue1)
    cv2.imshow('Background (Running Average)', resultingFrames1)
    cv2.imshow('foreground', cv2.absdiff(img, resultingFrames1))

    # Exit on Esc key
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
