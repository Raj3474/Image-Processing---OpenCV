import cv2 as cv

# Capture video from webcam
cap = cv.VideoCapture(0)
backgroundObject = cv.createBackgroundSubtractorMOG2(detectShadows = False)


while True:

  ret, frame = cap.read()

  fgmask = backgroundObject.apply(frame)

  real_part = cv.bitwise_and(frame, frame, mask=fgmask)
  cv.imshow('fmask', fgmask)
  cv.imshow('realpart', real_part)
  if cv.waitKey(30) & 0xFF == 27:
    break
