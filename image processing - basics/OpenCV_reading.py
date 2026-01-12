import cv2 as cv

# reading images
img = cv.imread('resource/pic1.jpg')
cv.imshow('Cat', img)

cv.waitKey(0)

# Reading Videos

# use cv.VideoCapture(0/1/2) to builtin camera.
capture = cv.VideoCapture(0)
while True:

  isTrue, frame = capture.read()


  cv.rectangle(frame, (0, 0), (frame.shape[1]//2, frame.shape[0]//2), (0, 0, 255), thickness=2)
  cv.imshow('Video', frame)
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()


