import cv2
import numpy as np


def change_colors(x):
    # print('체크', x)
    r = cv2.getTrackbarPos("R", "Image")
    g = cv2.getTrackbarPos("G", "Image")
    b = cv2.getTrackbarPos("B", "Image")
    img[:] = [b, g, r]
    cv2.imshow("Image", img)


# np.zeros(shape[width, height, BGR], dataType)
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.namedWindow("Image")

cv2.createTrackbar("R", "Image", 0, 255, change_colors)
cv2.createTrackbar("G", "Image", 0, 255, change_colors)
cv2.createTrackbar("B", "Image", 0, 255, change_colors)

cv2.imshow("Image", img)
cv2.waitKey(0)
