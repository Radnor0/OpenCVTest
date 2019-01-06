import numpy as np
import cv2
from math import atan, pi

def trueHeight(rect):
    pts = cv2.boxPoints(rect)
    y = list(map(lambda pt: pt[1], pts))
    return max(y[0] - y[1], y[0] - y[2], y[0] - y[3])

def trueAngle(rect):
    pts = cv2.boxPoints(rect)
    point1 = pts[0]
    y = list(map(lambda p: p[1], pts))
    if abs(y[0] - y[1]) < abs(y[0] - y[3]):
        point2 = pts[1]
    else:
        point2 = pts[3]
    dy = point2[1] - point1[1]
    dx = point2[0] - point1[0]
    return atan(dy/dx) * 180 / pi

def trueWidth(rect):
    pts = cv2.boxPoints(rect)
    x = list(map(lambda pt: pt[0], pts))
    return max(abs(x[0] - x[2]), abs(x[1] - x[3]))
    

image = cv2.imread("RocketPanelStraight48in.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

green_lower = np.array([68,0,250], dtype="uint8")
green_upper = np.array([98, 17, 255], dtype="uint8")

mask = cv2.inRange(hsv, green_lower, green_upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
#mask = cv2.erode(mask, kernel)
mask = cv2.dilate(mask, kernel)
mask = cv2.GaussianBlur(mask, (5,5), 0)

cnts= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=1)[:2]
#centerX = 0
#centerY = 0
#for c in cnts:
#    M = cv2.moments(c)
#    centerX += int(M["m10"]/M['m00'])
#    centerY += int(M['m01']/M['m00'])
#centerX = int(centerX / len(cnts))
#centerY = int(centerY / len(cnts))

rects = list(map(lambda c: cv2.minAreaRect(c), cnts))
centerX = 0
centerY = 0
for r in rects:
    centerX += r[0][0]
    centerY += r[0][1]
    print(trueHeight(r))
    print(trueAngle(r))
    print(str(trueWidth(r)) + '\n')
centerX /= len(rects)
centerY /= len(rects)

boxes = list(map(lambda r: np.int0(cv2.boxPoints(r)), rects))

cv2.drawContours(image, boxes, -1, (0,255,0), 1)
cv2.circle(image, (int(centerX), int(centerY)), 5, (0, 0, 255))

height, width, _ = image.shape
midX = width / 2
cv2.circle(image, (int(midX), int(height/2)), 5, (255,0,0))
print(centerX - midX)

cv2.imshow("Image", image)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()