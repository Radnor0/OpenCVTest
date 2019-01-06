import numpy as np
import cv2

vid = cv2.VideoCapture("cubevideo.avi")

while True :
    success, image = vid.read()

    if not success :
        break

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([22,78,90], dtype="uint8")
    yellow_upper = np.array([32, 255, 255], dtype="uint8")

    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    #mask = cv2.erode(mask, kernel)
    #mask = cv2.dilate(mask, kernel)
    mask = cv2.GaussianBlur(mask, (3,3), 0)

    cv2.imshow("Image", image)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()