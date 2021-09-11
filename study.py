import cv2
import sys


cap=cv2.VideoCapture(0)
w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if cap.isOpened():
    while True:
        ret,img=cap.read()
        if ret:
            cv2.imshow("camera",img)
            if cv2.waitKey()==27:
                sys.exit()
        else:
            sys.exit()


cap.release()
cv2.destroyAllWindows()