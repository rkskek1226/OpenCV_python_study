import numpy as np
import cv2


# 템플릿 매칭
src=cv2.imread("circuit.bmp",cv2.IMREAD_GRAYSCALE)
template=cv2.imread("crystal.bmp",cv2.IMREAD_GRAYSCALE)

result=cv2.matchTemplate(src,template,cv2.TM_CCOEFF_NORMED)
result_norm=cv2.normalize(result,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

_,_,_,maxloc=cv2.minMaxLoc(result)
th,tw=template.shape[:2]
dst=cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
cv2.rectangle(dst,maxloc,(maxloc[0]+tw,maxloc[1]+th),(0,0,255),2)

cv2.imshow("result_norm",result_norm)
cv2.imshow("dst",dst)
cv2.waitKey()