import sys
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

# 영상 이동
# src=cv2.imread("lenna.bmp")
# rows,cols=src.shape[:2]
# aff=np.array([[1,0,200],[0,1,100]],dtype=np.float32)
# dst1=cv2.warpAffine(src,aff,(cols+200,rows+100))
# dst2=cv2.warpAffine(src,aff,(0,0))
# dst3=cv2.warpAffine(src,aff,(cols+200,rows+100),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,0,0))


# 영상 확대, 축소
# src=cv2.imread("lenna.bmp")
# rows,cols=src.shape[0:2]
# m_small=np.array([[0.5,0,0],[0,0.5,0]])
# m_big=np.float32([[2,0,0],[0,2,0]])
# dst1=cv2.warpAffine(src,m_small,(int(rows*0.5),int(cols*0.6)))
# dst2=cv2.warpAffine(src,m_big,(int(rows*2),int(cols*2)),None,cv2.INTER_CUBIC)
# dst3=cv2.resize(src,(0,0),fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
# dst4=cv2.resize(src,(1536,1536))
