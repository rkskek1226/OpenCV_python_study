import sys
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

# 평균 블러링
# src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)
# kernel1=np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]) # 커널의 크기와 값 설정
# kernel2=np.ones((3,3))/3**2. # 커널 설정 다른 방법
#
# dst1=cv2.filter2D(src,-1,kernel1)
# dst2=cv2.blur(src,(3,3)) # 커널 크기가 3X3
#
# cv2.imshow("src",src)
# cv2.imshow("dst",dst1)
# cv2.imshow("dst2",dst2)
# cv2.waitKey()


# 가우시안 블러링
# src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)
# dst=cv2.GaussianBlur(src,(0,0),0)
#
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 미디언 블러링
# src=cv2.imread("salt_pepper_noise.jpg")
# dst=cv2.medianBlur(src,3)
# cv2.imshow("noise",src)
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 바이레터럴 필터(양방향 필터)
# src=cv2.imread("lenna.bmp")
# dst=cv2.bilateralFilter(src,(-1),75,75)
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 소벨 필터
# src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)
# kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32) # 소벨 커널을 생성
# gx_k=cv2.filter2D(src,-1,kernel,delta=128) # 소벨 필터 적용
#
# dx=cv2.Sobel(src,-1,1,0,ksize=3,delta=128)
# dy=cv2.Sobel(src,-1,0,1,ksize=3,delta=128)


# 캐니 엣지
src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)
dst=cv2.Canny(src,100,150)
cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey()














