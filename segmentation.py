import sys
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import math
import random


# 외곽선 검출
# src=cv2.imread("contours.bmp",cv2.IMREAD_GRAYSCALE)
# contours,hier=cv2.findContours(src,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
# dst=cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
# idx=0
# while idx>=0:
#     c=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
#     cv2.drawContours(dst,contours,idx,c,4,cv2.LINE_8,hier)
#     idx=hier[0,idx,0]
#
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 허프 선 변환
# src=cv2.imread("sudoku.jpg")
# srcGray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# src2=src.copy()
# rows,cols=src.shape[:2]
# edges=cv2.Canny(srcGray,100,200)
# lines=cv2.HoughLines(edges,1,np.pi/180,130)
# for line in lines:
#     r,theta=line[0]
#     tx,ty=np.cos(theta),np.sin(theta)
#     x0,y0=tx*r,ty*r
#     cv2.circle(src2,(abs(x0),abs(y0)),3,(0,0,255),-1)
#     x1,y1=int(x0+cols*(-ty)),int(y0+rows*tx)
#     x2,y2=int(x0-cols*(-ty)),int(y0-rows*tx)
#     cv2.line(src2,(x1,y1),(x2,y2),(0,255,0),1)
#
# cv2.imshow("hough line",src2)
# cv2.waitKey()


# 확률적 허프 선 변환
# src=cv2.imread("sudoku.jpg",cv2.IMREAD_GRAYSCALE)
# edges=cv2.Canny(src,50,150)
# lines=cv2.HoughLinesP(edges,1.0,np.pi/180,160,minLineLength=50,maxLineGap=5)
# dst=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
#
# for line in lines:
#     x1,y1,x2,y2=line[0]
#     cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2,cv2.LINE_AA)
#
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 허프 원 변환
# src=cv2.imread("coins_connected.jpg")
# src_gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# blur=cv2.GaussianBlur(src_gray,(3,3),0) # 노이즈 제거를 위한 가우시안 블러
#
# def on_trackbar(pos):
#     rmin=cv2.getTrackbarPos("minRadius","src")
#     rmax=cv2.getTrackbarPos("maxRadius","src")
#     th=cv2.getTrackbarPos("threshold","src")
#
#     circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,50,circles=None,param1=120,param2=th,minRadius=rmin,maxRadius=rmax)
#     dst=src.copy()
#
#     if circles is not None:
#         for i in range(circles.shape[1]):
#             cx,cy,radius=circles[0][i]
#             cv2.circle(dst,(cx,cy),radius,(0,0,255),2,cv2.LINE_AA)
#
#     cv2.imshow("img",dst)
#
# circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1.1,30,None,200)
# circles=np.uint16(np.around(circles))
# for i in circles[0,:]:
#     cv2.circle(src,(i[0],i[1]),i[2],(0,0,255),2)
#
# cv2.imshow("src",src)
# cv2.createTrackbar("minRadius","src",0,100,on_trackbar)
# cv2.createTrackbar("maxRadius","src",0,150,on_trackbar)
# cv2.createTrackbar("threshold","src",0,100,on_trackbar)
# cv2.setTrackbarPos("minRadius","src",10)
# cv2.setTrackbarPos("maxRadius","src",80)
# cv2.setTrackbarPos("threshold","src",40)
# cv2.waitKey()


# 레이블링
# src=cv2.imread("keyboard.bmp",cv2.IMREAD_GRAYSCALE)
# _,src_bin=cv2.threshold(src,0,255,cv2.THRESH_OTSU)
#
# cnt,labels,stats,centroids=cv2.connectedComponentsWithStats(src_bin)
# dst=cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
#
# for i in range(1,cnt): # 0은 배경이므로 제외했음
#     x,y,w,h,a=stats[i]
#     if a<20: # 노이즈 제거용
#         continue
#     cv2.rectangle(dst,(x,y,w,h),(0,0,255),2)
#
# cv2.imshow("src_bin",src_bin)
# cv2.imshow("dst",dst)
# cv2.waitKey()













