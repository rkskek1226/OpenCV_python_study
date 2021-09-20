import sys
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import math

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


# 영상 회전
# src=cv2.imread("lenna.bmp")
# rows,cols=src.shape[:2]
# rad=20*math.pi/180
# aff=np.array([[math.cos(rad),math.sin(rad),0],[-math.sin(rad),math.cos(rad),0]],dtype=np.float32)
# dst1=cv2.warpAffine(src,aff,(0,0))
# mtrx=cv2.getRotationMatrix2D((cols/2,rows/2),30,0.5) # Affine 변환 행렬을 리턴
# dst2=cv2.warpAffine(src,mtrx,(0,0))


# 어파인 변환
# src=cv2.imread("lenna.bmp")
# rows,cols=src.shape[:2]
# pts1=np.float32([[100,50],[200,50],[100,200]])
# pts2=np.float32([[80,70],[210,60],[250,120]])
# mtrx=cv2.getAffineTransform(pts1,pts2) # 변환 행렬을 리턴
# dst=cv2.warpAffine(src,mtrx,(0,0))


# 원근 변환(투시 변환)
# src=cv2.imread("lenna.bmp")
# rows,cols=src.shape[:2]
# pts1=np.float32([[0,0],[0,rows],[cols,0],[cols,rows]])
# pts2=np.float32([[100,50],[10,rows-50],[cols-50,10],[cols-100,rows-100]])
# mtrx=cv2.getPerspectiveTransform(pts1,pts2)
# dst=cv2.warpPerspective(src,mtrx,(0,0))

# 원근 변환 활용1
# src=cv2.imread("paper.jpg")
# win_name="scanning"
# rows,cols=src.shape[:2]
# draw=src.copy()
# pts_cnt=0
# pts=np.zeros((4,2),dtype=np.float32)
#
# def onMouse(event,x,y,flags,params):
#     global pts_cnt
#     if event==cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(draw,(x,y),10,(0,255,0),-1)
#         cv2.imshow(win_name,draw)
#
#         pts[pts_cnt]=[x,y]
#         pts_cnt+=1
#         if pts_cnt==4:
#             sm=pts.sum(axis=1)
#             diff=np.diff(pts,axis=1)
#
#             topLeft=pts[np.argmin(sm)] # x+y가 가장 작으므로 좌상단 좌표
#             bottomRight=pts[np.argmax(sm)] # x+y가 가장 커서 우하단 좌표
#             topRight=pts[np.argmin(diff)] # x-y가 가장 작으므로 우상단 좌표
#             bottomLeft=pts[np.argmax(diff)] # x-y가 가장 커서 좌하단 좌표
#
#             pts1=np.float32([topLeft,topRight,bottomRight,bottomLeft])
#             w1=abs(bottomRight[0]-bottomLeft[0])
#             w2=abs(topRight[0]-topLeft[0])
#             h1=abs(bottomRight[1]-topRight[1])
#             h2=abs(bottomLeft[1]-topLeft[1])
#             width=max([w1,w2])
#             height=max([h1,h2])
#             pts2=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
#             mtrx=cv2.getPerspectiveTransform(pts1,pts2)
#             result=cv2.warpPerspective(src,mtrx,(width,height))
#             cv2.imshow("scanned",result)
#
# cv2.imshow(win_name,src)
# cv2.setMouseCallback(win_name,onMouse)
# cv2.waitKey()


# 원근 변환 활용2
# def drawROI(img,corners):
#     cpy=img.copy()
#     c1=(192,192,255)
#     c2=(128,128,255)
#
#     for pt in corners:
#         cv2.circle(cpy,tuple(pt.astype(int)),25,c1,-1,cv2.LINE_AA)
#
#     cv2.line(cpy,tuple(corners[0].astype(int)),tuple(corners[1].astype(int)),c2,2,cv2.LINE_AA)
#     cv2.line(cpy,tuple(corners[1].astype(int)),tuple(corners[2].astype(int)),c2,2,cv2.LINE_AA)
#     cv2.line(cpy,tuple(corners[2].astype(int)),tuple(corners[3].astype(int)),c2,2,cv2.LINE_AA)
#     cv2.line(cpy,tuple(corners[3].astype(int)),tuple(corners[0].astype(int)),c2,2,cv2.LINE_AA)
#
#     disp=cv2.addWeighted(img,0.3,cpy,0.7,0)
#     return disp
#
# def onMouse(event,x,y,flags,params):
#     global srcQuad,dragSrc,ptOld,src
#
#     if event==cv2.EVENT_LBUTTONDOWN:
#         for i in range(4):
#             if cv2.norm(srcQuad[i]-(x,y))<25:
#                 dragSrc[i]=True
#                 ptOld=(x,y)
#                 break
#
#     if event==cv2.EVENT_LBUTTONUP:
#         for i in range(4):
#             dragSrc[i]=False
#
#     if event==cv2.EVENT_MOUSEMOVE:
#         for i in range(4):
#             if dragSrc[i]:
#                 dx=x-ptOld[0]
#                 dy=y-ptOld[1]
#                 srcQuad[i]+=(dx,dy)
#                 cpy=drawROI(src,srcQuad)
#                 cv2.imshow("img",cpy)
#                 ptOld=(x,y)
#                 break
#
# src=cv2.imread("paper.jpg")
# rows,cols=src.shape[:2]
# width=500 # 출력 영상 width 지정
# height=round(width*297/210) # 출력 영상 height 지정
# srcQuad=np.array([[30,30],[30,rows-30],[cols-30,rows-30],[cols-30,30]],np.float32)
# dstQuad=np.array([[0,0],[0,height-1],[width-1,height-1],[width-1,0]],np.float32)
# dragSrc=[False,False,False,False]
#
# disp=drawROI(src,srcQuad)
# cv2.imshow("img",disp)
# cv2.setMouseCallback("img",onMouse)
#
# while True:
#     key=cv2.waitKey()
#     if key==13:
#         break
#     elif key==27:
#         cv2.destroyWindow("img")
#         sys.exit()
#
# pers=cv2.getPerspectiveTransform(srcQuad,dstQuad)
# dst=cv2.warpPerspective(src,pers,(width,height),flags=cv2.INTER_CUBIC)
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 리매핑
# src=cv2.imread("lenna.bmp")
# rows,cols=src.shape[:2]
# mapy,mapx=np.indices((rows,cols),dtype=np.float32)
# mapx=cols-mapx-1
# mapy=rows-mapy-1
# result=cv2.remap(src,mapx,mapy,cv2.INTER_LINEAR)
# cv2.imshow("src",src)
# cv2.imshow("dst",result)
# cv2.waitKey()














