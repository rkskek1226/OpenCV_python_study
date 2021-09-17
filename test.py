import sys
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

# 기본 영상 열기
# img=cv2.imread("cat.bmp",cv2.IMREAD_GRAYSCALE)
# if img is None:
#     print("Image load failed")
#     sys.exit()
#
# #cv2.imwrite("cat_gray.bmp",img)
#
# cv2.namedWindow("image")
# cv2.imshow("image",img)
#
# while True:
#     if cv2.waitKey()==ord('q'):
#         break
#
# cv2.destroyAllWindows()


# Matplotlib 라이브러리 사용
# imgBGR=cv2.imread("cat.bmp")
# img=cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#
# plt.subplot(121),plt.axis("off"),plt.imshow(img)
# plt.subplot(122),plt.axis("off"),plt.imshow(img)
# plt.show()


# 전체 화면 크기로 사진들 넘겨보기
# img_files=glob.glob("*.bmp")
# cv2.namedWindow("image",cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
#
# cnt=len(img_files)
# idx=0
# while True:
#
#     img=cv2.imread(img_files[idx])
#
#     cv2.imshow("image",img)
#     if cv2.waitKey(1500)==27:
#         break
#     idx+=1
#     if idx>=cnt:
#         idx=0


# 영상 생성
# img1=np.empty((240,320),dtype=np.uint8)
# img2=np.zeros((240,320,3),dtype=np.uint8)
# img3=np.ones((240,320,3),dtype=np.uint8)
# img4=np.full((240,320,3),100,dtype=np.uint8)


# 마스크 영상을 이용한 영상 합성
# src=cv2.imread("airplane.bmp",cv2.IMREAD_COLOR)
# mask=cv2.imread("mask_plane.bmp",cv2.IMREAD_GRAYSCALE)
# dst=cv2.imread("field.bmp",cv2.IMREAD_COLOR)
#
# cv2.copyTo(src,mask,dst)
#
# cv2.imshow("src",src)
# cv2.imshow("mask",mask)
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 도형 그리기
# img=np.full((400,400,3), 255, np.uint8)
#
# cv2.line(img,(50,50),(200,50),(0,0,255),5) # (50,50)에서 (200,50)까지 빨간색 선을 두께 5로 직선 그림
# cv2.line(img,(50,60),(150,160),(0,0,128)) # (50,60)에서 (150,160)까지 BGR(0,0,128)을 두께 1로 직선 그림
#
# cv2.rectangle(img,(50,200,150,100),(0,255,0),2) # (50,200)에서 가로 150, 세로 100를 초록색으로 두께 2로 사각형 그림
# cv2.rectangle(img,(70,220),(180,280),(0,128,0),-1) # 좌상단(70,220)에서 우하단(180,280)까지 BGR(0,128,0)으로 내부가 채워진 사각형 그림
#
# cv2.circle(img,(300,100),30,(255,255,0),-1,cv2.LINE_AA) # (300,100)에서 반지름 30인 BGR(255,255,0)을 내부가 채워진 원을 그림
# cv2.circle(img,(300,100),60,(255,0,0),3,cv2.LINE_AA) # (300,100)에서 반지름 60인 파란색 두께가 3인 원을 그림
#
# pts=np.array([[250,200],[300,200],[350,300],[250,300]])
# cv2.polylines(img,[pts],True,(255,0,255),2)
#
# text="OpenCV"+cv2.__version__
# cv2.putText(img,text,(50,350),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)
#
# cv2.imshow("img",img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 카메라, 동영상 처리
# cap=cv2.VideoCapture(0) # cap=cv2.VideoCapture()와 cap.open(0)를 한줄로 작성 가능, 카메라 처리
##cap=cv2.VideoCapture("video.avi") # 동영상 처리
# if not cap.isOpened():
#     print("camera open faile")
#     sys.exit()
#
# w=cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 프레임 width 가져오기
# h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 프레임 height 가져오기
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,320) # 프레임 width을 320으로 설정하기
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240) # 프레임 height을 240으로 설정하기
#
# while True:
#     ret, frame=cap.read() # read()는 프레임과 성공 여부를 리턴하기때문에 변수 2개를 사용
#
#     if not ret:
#         break
#
#     cv2.imshow("frame",frame)
#     if cv2.waitKey(20)==27:
#         break
#
# cap.release()


# 카메라로부터 영상을 입력받아 동영상 파일로 저장
# cap=cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera Open Fail")
#     sys.exit()
#
# w=round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h=round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps=cap.get(cv2.CAP_PROP_FPS)
# fourcc=cv2.VideoWriter_fourcc(*'DIVX')
# delay=round(1000/fps) # 프레임간의 시간 간격을 구함
#
# out=cv2.VideoWriter("output.avi",fourcc,fps,(w,h))
#
# if not out.isOpened():
#     print("File open fail")
#     cap.release()
#     sys.exit()
#
# while True:
#     ret,frame=cap.read()
#     if not ret:
#         break
#
#     out.write(frame)
#     cv2.imshow("frame",frame)
#
#     if cv2.waitKey(delay)==27:
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()


# Trackbar
# def on_level_changed(pos):
#     global img
#
#     level=pos*16
#     if level>=255:
#         level=255
#     img[:,:]=level
#     cv2.imshow("image",img)
#
# img=np.zeros((480,640),np.uint8)
#
# cv2.imshow("image",img)
# cv2.createTrackbar("level","image",0,16,on_level_changed) # 창이 열리고 실행시켜야하므로 imshow() 뒤에 작성
# cv2.waitKey()


# ROI 지정 함수(selectROI)
# img=cv2.imread("lenna.bmp")
# cv2.imshow("img",img)
#
# x,y,w,h=cv2.selectROI("img",img,False)
# roi=img[y:y+h,x:x+h]
# cv2.imshow("cropped",roi)
# cv2.moveWindow("cropped",0,0)
#
# cv2.waitKey()
# cv2.destroyAllWindows()


# 연산 시간 측정(TickMeter 클래스)
# img=cv2.imread("cat.bmp")
# if img is None:
#     print("Image load fail")
#     sys.exit()
#
# tm=cv2.TickMeter()
# tm.start() # 시작
# edge=cv2.Canny(img,50,150)
# tm.stop() # 끝
#
# ms=tm.getTimeMilli() # 밀리초 단위로 걸린 시간을 가져옴
# print(ms)


# # 영상 밝기 조절
# #src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)
# #dst=cv2.add(src,100)
# #src=cv2.imread("lenna.bmp")
# #dst=cv2.add(src,(100,100,100,0))
#
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 산술 연산
# src1=cv2.imread("lenna256.bmp",cv2.IMREAD_GRAYSCALE)
# src2=cv2.imread("square.bmp",cv2.IMREAD_GRAYSCALE)
#
# if src1 is None or src2 is None:
#     sys.exit()
#
# dst1=cv2.add(src1,src2,dtype=cv2.CV_8U)
# dst2=cv2.addWeighted(src1,0.5,src2,0.5,0.0)
# dst3=cv2.subtract(src1,src2)
# dst4=cv2.absdiff(src1,src2)   # 차이 연산(절대값을 씌움)
#
# plt.subplot(231),plt.axis("off"),plt.imshow(src1,"gray"),plt.title("src1")
# plt.subplot(232),plt.axis("off"),plt.imshow(src2,"gray"),plt.title("src2")
# plt.subplot(233),plt.axis("off"),plt.imshow(dst1,"gray"),plt.title("add")
# plt.subplot(234),plt.axis("off"),plt.imshow(dst2,"gray"),plt.title("addWeighted")
# plt.subplot(235),plt.axis("off"),plt.imshow(dst3,"gray"),plt.title("subtract")
# plt.subplot(236),plt.axis("off"),plt.imshow(dst4,"gray"),plt.title("absdiff")
# plt.show()


# 논리 연산
# img1=np.zeros((200,400),dtype=np.uint8)
# img2=np.zeros((200,400),dtype=np.uint8)
# img1[:,:200]=255
# img2[100:200,:]=255
# bitAnd=cv2.bitwise_and(img1,img2)
# bitOr=cv2.bitwise_or(img1,img2)
# bitXor=cv2.bitwise_xor(img1,img2)
# bitNot=cv2.bitwise_not(img1)


# 히스토그램
# src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)

# hist=cv2.calcHist([src],[0],None,[256],[0,256])

# plt.plot(hist)
# plt.show()
#
# src=cv2.imread("lenna.bmp")
#
# colors=['b','g','r']
# channels=cv2.split(src)
# for(ch,color) in zip(channels,colors):
#     hist=cv2.calcHist([ch],[0],None,[256],[0,256])
#     plt.plot(hist,color=color)
# plt.show()


# 영상 명암비
# src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)

# alpha=1.0   # alpha 값을 키우면 대비가 더 잘됨
# dst=np.clip((1+alpha)*src-128*alpha,0,255).astype(np.uint8)   # 명암비 조절 함수
#
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 영상 자동 명암비(노멀라이즈)
# src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)
#
# dst=cv2.normalize(src,None,0,255,cv2.NORM_MINMAX)
#
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 영상 자동 명암비(이퀄라이즈)
# src=cv2.imread("lenna.bmp",cv2.IMREAD_GRAYSCALE)
#
# if src is None:
#     sys.exit()
#
# dst=cv2.equalizeHist(src)
#
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 컬러 영상 이퀄라이즈
# src=cv2.imread("lenna.bmp")
# src_ycrcb=cv2.cvtColor(src,cv2.COLOR_BGR2YCrCb)
#
# planes=cv2.split(src_ycrcb)
# planes[0]=cv2.equalizeHist(planes[0])
# dst=cv2.merge(planes)
# dst=cv2.cvtColor(dst,cv2.COLOR_YCrCb2BGR)
#
# cv2.imshow("dst",dst)
# cv2.waitKey()


# 특정 색상 영상 추출
# src=cv2.imread("candies.png")
# src_hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
# dst1=cv2.inRange(src,(0,128,0),(100,255,100)) # BGR(0,128,0) ~ BGR(100,255,100)의 범위를 설정하고 범위안의 픽셀들을 255로 설정(아닌 픽셀들은 0으로 설정)
# dst2=cv2.inRange(src_hsv,(50,150,0),(80,255,255)) # HSV(50,150,0) ~ HSV(80,255,255)의 범위를 설정하고 범위안의 픽셀들을 255로 설정(아닌 픽셀들은 0으로 설정)
#
# plt.subplot(141),plt.axis("off"),plt.imshow(src)
# plt.subplot(142),plt.axis("off"),plt.imshow(src_hsv)
# plt.subplot(143),plt.axis("off"),plt.imshow(dst1)
# plt.subplot(144),plt.axis("off"),plt.imshow(dst2)
# plt.show()


# 차영상(absdiff)
# img1=cv2.imread("robot_arm1.jpg")
# img2=cv2.imread("robot_arm2.jpg")
# img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# diff=cv2.absdiff(img1_gray,img2_gray)
#
# ret,diff=cv2.threshold(diff,1,255,cv2.THRESH_BINARY)
# diff_red=cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR)
# diff_red[:,:,2]=0
# spot=cv2.bitwise_xor(img2,diff_red)
#
# cv2.imshow("img1",img1)
# cv2.imshow("img2",img2)
# cv2.imshow("diff",diff)
# cv2.imshow("spot",spot)
# cv2.waitKey()


# 크로마키
# img1=cv2.imread("man_chromakey.jpg")
# img2=cv2.imread("street.jpg")
#
# height1,width1=img1.shape[:2]
# height2,width2=img2.shape[:2]
# x=(width2-width1)//2
# y=height2-height1
# w=x+width1
# h=y+height1
#
# chromakey=img1[:10,:10,:]
# offset=20
#
# hsv_chroma=cv2.cvtColor(chromakey,cv2.COLOR_BGR2HSV)
# hsv_img=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
# chroma_h=hsv_chroma[:,:,0]
# lower=np.array([chroma_h.min()-offset,100,100])
# upper=np.array([chroma_h.max()+offset,255,255])
#
# mask=cv2.inRange(hsv_img,lower,upper)
# mask_inv=cv2.bitwise_not(mask)
# roi=img2[y:h,x:w]
# fg=cv2.bitwise_and(img1,img1,mask=mask_inv)
# bg=cv2.bitwise_and(roi,roi,mask=mask)
# img2[y:h,x:w]=fg+bg
#
# cv2.imshow("chromakey",img1)
# cv2.imshow("addrd",img2)
# cv2.waitKey()


# 크로마키(SeamlessClone)
# img1=cv2.imread("man_chromakey.jpg")
# img2=cv2.imread("street.jpg")
# mask=np.full_like(img1,255)
# height,width=img2.shape[:2]
# center=(width//2,height//2)
#
# normal=cv2.seamlessClone(img1,img2,mask,center,cv2.NORMAL_CLONE)
# mixed=cv2.seamlessClone(img1,img2,mask,center,cv2.MIXED_CLONE)
#
# cv2.imshow("normal",normal)
# cv2.imshow("mixes",mixed)
# cv2.waitKey()



