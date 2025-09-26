import cv2
import numpy as np

img = cv2.imread("Church.png")
start_point, end_point = (0, 0), (0, 0)
COLOR = (0, 0, 255)
THICKNESS = 3
drawing = False #선을 그릴지 여부

def onMouse(event, x, y, flags, param):
    global drawing, start_point, end_point
    dst_img = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True #선을 그리기 시작
        start_point = (x, y)
        cv2.circle(dst_img, start_point, 5, (0, 0, 255), cv2.FILLED)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(dst_img, (start_point[0], start_point[1]), (end_point[0], end_point[1]), (0, 0, 255), 2)
            end_point = (x, y)
            cv2.imshow("img", dst_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        show_result()

def show_result():
    width, height = (start_point[0]-end_point[0])/2, (start_point[1]-end_point[1])/2
    width, height = np.int32(np.abs(width)), np.int32(np.abs(height))

    start_x, start_y = start_point
    end_x, end_y = end_point
    point_list = []
    if (start_x < end_x) & (start_y < end_y):
        point_list = [[start_x, start_y], [end_x, start_y],
                  [end_x, end_y], [start_x, end_y]]
    elif (start_x > end_x) & (start_y < end_y):
        point_list = [[end_x, start_y], [start_x, start_y],
                      [start_x, end_y], [end_x, end_y]]
    elif (start_x > end_x) & (start_y > end_y):
        point_list = [[end_x, end_y], [start_x, end_y],
                      [start_x, start_y], [end_x, start_y]]
    elif (start_x < end_x) & (start_y > end_y):
        point_list = [[start_x, end_y], [end_x, end_y],
                      [end_x, start_y], [start_x, start_y]]

    src = np.float32(point_list) #input 4개지점
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32) #output 4개지점
    #좌상, 우상, 우하, 좌하 시계 방향으로 4개 지점 정의

    matrix = cv2.getPerspectiveTransform(src, dst)
    new_img = cv2.warpPerspective(img, (matrix), (width, height))  # matrix 대로 변환을 함

    result1 = cv2.resize(new_img, dsize=None, fx=8, fy=8)
    mask1 = [[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]]
    kernel1 = np.array(mask1, np.float32)
    bil = cv2.bilateralFilter(new_img, 9, 30, 30)
    sharpen1 = cv2.filter2D(new_img, -1, kernel1)
    bil2 = cv2.resize(bil, dsize=None, fx=8, fy=8)
    sharpen2 = cv2.resize(sharpen1, dsize=None, fx=8, fy=8)

    result2 = sr.upsample(new_img)
    result3 = sr.upsample(sharpen1)
    result4 = sr.upsample(bil)
    result5 = cv2.hconcat((result2, result3, result4))
    before = cv2.hconcat((result1, sharpen2, bil2))

    cv2.imshow("before", before)
    cv2.imshow("result", result5)
# DNN의 Super-Resolution 모델 생성
sr = cv2.dnn_superres.DnnSuperResImpl.create()
# LapSRN 모델 8배 확대
sr.readModel('LapSRN_x8.pb')
sr.setModel("lapsrn",8)

cv2.namedWindow("img") #img란 이름의 윈도우를 먼저 만들어두는 것. 여기에 마우스 이벤트를 처리하기 위한 핸들러 작성
cv2.setMouseCallback("img", onMouse)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()