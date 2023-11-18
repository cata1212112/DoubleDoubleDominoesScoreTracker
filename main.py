import cv2

from utility import *
from imports import *
from dataloader import *

# imagePath = "antrenare/3_20.jpg"
# imagePath = "emptyBoard.jpg"
# imagePath = "decupari/1_20.jpg"
imagePath = "linii/1_19.jpg"
# imagePath = "imagini_auxiliare/01.jpg"

def find_color_values_using_trackbar(frame):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    def nothing(x):
        pass

    cv.namedWindow("Trackbar")
    cv.createTrackbar("LH", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LS", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LV", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("UH", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("US", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("UV", "Trackbar", 255, 255, nothing)

    while True:

        l_h = cv.getTrackbarPos("LH", "Trackbar")
        l_s = cv.getTrackbarPos("LS", "Trackbar")
        l_v = cv.getTrackbarPos("LV", "Trackbar")
        u_h = cv.getTrackbarPos("UH", "Trackbar")
        u_s = cv.getTrackbarPos("US", "Trackbar")
        u_v = cv.getTrackbarPos("UV", "Trackbar")

        l = np.array([l_h, l_s, l_v])
        u = np.array([u_h, u_s, u_v])
        mask_table_hsv = cv.inRange(frame_hsv, l, u)

        frame = cv.resize(frame, (400, 400))
        mask_table_hsv = cv.resize(mask_table_hsv, (400, 400))


        res = cv.bitwise_and(frame, frame, mask=mask_table_hsv)
        cv.imshow("Frame", frame)
        cv.imshow("Mask", mask_table_hsv)
        cv.imshow("Res", res)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

board = cv.imread(imagePath)
# board, patrat = Preprocessing(board).extractGameBoard()
# find_color_values_using_trackbar(board)
# cv2.imwrite("emptyBoard.jpg", patrat)

# find_color_values_using_trackbar(board)
# showImage(board)
dataset = DataLoader().games

for i in range(5):
    for j in range(1,21):
        cv.imwrite(f"decupari/{i + 1}_{str(j).zfill(2)}.jpg", dataset[i][j])

        d1 = cv.inRange(cv.cvtColor(dataset[i][j], cv.COLOR_BGR2HSV), np.array([50, 0, 215]), np.array([255, 255, 255]))
        d2 = cv.inRange(cv.cvtColor(dataset[i][j-1], cv.COLOR_BGR2HSV), np.array([50, 0, 215]), np.array([255, 255, 255]))

        diff = d1 - d2

        kernel = np.ones((3, 3), np.uint8)
        diff = cv.erode(diff, kernel, iterations=2)
        kernel = np.ones((2, 2), np.uint8)
        diff = cv.erode(diff, kernel, iterations=2)


        kernel = np.ones((15, 15), np.uint8)
        diff = cv.dilate(diff, kernel, iterations=3)

        kernel = np.ones((15, 15), np.uint8)
        diff = cv.erode(diff, kernel, iterations=2)

        contours, hierarchy = cv.findContours(diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        boundRect = cv.boundingRect(contours[0])
        # daaa = cv.drawContours(dataset[i][j], contours,  -1, (0, 0, 255), 3)
        daaa = cv.rectangle(dataset[i][j], (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0, 0, 255), 2)
        cv.imwrite(f"diferente/{i + 1}_{str(j).zfill(2)}.jpg", daaa)

