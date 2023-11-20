import cv2
import matplotlib.pyplot as plt
import numpy as np

from utility import *
from imports import *
from dataloader import *

# imagePath = "antrenare/3_20.jpg"
# imagePath = "emptyBoard.jpg"
# imagePath = "decupari/1_20.jpg"
imagePath = "linii/1_19.jpg"
# imagePath = "imagini_auxiliare/01.jpg"
# imagePath = "diferente/3_15.jpg"
# imagePath = "diferente/1_17.jpg"
# imagePath = "diferente/1_01.jpg"


# imagePath = "diferente/1_01.jpg"

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
#
# find_color_values_using_trackbar(board)
# # showImage(board)
dataset = DataLoader().games

dic = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M',
       14: 'N', 15: 'O'}

cerc = cv.imread("circle.png")

for i in range(5):
    for j in range(1, 21):
        cv.imwrite(f"decupari/{i + 1}_{str(j).zfill(2)}.jpg", dataset[i][j])

        # d1 = cv.inRange(cv.cvtColor(dataset[i][j], cv.COLOR_BGR2HSV), np.array([50, 0, 215]), np.array([255, 255, 255]))
        # d2 = cv.inRange(cv.cvtColor(dataset[i][j - 1], cv.COLOR_BGR2HSV), np.array([50, 0, 215]), np.array([255, 255, 255]))


        # d1 = cv.inRange(cv.cvtColor(dataset[i][j], cv.COLOR_BGR2HSV), np.array([0, 0, 230]), np.array([255, 255, 255]))
        # d2 = cv.inRange(cv.cvtColor(dataset[i][j - 1], cv.COLOR_BGR2HSV), np.array([0, 0, 230]), np.array([255, 255, 255]))
        #
        # kernel = np.ones((5, 5), np.uint8)
        # d1 = cv.erode(d1, kernel, iterations=1)
        # d2 = cv.erode(d2, kernel, iterations=1)
        #
        # kernel = np.ones((5, 5), np.uint8)
        # d1 = cv.dilate(d1, kernel, iterations=1)
        # d2 = cv.dilate(d2, kernel, iterations=1)
        #
        # diff = d1 - d2
        # kernel = np.ones((5, 5), np.uint8)
        # diff = cv.erode(diff, kernel, iterations=1)
        #
        # kernel = np.ones((10, 10), np.uint8)
        # diff = cv.dilate(diff, kernel, iterations=1)

        d1 = cv.inRange(cv.cvtColor(dataset[i][j], cv.COLOR_BGR2HSV), np.array([50, 0, 215]), np.array([255, 255, 255]))
        d2 = cv.inRange(cv.cvtColor(dataset[i][j - 1], cv.COLOR_BGR2HSV), np.array([50, 0, 215]), np.array([255, 255, 255]))

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

        boundRectDomino = None
        maxArea = -1
        for c in contours:
            boundRect = cv.boundingRect(c)
            if boundRect[2] * boundRect[3] > maxArea:
                maxArea = boundRect[2] * boundRect[3]
                boundRectDomino = boundRect


        # daaa = cv.drawContours(dataset[i][j], contours,  -1, (0, 0, 255), 3)

        # if len(contours) > 1:
        #     plt.imshow(diff, cmap='gray')
        #     plt.show()
        #
        #     plt.imshow(daaa)
        #     plt.show()
        daaa = cv.rectangle(dataset[i][j].copy(), (int(boundRectDomino[0]), int(boundRectDomino[1])),
                            (int(boundRectDomino[0] + boundRectDomino[2]), int(boundRectDomino[1] + boundRectDomino[3])), (0, 0, 255), 2)
        img = dataset[i][j]
        piece = img[int(boundRectDomino[1]):int(boundRectDomino[1] + boundRectDomino[3]),
                int(boundRectDomino[0]):int(boundRectDomino[0] + boundRectDomino[2]), :].copy()


        # cercuri = cv.inRange(cv.cvtColor(piece, cv.COLOR_BGR2HSV), np.array([0, 0, 0]), np.array([255, 255, 90]))
        # kernel = np.ones((5, 5), np.uint8)
        # cercuri = cv.erode(cercuri, kernel, iterations=1)
        #
        # ret, cercuri = cv.threshold(cercuri, 127, 255, 0)
        #
        # contours, hierarchy = cv.findContours(cercuri, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #
        # circleContours = []
        #
        # for c in contours:
        #     approx = cv.approxPolyDP(c, 0.02 * cv.arcLength(c, True), True)
        #     aux = len(approx)
        #
        #     if len(approx) >= 7:
        #         circleContours.append(c)
        #
        #


        h, w, _ = cerc.shape
        res = cv.matchTemplate(piece, cerc, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)

        for pt in zip(*loc[::-1]):
            piece = cv.rectangle(piece, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        cv.imwrite(f"diferente/{i + 1}_{str(j).zfill(2)}.jpg", piece)

        # piece = cv.drawContours(piece, circleContours, -1, (0, 0, 255), 3)



        # circles = cv.HoughCircles(cercuri, cv.HOUGH_GRADIENT_ALT, 1, 5, param1=30, param2=0.75, minRadius=0, maxRadius=0)
        #
        # if circles is not None:
        #
        #     circles = np.uint16(np.around(circles))
        #
        #     for k in circles[0, :]:
        #         piece = cv.circle(piece, (k[0], k[1]), 2, (0, 0, 255), 4)

        # cv.imwrite(f"diferente/{i + 1}_{str(j).zfill(2)}.jpg", 255 * res)
        # cv.imwrite(f"diferente/{i + 1}_{str(j).zfill(2)}.jpg", daaa)

        # highest = None
        # max = -1
        # for ii in range(15):
        #     for jl in range(15):
        #         val = np.sum(diff[ii * 67:(ii + 1) * 67, jl * 67:(jl + 1) * 67])
        #         if val > max:
        #             max = val
        #             highest = (ii, jl)
        #
        # diff[highest[0] * 67:(highest[0] + 1) * 67, highest[1] * 67:(highest[1] + 1) * 67] = 0
        #
        # secondHighest = None
        # max = -1
        #
        # for ii in range(15):
        #     for jl in range(15):
        #         val = np.sum(diff[ii * 67:(ii + 1) * 67, jl * 67:(jl + 1) * 67])
        #         if val > max:
        #             max = val
        #             secondHighest = (ii, jl)
        #
        # highest = (highest[0] + 1, highest[1] + 1)
        # secondHighest = (secondHighest[0] + 1, secondHighest[1] + 1)
        #
        # print(highest[0], dic[highest[1]])
        # print(secondHighest[0], dic[secondHighest[1]])
        # print()
