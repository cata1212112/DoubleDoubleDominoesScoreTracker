import numpy as np

from imports import *
from utility import *

class Preprocessing:
    def __init__(self, image):
        self.image = image
        self.imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        self.boardMask = cv.inRange(self.imageHSV, np.array([0, 0, 0]), np.array([30, 255, 255]))


    def cleanLines(self, lines):
        pass

    def extractGameBoard(self):

        self.boardMask = (255 - self.boardMask)

        kernel = np.ones((10, 10), np.uint8)
        self.boardMask = cv.erode(self.boardMask, kernel, iterations=1)

        kernel = np.ones((15, 15), np.uint8)
        self.boardMask = cv.dilate(self.boardMask, kernel, iterations=1)
        contours, hierarchy = cv.findContours(self.boardMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        maximumPerimeter = -1
        board = None

        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4 and perimeter > maximumPerimeter:
                maximumPerimeter = perimeter
                board = approx


        board = cv.convexHull(board)
        upperLeftCornerIndex = np.argmin(np.array([b[0][0] ** 2 + b[0][1] ** 2 for b in board]))
        board = np.concatenate((board[upperLeftCornerIndex:], board[:upperLeftCornerIndex]))

        srcPoints = np.float32([b[0] for b in board])

        width = max(int(np.round(
            np.linalg.norm(np.array([[board[0][0][0] - board[1][0][0], board[0][0][1] - board[1][0][1]]])))),
                    int(np.round(np.linalg.norm(
                        np.array([[board[2][0][0] - board[3][0][0], board[2][0][1] - board[3][0][1]]])))))
        height = max(int(np.round(
            np.linalg.norm(np.array([[board[0][0][0] - board[3][0][0], board[0][0][1] - board[3][0][1]]])))),
                     int(np.round(np.linalg.norm(
                         np.array([[board[1][0][0] - board[2][0][0], board[1][0][1] - board[2][0][1]]])))))

        dstPoints = np.float32([[0, 0], [2000, 0], [2000, 2000], [0, 2000]])

        M = cv.getPerspectiveTransform(srcPoints, dstPoints)

        gameBoard = cv.warpPerspective(self.image, M, (2000, 2000))

        self.centerSquareMask = cv.inRange(cv.cvtColor(gameBoard, cv.COLOR_BGR2HSV), np.array([40, 0, 0]), np.array([130, 255, 255]))
        imgCopy = gameBoard.copy()
        edges = cv.Canny(self.centerSquareMask, threshold1=50, threshold2=100)
        lines = cv.HoughLines(edges, 1, 1 * np.pi / 180, threshold=250)

        liniiVerticale = []
        liniiOrizontale = []

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 10000 * (-b))
            y1 = int(y0 + 10000 * (a))
            x2 = int(x0 - 10000 * (-b))
            y2 = int(y0 - 10000 * (a))


            if theta < 1:
                # verticale
                liniiVerticale.append(x0)
                color = (255, 0, 0)
            else:
                # orizontale
                liniiOrizontale.append(y0)
                color = (0, 0, 255)

        liniiVerticale = np.array(liniiVerticale)
        liniiOrizontale = np.array(liniiOrizontale)

        h, w = gameBoard.shape[:2]

        liniiVerticale = liniiVerticale - w // 2
        liniiOrizontale = liniiOrizontale - h // 2

        left, right = maxNegativesMinPositives(liniiVerticale)
        up, down = maxNegativesMinPositives(liniiOrizontale)

        # print(left + w // 2, right + w // 2, up + h // 2, down + h // 2)

        left += w // 2
        right += w // 2
        up += h // 2
        down += h // 2

        left = int(left)
        right = int(right)
        up = int(up)
        down = int(down)

        # imgCopy = cv.line(imgCopy, (left, 0), (left, 2000), (0, 0, 255), 3)
        # imgCopy = cv.line(imgCopy, (right, 0), (right, 2000), (0, 0, 255), 3)
        # imgCopy = cv.line(imgCopy, (0, up), (2000, up), (0, 0, 255), 3)
        # imgCopy = cv.line(imgCopy, (0, down), (2000, down), (0, 0, 255), 3)


        srcPoints = np.float32([[left, up], [right, up], [right, down], [left, down]])
        dstPoints = np.float32([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])

        M = cv.getPerspectiveTransform(srcPoints, dstPoints)

        imgCopy = cv.warpPerspective(imgCopy, M, (1000, 1000))
        # cercuri = cv.inRange(cv.cvtColor(imgCopy, cv.COLOR_BGR2HSV), np.array([0, 0, 220]), np.array([255, 255, 255]))
        #
        # circles = cv.HoughCircles(cercuri, cv.HOUGH_GRADIENT, 1, 1, param1=40, param2=15, minRadius=0, maxRadius=30)
        # circles = np.uint16(np.around(circles))
        # for i in circles[0, :]:
        #     imgCopy = cv.circle(imgCopy, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #     # draw the center of the circle


        return gameBoard, imgCopy

