from imports import *
from utility import *

class Preprocessing:
    def __init__(self, image):
        self.image = image
        self.imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        self.boardMask = cv.inRange(self.imageHSV, np.array([0, 0, 0]), np.array([30, 255, 255]))

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

        dstPoints = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        M = cv.getPerspectiveTransform(srcPoints, dstPoints)

        gameBoard = cv.warpPerspective(self.image, M, (width, height))

        return gameBoard

    def getCellsCorners(self):
        board = self.extractGameBoard()

        pass