import cv2

from utility import *
from imports import *
from preprocess import *
from dataloader import *

imagePath = "antrenare/3_20.jpg"
# imagePath = "emptyBoard.jpg"

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
# board = Preprocessing(board).extractGameBoard()
# cv2.imwrite("emptyBoard.jpg", board)

# find_color_values_using_trackbar(board)
# showImage(board)
dataset = DataLoader().games

for i in range(5):
    for j in range(20):
        board = Preprocessing(dataset[i][j]).extractGameBoard()
        cv.imwrite(f"decupari/{i + 1}_{str(j + 1).zfill(2)}.jpg", board)

        h, w, ch = board.shape
