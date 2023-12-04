from utility import *


class Preprocessing:
    def __init__(self, image, i, j):
        self.image = image
        self.imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        self.i = i
        self.j = j

    def extract_outer_square(self):
        outer_square_mask = 255 - cv.inRange(self.imageHSV, np.array([0, 0, 0]), np.array([30, 255, 255]))

        # if SAVE:
        #     cv.imwrite(f"masti/{self.i}_{str(self.j).zfill(2)}.jpg", outer_square_mask)

        kernel = np.ones((10, 10), np.uint8)
        outer_square_mask = cv.erode(outer_square_mask, kernel, iterations=1)

        kernel = np.ones((15, 15), np.uint8)
        outer_square_mask = cv.dilate(outer_square_mask, kernel, iterations=1)

        if SAVE:
            cv.imwrite(f"masti/{self.i}_{str(self.j).zfill(2)}.jpg", outer_square_mask)

        contours, hierarchy = cv.findContours(outer_square_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        maximum_perimeter = -1
        board = None

        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4 and perimeter > maximum_perimeter:
                maximum_perimeter = perimeter
                board = approx

        board = cv.convexHull(board)
        upper_left_corner_index = np.argmin(np.array([b[0][0] ** 2 + b[0][1] ** 2 for b in board]))
        board = np.concatenate((board[upper_left_corner_index:], board[:upper_left_corner_index]))

        src_points = np.float32([b[0] for b in board])
        dst_points = np.float32([[0, 0], [2000, 0], [2000, 2000], [0, 2000]])

        m = cv.getPerspectiveTransform(src_points, dst_points)

        outer_square = cv.warpPerspective(self.image, m, (2000, 2000))

        return outer_square

    def extract_inner_square(self):
        outer_square = self.extract_outer_square()

        inner_square_mask = cv.inRange(cv.cvtColor(outer_square, cv.COLOR_BGR2HSV), np.array([40, 0, 0]),
                                       np.array([130, 255, 255]))

        # if SAVE:
        #     cv.imwrite(f"masti/{self.i}_{str(self.j).zfill(2)}.jpg", inner_square_mask)

        edges = cv.Canny(inner_square_mask, threshold1=50, threshold2=100)
        lines = cv.HoughLines(edges, 1, 1 * np.pi / 180, threshold=250)

        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            if theta < 1:
                vertical_lines.append(x0)
            else:
                horizontal_lines.append(y0)

        vertical_lines = np.array(vertical_lines)
        horizontal_lines = np.array(horizontal_lines)

        h, w = outer_square.shape[:2]

        vertical_lines = vertical_lines - w // 2
        horizontal_lines = horizontal_lines - h // 2

        left, right = max_negatives_min_positives(vertical_lines)
        up, down = max_negatives_min_positives(horizontal_lines)

        left += w // 2
        right += w // 2
        up += h // 2
        down += h // 2

        left = int(left) - 15
        right = int(right) + 15
        up = int(up) - 15
        down = int(down) + 15

        src_points = np.float32([[left, up], [right, up], [right, down], [left, down]])
        dst_points = np.float32([[0, 0], [1005, 0], [1005, 1005], [0, 1005]])

        m = cv.getPerspectiveTransform(src_points, dst_points)

        inner_square = cv.warpPerspective(outer_square, m, (1005, 1005))

        return inner_square

    def extract_game(self):
        return self.extract_inner_square()
