from utility import *
from constants import *


class Move:
    circle = cv.imread("circle.png")

    def __init__(self, before, after):
        self.before = before
        self.after = after

    def extract_piece(self):
        pieces_mask_after = cv.inRange(cv.cvtColor(self.after, cv.COLOR_BGR2HSV), np.array([50, 0, 215]),
                                       np.array([255, 255, 255]))
        pieces_mask_before = cv.inRange(cv.cvtColor(self.before, cv.COLOR_BGR2HSV), np.array([50, 0, 215]),
                                        np.array([255, 255, 255]))

        piece_mask = pieces_mask_after - pieces_mask_before

        kernel = np.ones((3, 3), np.uint8)
        piece_mask = cv.erode(piece_mask, kernel, iterations=2)
        kernel = np.ones((2, 2), np.uint8)
        piece_mask = cv.erode(piece_mask, kernel, iterations=2)

        kernel = np.ones((15, 15), np.uint8)
        piece_mask = cv.dilate(piece_mask, kernel, iterations=3)

        kernel = np.ones((15, 15), np.uint8)
        piece_mask = cv.erode(piece_mask, kernel, iterations=2)

        contours, hierarchy = cv.findContours(piece_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        bound_rect_domino = None
        max_area = -1
        for c in contours:
            bound_rect = cv.boundingRect(c)
            if bound_rect[2] * bound_rect[3] > max_area:
                max_area = bound_rect[2] * bound_rect[3]
                bound_rect_domino = bound_rect

        piece = self.after[int(bound_rect_domino[1]):int(bound_rect_domino[1] + bound_rect_domino[3]),
                           int(bound_rect_domino[0]):int(bound_rect_domino[0] + bound_rect_domino[2]), :].copy()

        piece_mask = piece_mask[BORDER:-BORDER, BORDER:-BORDER]
        cell_size = piece_mask.shape[0] // ROWS

        highest = get_maximum_sum_cell(piece_mask, cell_size)

        piece_mask[highest[0] * cell_size:(highest[0] + 1) * cell_size, highest[1] * cell_size:(highest[1] + 1) * cell_size] = 0

        second_highest = get_maximum_sum_cell(piece_mask, cell_size)

        highest, second_highest = sort_points(highest, second_highest)

        return piece, highest, second_highest

    def get_move(self):
        piece, first_pos, second_pos = self.extract_piece()

        h, w, _ = Move.circle.shape
        res = cv.matchTemplate(piece, Move.circle, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        one = 0
        two = 0
        piece_h, piece_w, _ = piece.shape
        mask = np.zeros((piece_w, piece_h))

        if piece_h > piece_w:
            for pt in zip(*loc[::-1]):
                if np.sum(mask[pt[0]:pt[0] + w, pt[1]:pt[1] + h]) == 0:
                    mask[pt[0]:pt[0] + w, pt[1]:pt[1] + h] = 1
                    if pt[1] + h // 2 < piece_h // 2:
                        one += 1
                    else:
                        two += 1
        else:
            for pt in zip(*loc[::-1]):
                if np.sum(mask[pt[0]:pt[0] + w, pt[1]:pt[1] + h]) == 0:
                    mask[pt[0]:pt[0] + w, pt[1]:pt[1] + h] = 1
                    if pt[0] + w // 2 < piece_w // 2:
                        one += 1
                    else:
                        two += 1

        return first_pos, second_pos, one, two
