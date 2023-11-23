from imports import *
from constants import *


def show_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def max_negatives_min_positives(arr):
    negatives = arr[arr < 0]
    positives = arr[arr > 0]

    return np.max(negatives), np.min(positives)


def sort_points(point_a, point_b):
    if point_a < point_b:
        return point_a, point_b
    return point_b, point_a


def check_files(num_games=5):
    for i in range(num_games):
        for j in range(20):
            truth = f"antrenare/{i + 1}_{str(j + 1).zfill(2)}.txt"
            pred = f"solutii/{i + 1}_{str(j + 1).zfill(2)}.txt"

            with open(truth, 'r') as f1, open(pred, 'r') as f2:
                lines_file1 = [f1.readline().strip() for _ in range(3)]
                lines_file2 = [f2.readline().strip() for _ in range(3)]

                assert lines_file1 == lines_file2, f"{truth}, {pred}"


def get_maximum_sum_cell(arr, cell_size):
    highest = None
    maxi = -1
    for i in range(0, ROWS * cell_size, cell_size):
        for j in range(0, COLUMNS * cell_size, cell_size):
            val = np.sum(arr[i:i + cell_size, j:j + cell_size])
            if val > maxi:
                maxi = val
                highest = (i // cell_size, j // cell_size)
    return highest

def parse_moves_file(file):
    moves = []
    with open(file, 'r') as f:
        lines = [f.readline().strip() for _ in range(20)]
        for line in lines:
            _, player = line.split()
            if player == "player1":
                moves.append(1)
            else:
                moves.append(2)
    return moves