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
            # truth = f"evaluare/fisiere_solutie/331_Alexe_Bogdan/{i + 1}_{str(j + 1).zfill(2)}.txt"
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


def change_luminosity(image, factor=0.80):
    image = image.astype(np.float32)
    image = image * factor
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def augment_luminosity():
    files = os.listdir(trainPath)
    for file in tqdm(files, desc="Augmenting"):
        if file[-3:] == "jpg":
            image_before = cv.imread(os.path.join(trainPath, file))
            image = change_luminosity(image_before)
            cv.imwrite(os.path.join("custom_test_input_1", file), image)
        else:
            with open(os.path.join(trainPath, file), "rb") as f1:
                with open(os.path.join("custom_test_input_1", file), "wb") as f2:
                    f2.write(f1.read())


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