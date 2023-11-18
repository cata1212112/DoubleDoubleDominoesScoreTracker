from imports import *
from constants import *
from preprocess import *

class DataLoader:
    def __init__(self, path=trainPath, numGames = 5):
        self.numGames = numGames
        self.games = []
        self.emptyBoard = cv.imread(emptyBoardPath)

        for i in range(1, self.numGames + 1):
            currentGame = [self.emptyBoard]
            for j in range(1, ROUNDS + 1):
                image = cv.imread(os.path.join(path, f"{i}_{str(j).zfill(2)}.jpg"))
                board, lines = Preprocessing(image).extractGameBoard()
                cv.imwrite(f"linii/{i}_{str(j).zfill(2)}.jpg", lines)
                currentGame.append(lines)

            self.games.append(currentGame)