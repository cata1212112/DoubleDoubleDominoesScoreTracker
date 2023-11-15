from imports import *
from constants import *

class DataLoader:
    def __init__(self, path=trainPath, numGames = 5):
        self.numGames = numGames
        self.games = []

        for i in range(1, self.numGames + 1):
            currentGame = []
            for j in range(1, ROUNDS + 1):
                image = cv.imread(os.path.join(path, f"{i}_{str(j).zfill(2)}.jpg"))
                currentGame.append(image)

            self.games.append(currentGame)