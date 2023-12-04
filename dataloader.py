from preprocess import *
from utility import *


class DataLoader:
    def __init__(self, path=TEST_PATH, num_games=5):
        self.numGames = num_games
        self.games = []
        self.emptyBoard = cv.imread(emptyBoardPath)
        self.gameMoves = []

        for i in range(0, self.numGames):
            current_game = [self.emptyBoard]
            self.gameMoves.append(parse_moves_file(os.path.join(path, f"{i + 1}_mutari.txt")))
            for j in tqdm(range(0, ROUNDS), desc=f"Preprocessing round {i+1}"):
                image = cv.imread(os.path.join(path, f"{i + 1}_{str(j + 1).zfill(2)}.jpg"))
                game = Preprocessing(image, i+1, j+1).extract_game()
                current_game.append(game)

            self.games.append(current_game)

    def __getitem__(self, item):
        return self.games[item], self.gameMoves[item]
