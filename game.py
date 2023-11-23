from move import *


class Game:
    scorePath = [-1, 1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3, 5, 4, 1, 6,2, 4, 5, 5, 0, 6, 3, 4, 2, 0, 1, 5, 1, 3,
                 4,
                 4, 4, 5, 0, 6, 3, 5, 4, 1, 3, 2, 0, 0, 1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2, 1, 2, 5, 0, 3, 3, 5,
                 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2, 6, 2, 3, 1, 6, 5, 6, 2, 0, 4, 0, 1, 6, 4, 4, 1, 6, 6, 3, 0]

    board = [[5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
             [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
             [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
             [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
             [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
             [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
             [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
             [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
             [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
             [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
             [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5]]

    dic = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M',
           14: 'N', 15: 'O'}

    def __init__(self, images, moves, index):
        self.images = images
        self.playerOneScore = 0
        self.playerTwoScore = 0
        self.playerOnePos = 0
        self.playerTwoPos = 0
        self.moves = moves
        self.index = index

    def play(self):
        for j in range(1, 21):
            first_pos, second_pos, one, two = Move(self.images[j - 1], self.images[j]).get_move()

            total_score_one = 0
            total_score_two = 0

            multiplier = 2 if one == two else 1

            if Game.scorePath[self.playerOnePos] in [one, two]:
                total_score_one += 3

            if Game.scorePath[self.playerTwoPos] in [one, two]:
                total_score_two += 3

            if self.moves[j - 1] == 1:
                total_score_one += multiplier * (Game.board[first_pos[0]][first_pos[1]] + Game.board[second_pos[0]][second_pos[1]])
                score_to_write = total_score_one
            else:
                total_score_two += multiplier * (Game.board[first_pos[0]][first_pos[1]] + Game.board[second_pos[0]][second_pos[1]])
                score_to_write = total_score_two

            self.playerOnePos += total_score_one
            self.playerTwoPos += total_score_two

            with open(f"solutii/{self.index}_{str(j).zfill(2)}.txt", "w") as f:
                f.write(f"{first_pos[0] + 1}{Game.dic[first_pos[1] + 1]} {one}\n")
                f.write(f"{second_pos[0] + 1}{Game.dic[second_pos[1] + 1]} {two}\n")
                f.write(f"{score_to_write}")
