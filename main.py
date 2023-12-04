from dataloader import *
from game import *

dataset = DataLoader(num_games=1)
for i in range(1):
    images, moves = dataset[i]
    game = Game(images, moves, i+1)
    game.play()

# check_files(5)
