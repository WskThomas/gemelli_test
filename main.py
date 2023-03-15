from Game import Game
from GameState import GameState
from Player import PlayerRandom, PlayerHuman, PlayerAI, PlayerAI_MCTS
from model.pytorch.NNet import NNet, load_model
from tqdm import tqdm

# Create the players, the class defines the strategy

# Load the model
nnet = NNet()
load_model(nnet, "model/pytorch/best_model", "best_model_1.pth")
numMCTSSims = 100
player0 = PlayerAI(nnet, numMCTSSims=numMCTSSims, cpuct=1.)
player1 = PlayerRandom()
# player1 = PlayerAI_MCTS(numMCTSSims=numMCTSSims)
# use PlayerAI for your own implementation (for player0 or player1)
#player1 = PlayerAI()


numberOfGames = 100
gameLengths = [None] * numberOfGames
winners = [None] * numberOfGames

for i in tqdm(range(numberOfGames)):
    game = Game(player0= player0, player1= player1, isVerbose= False, gameState=GameState())
    game.run()
    #basic statistic collection on the game once it's ended
    gameLengths[i] = game.CurrentGameState.MoveCount
    winners[i] = game.CurrentGameState.getWinner()

print(f"Average game length : {sum(gameLengths)/len(gameLengths)} moves")
print(f"Player 0 won {len([x for x in winners if x == 0])} \nPlayer 1 won {len([x for x in winners if x == 1])}")
