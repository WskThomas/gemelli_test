from abc import ABC, abstractmethod
from GameState import GameState
import random

from model.pytorch.NNet import NNet
from model.MCTS import MCTS
import numpy as np

class Player(ABC):
    @abstractmethod
    def strategy(self, gameState : GameState) -> tuple: #return a move : a legal triplet of coordinates in the grid
        pass

class PlayerRandom(Player):
    def strategy(self, gameState: GameState) -> tuple:
        return random.choice(gameState.getPossibleMoves())
    
class PlayerHuman(Player) : 
    def strategy(self, gameState: GameState) -> tuple:
        print(f"Possible moves pick a number between 0 and {len(gameState.getPossibleMoves()) - 1} : \n {gameState.getPossibleMoves()}")
        return gameState.getPossibleMoves()[int(input())]
    

### Stratégie obtenue en effectuant une MCTS guidées par un réseau entrainé par l'algorithme AlphaZERO
class PlayerAI(Player) :
    def __init__(self, nnet : NNet, numMCTSSims=40, cpuct=1):
        self.nnet = nnet
        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        
    def strategy(self, gameState: GameState) -> tuple:
        #TODO : implement player strategy here
        self.nnet.eval()
        mcts = MCTS(self.nnet, self.cpuct)
        a = mcts.getActionProb(gameState, self.numMCTSSims, temp=0)
        len_grid = len(gameState.Grid)
        all_moves = [(x,y) for x in range(len_grid) for y in range(len_grid)]
        x,y = all_moves[np.argmax(a)]
        return (x, y, gameState.Grid[x][y].index(None))

### Stratégie obtenue en effectuant une MCTS non guidée
class PlayerAI_MCTS(Player):
    def __init__(self, numMCTSSims=40, cpuct=1):
        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        
    def strategy(self, gameState: GameState) -> tuple:
        mcts = MCTS(None, self.cpuct)
        a = mcts.getActionProb(gameState, self.numMCTSSims, temp=0)
        len_grid = len(gameState.Grid)
        all_moves = [(x,y) for x in range(len_grid) for y in range(len_grid)]
        x,y = all_moves[np.argmax(a)]
        return (x, y, gameState.Grid[x][y].index(None))

        