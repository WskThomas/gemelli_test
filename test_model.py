from Player import PlayerAI, PlayerRandom, PlayerAI_MCTS
import os
from pathlib import Path
from model.pytorch.NNet import NNet, load_model
from tqdm import tqdm
from Game import Game
from GameState import GameState
import matplotlib.pyplot as plt


### Script permettant de tester les performances du modèle lors de l'entraînement (voir script train.py)

def test_training(training_path, simple_model, numberOfGames, numMCTSSims=100, name_figure_file="analyse.png"):
    """
    Cette fonction effectue un test de réseau entrainé par la méthode AlphaZERO.
    Créer un fichier png contenant les informations de ce test.

    Input:
        training_path: (str) path contenant les poids des différents modèles entrainés (voir script train.py)
        simple_model: modèle contre lequel les différents modèles entrainés jouent 
        numberOfGames: nombre de simulation MCTS utilisé
        name_figure_file: nom de la figure contenant toutes les informations de test
    """
    models_path = sorted(Path(training_path).iterdir(), key=os.path.getmtime)
    frac_win = [0.]* len(models_path)
    gameLengths = [0.] * len(models_path)
    for i in tqdm(range(len(models_path))):
        nnet = NNet()
        file_name = os.path.basename(os.path.normpath(models_path[i]))
        load_model(nnet, training_path, file_name)   
        model_AI = PlayerAI(nnet, numMCTSSims=numMCTSSims, cpuct=1.)

        for j in range(numberOfGames):
            if j < numberOfGames//2:
                game = Game(player0= model_AI, player1=simple_model, isVerbose= False, gameState=GameState())
            else:
                game = Game(player0= simple_model, player1=model_AI, isVerbose= False, gameState=GameState())
            game.run()
            gameLengths[i] += game.CurrentGameState.MoveCount/numberOfGames
            if j < numberOfGames//2:
                if game.CurrentGameState.getWinner()==0:
                   frac_win[i] += 1/numberOfGames

            else:
                 if game.CurrentGameState.getWinner()==1:
                     frac_win[i] += 1/numberOfGames

    fig, axs = plt.subplots(2)
    fig.suptitle('Pourcentage de partie gagnés et nombre de coups par parties')
    axs[0].plot(frac_win)
    axs[1].plot(gameLengths)

    fig.savefig(name_figure_file, dpi=fig.dpi)


numMCTSSims = 5
numberOfGames = 100

training_path = "model/pytorch/checkpoint"
# simple_model = PlayerRandom()
simple_model = PlayerAI_MCTS(numMCTSSims)

test_training(training_path, simple_model, numberOfGames, numMCTSSims=numMCTSSims)

