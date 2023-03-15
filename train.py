from collections import deque
from model.MCTS import MCTS
from GameState import GameState
from Game import Game
from Player import PlayerAI
import numpy as np
from tqdm import tqdm
import copy
from model.pytorch.NNet import NNet, train_nnet, save_model, load_model

#####INFO
# Il s'agit du script d'entrainement du modèle, il s'agit d'une implémentation d'AlphaZero
# Cette implémentation se base sur le code présent ici: https://github.com/suragnair/alpha-zero-general
# Je l'ai remanié lors d'un précédent projet de cours.
#####

class Coach():
    """
    Cette classe permet l'entrainement du réseaux. 
    L'entrainement se fait en 3 temps: 
        - Phase 1: Le réseau joue contre lui même afin d'évaluer les différents plateaux (self-play).
        - Phase 2: Le réseau est entrainé afin de coller au mieux a ces évaluation.
        - Phase 3: Le réseau nouvellement entrainé joue contre le réseaux entrainé précédement afin de valider (phase de validation)

    Dans les faits le réseaux seul ne choisi pas directement les actions à faire. 
    Le réseau guide une Monte Carlo tree search (MCTS) afin de choisir la meilleure action lors du self-play et lors de l'evalution du plateau.

    Il s'agit d'une recherche recursive qui permet d'obtenir une action meilleure que le réseau seul.
    L'idée de AlphaZero est de faire coller le résultat du réseau à cette exploration qui est plus efficace lors de l'entraînement.
    """

    def __init__(self, nnet):
        self.nnet = nnet    # Réseau que l'on entraine
        self.mcts = MCTS(self.nnet,1) # Il s'agit de la classe implémentant la Monte Carlo search tree

        self.numMCTSSims = 40 # Nombre d'exploration faite durant la Monte Carlo search tree
        self.cpuct = 1.     # Coefficient permettant de contrôler le rapport performance exploration/efficacité lors de la MCTS
        self.tempThreshold = 15 # Nombre d'itération après laquelle la recherche par MCTS n'est plus focalisé sur l'exloration

        self.maxlenOfQueue = 10000 # Nombre d'exemple que l'on veut garder lors de l'entraînement du réseau
        self.win_threshold = 0.55 # Proportion de victoire que
        self.numEps = 100 # nombre de de jeu contre soit même effectué (phase 1)
        self.numIters = 1000 # Nombre d'iteration d'entrainement
        self.n_self_game = 40 # Nombre de jeu contre l'ancienne version de lui-même (phase 3)

        self.lr = 0.001 # Learning rate utilisé lors de l'entraînement du réseau
        self.dropout = 0.3  # dropout utilisé lors de l'entrainement du réseau
        self.epochs = 10 # Nombre d'épisode utilisé lors de la phase d'entrainement du réseau
        self.batch_size = 64  # batch size utilisé lors de la phase d'entrainement du réseau  

        self.checkpoint_path = "model/pytorch/checkpoint"
        self.final_model_path = "model/pytorch/pretrained_model"

    def executeEpisode(self):
        """
        Cette fonction exécute un épisode de jeu contre soit-meme (self-play), en commençant par le joueur 0.
        Au fur et à mesure que le jeu se déroule, chaque tour est ajouté en tant qu'exemple d'entraînement à la variable
        trainExamples. Le jeu est joué jusqu'à ce qu'il se termine. Après la fin du jeu
        le résultat du jeu est utilisé pour attribuer des valeurs à chaque exemple dans trainExamples.

        On utilise temp=1 si episodeStep < tempThreshold, et ensuite temp=0. 
        (On passe d'un mode exploration à un mode focalisé sur la meilleure stratégie estimée)

        Output :
            trainExamples : une liste d'exemples de la forme (gamesate, IsPlayerZeroTurn, pi,v)
                           pi est le vecteur de stratégie (de probabilité d'action) obtenue par MCTS, v vaut +1 si
                           le joueur a finalement gagné la partie, sinon -1.
        """

        trainExamples = []
        gamestate = GameState()
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.tempThreshold)
            pi = self.mcts.getActionProb(gamestate, self.numMCTSSims, temp=temp) # Stratégie obtenue après une recherche MCTS (probabilité d'action)
            trainExamples.append([copy.deepcopy(gamestate), gamestate.IsPlayerZeroTurn, pi, None]) # Informations gardées afin d'entrainer le réseau
    
            action = np.random.choice(len(pi), p=pi) # On selectionne l'action à prendre en fonction de la stratégie pi

            all_moves = [(x,y) for x in range(4) for y in range(4)]
            x,y = all_moves[action]

            a = (x, y, gamestate.Grid[x][y].index(None)) # on modifie l'action en sortie du MCTS afin de la rendre compatible au gamestate

            gamestate.playLegalMove(a)
            if gamestate.checkEnd():
                r = gamestate.getWinner()
                if r is not None:
                    return [(x[0], x[2], 2*float(r==(1-x[1]))-1) for x in trainExamples]
                else: 
                    return [(x[0], x[2], 0) for x in trainExamples]
                

    def pit(self, nnet1, nnet2, n_self_game):

        """
        Cette fonction effectue plusieurs parties entre le réseau nnet1 et nnet2.
        Dans les faits on l'utilise pour la phase 3: le réseau nouvellement entrainer joue contre
        l'ancien afin d'avoir une validation des performances.
        Output :
            frac : La proportion de jeu gagné par le réseau nnet1 contre le réseau nnet2
        """

        wins = [0,0,0]
        print()
        for i in tqdm(range(n_self_game), desc="PLAY AGAINST OLDER VERSION: "):
            if i < n_self_game//2:
                game = Game(player0=PlayerAI(nnet1, self.numMCTSSims, self.cpuct), player1=PlayerAI(nnet2, self.numMCTSSims, self.cpuct), isVerbose= False, gameState=GameState())
            else:
                game = Game(player0=PlayerAI(nnet2, self.numMCTSSims, self.cpuct), player1=PlayerAI(nnet1, self.numMCTSSims, self.cpuct), isVerbose= False, gameState=GameState())
            while (not game.CurrentGameState.checkEnd()) :
                currentPlayer = game.Player0 if game.CurrentGameState.IsPlayerZeroTurn else game.Player1
                move = currentPlayer.strategy(game.CurrentGameState)
                game.CurrentGameState.playLegalMove(move)

            
            if game.CurrentGameState.getWinner() is not None:
                if i < n_self_game//2:
                    wins[game.CurrentGameState.getWinner()] += 1
                else:
                    wins[1 - game.CurrentGameState.getWinner()] += 1
            else :
                wins[2] +=1

        frac = (wins[0] + wins[2])/n_self_game
        print()
        print(f'NEW MODEL WINS:{wins[0]}, OLD MODEL WINS:{wins[1]}, DRAW:{wins[2]}')

        return frac

    def learn(self):
        """
        Cette fonction effectue l'entraînement en trois phase du réseau. A chaque itération le réseau est 
        sauvegardé à l'adresse self.checkpoint_path.
        A la fin de l'entraînement le réseau obtenu est enregistré dans self.final_model_path
        """
        iterationTrainExamples = deque([], maxlen=self.maxlenOfQueue)

        for i in range(self.numIters):
            
            # Phase 1: le réseau joue contre lui-même afin d'obtenir une estimation de différentes états du plateau
            for _ in tqdm(range(self.numEps), desc="Self Play"):
                self.mcts = MCTS(self.nnet, self.cpuct)
                iterationTrainExamples += self.executeEpisode()

            # Phase 2: On entraine le réseau à coller à ces estimations plus précises
            new_nnet = copy.deepcopy(self.nnet)
            new_nnet = train_nnet(iterationTrainExamples, new_nnet, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size)
            frac_win = self.pit(new_nnet, self.nnet, self.n_self_game)
            
            # Phase 3: On fait jouer le réseau contre l'ancienne version de lui même.
            # Si il gagne en proportion plus que l'ancien on le garde, sinon on reprends l'ancien réseau
            if frac_win >= self.win_threshold: 
                print('ACCEPTING NEW MODEL...')
                self.nnet = new_nnet
                filename =  f"checkpoint_{i}.pth"
                save_model(self.nnet, self.checkpoint_path, filename)
                load_model(self.nnet, self.checkpoint_path, filename)
            else:
                print('REJECTING NEW MODEL...')

            print()

        filename = "best_model.pth"
        save_model(self.nnet, self.final_model_path, filename)
    
def main():
    g = GameState()
    nnet = NNet(len_board=len(g.Grid))
    coach = Coach(nnet)
    coach.learn()


if __name__ == '__main__':
    main()
