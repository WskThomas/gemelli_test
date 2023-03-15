# Score Four 
## Abstract game
For a rough information page on the game itself, check out [its Wikipedia page](https://en.wikipedia.org/wiki/Score_Four).
## Implementation
### GameState
This file and class contains most of the game logic. Also contains the parameters for the grid size and the win condition size (if you want to play Score 5). Not intended to be modified except for these parameters.
### Players
A player is required to implement a game strategy i.e return a legal move from a given gamestate. PlayerRandom and PlayerHuman give some examples of implementation for this class.
### main.py
Define the players to use then launch as many games of Score Four as you want and check your stats from here .
For reference about 3000 games per second can be completed with PlayerRandom and a standard 4 Size.

# Résolution par AlphaZERO

J'ai utilisé une implémentation d'AlphaZero basé sur une implémentation déjà existante ici: https://github.com/suragnair/alpha-zero-general
Afin de faire fonctionner le code, il est necessaire d'utiliser Pytorch
Par manque de temps je n'ai pas optimisé le temps d'entrainement (utilisation des GPU par exemple)

## main.py
J'ai modifié le dossier main.py. On fait jouer ici ici un joueur random vs l'agent obtenu après entrainement d'AlphaZero.
Je n'effectue que 100 game car la strategie obtenue par AlphaZero est lente (~ 3 minutes pour 100 parties).

## Player.py
J'ai modifié le dossier Player.py afin d'implémenter l'agent obtenu par la méthode AlphaZero se basant sur une recherche MCTS (Monte Carlo tree search) guidé par un réseau.
J'ai également implémenté une MCTS simple qui n'utilise aucun réseau de neurone (la recherche n'est pas guidé)

## train.py
Script d'entraînement d'AlphaZero. Tous les paramètres d'entraînement sont présent dans ce script.
Ce script contient toute la pipeline d'entrainement d'AlphaZero, les commentaires détaillent les différentes étapes d'entrainement.

## test_model.py
Script permettant de tester l'entraînement d'AlphaZero contre d'autres strategies (par exemple random)

## model/MCTS.py
Implémentation de la recherche MCTS (Monte Carlo tree search), AlphaZero se base sur une recherche par arbre guidée par un réseau.
Ce script est à la fois utilisé durant l'entrainement et pour obtenir la stratégie finale

## model/pytorch/NNet.py
Implémentation du réseau et des fonctions d'entraînement associé.


