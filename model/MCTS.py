import numpy as np
import copy
from model.pytorch.NNet import predict_from_nnet

#####INFO
# Ce script implémente la Monte Carlo tree search et les fonction utiles associées
#####

def stringRepresentation(gamestate):
    # Cette fonction renvoie une representation en string du plateau.
    # Elle est utilisé dans la Monte Carlo tree search afin de sauvagarder les états lors de la recherche
    board_transformed = np.array(gamestate.Grid)
    board_transformed[board_transformed==1] = -1
    board_transformed[board_transformed==0] = 1
    
    board_transformed = board_transformed.astype('float32')
    np.nan_to_num(board_transformed, copy=False)

    if not gamestate.IsPlayerZeroTurn:
        board_transformed = -board_transformed

    return str(board_transformed.astype('int'))
    

def get_valid_move_mask(GameState):
    # Renvoie un mask des coups possible
    # Cette fonction est utile car l'output du réseau est de longeur fixé 4x4, on applique donc un masque a cet output
    possible_moves_mask = []
    len_grid = len(GameState.Grid)
    for x in range(len_grid):
        for y in range(len_grid):
            if None in GameState.Grid[x][y]: possible_moves_mask.append(1)
            else: possible_moves_mask.append(0)
    return possible_moves_mask


class MCTS():
    """
    Cette classe gère l'arbre MCTS.
    """

    def __init__(self, nnet, cpuct):
        self.nnet = nnet # Réseau de neurone guidant la recherche MCTS
        self.cpuct = cpuct # Coefficient permettant de contrôler le rapport performance exploration/efficacité lors de la MCTS

        self.Qsa = {}  # Stocke la Q value lors de la recherche
        self.Nsa = {}  # stocke le nombre de fois où l'arête s,a (état, action) a été visitée
        self.Ns = {}  # stocke le nombre de fois où la feuille s (état) a été visitée
        self.Ps = {}  # stocke la politique initiale (obtenue grâce au réseau de neurone)

        self.Vs = {}  # stocke les mouvements valides pour un plateau s
        

    def getActionProb(self, GameState, numMCTSSims, temp=1):
        """
        Cette fonction effectue un nombre numMCTSSims de simulation MCTS à 
        partir d'un plateau donné GameState

        Output :
            probs : un vecteur de stratégie où la probabilité de la ième action est
                   proportionnelle à Nsa[(s,a)]**(1./temp)
        """
        for i in range(numMCTSSims):
            GameState_copy = copy.deepcopy(GameState)
            self.search(GameState_copy, dir_noise=False)
        s = stringRepresentation(GameState)

        len_grid = len(GameState.Grid)

        counts = [self.Nsa[(s, i)] if (s, i) in self.Nsa else 0 for i in range(len_grid*len_grid)]

        ## Si temp==0 alors on n'est pas dans un régime de recherche
        ## La probabilité d'action correspondà l'argmax de la policy
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        ## Si temp==1 alors on est dans un régime de recherche: la politique renvoyée est égale à:
        ## Nsa[(s,a)]**(1./temp)

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, GameState, dir_noise=False, alpha_dir_noise=0.7):
        """
        Cette fonction effectue une itération de MCTS. Elle est appelée récursivement jusqu'à ce qu'un noeud feuille soit trouvé.
        L'action choisie pour chaque noeud maximise l'UCB (upper confidence bound sur les Q-values) : https://web.stanford.edu/~surag/posts/alphazero.html.
        Une fois que l'on tombe sur une feuille est trouvé, on estime P et v (policy et v value) grâce au réseau.
        Si l'on tombe sur une fin de jeu alors v prends la valeur 1 si il y'a une victioire 0 si il y'a nul et -1 si il y'a une défaite.
        Suite à cela v est propagé le long de l'arbre, vers les résultats entérieurs et les valeurs de Ns, Nsa, Qsa sont
        mises à jour.

        REMARQUE : l'ouput de la fonction search est égale à -v
        On effectue cela car la valeur v corresponds à la valeur du plateau pour le joueur actuel[-1,1],
        la valeur pour le joeur précédent est égale à -v
        Output :
            -v : la valeur négative de l'état s (le plateau s) actuel
        """
        s = stringRepresentation(GameState)

        ## On check si le jeu est fini et on propage le long de l'arbre la v valeur correspondante
        if GameState.checkEnd():
            if GameState.getWinner() is None:
                return 0
            else:
                player = 1 - GameState.IsPlayerZeroTurn
                winner = GameState.getWinner()
                if player!=winner:
                    return 1
                else:
                    return -1
                
        ## Cet état n'a jamais été rencontré lors de la recherche, on utilise donc le réseau pour l'estimer. On stoppe également la recherche
        ## On propage le long de l'arbre -v où v est la valeur estimée par le réseau
        if s not in self.Ps:
            if self.nnet is not None: self.Ps[s], v = predict_from_nnet(self.nnet, GameState)
            else: self.Ps[s], v = np.ones((len(GameState.Grid)**2)), 0
            if dir_noise: self.Ps[s] = (0.75 * self.Ps[s]) + (0.25 * np.random.dirichlet([alpha_dir_noise]) * len(self.Ps[s]))
            valid_move = get_valid_move_mask(GameState)
            
            self.Ps[s] = self.Ps[s] * valid_move
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                print("The network prediction corresponds to an impossible movement, the network structure is probably not good enough / or overfitt")
                self.Ps[s] = self.Ps[s] + valid_move
                self.Ps[s] /= np.sum(self.Ps[s])
        
            self.Vs[s] = valid_move
            self.Ns[s] = 0
            return -v
    
        valid_move = self.Vs[s]

        ## Cet état a déjà été vu lors de la recherche et n'est pas terminal: on continue de chercher le long de l'arbre.
        ## L'action choisie est celle qui maximise l'UCB (upper confidence bound sur les Q-values): https://web.stanford.edu/~surag/posts/alphazero.html

        max_u, best_a, best_a_index = -float("inf"), -1, -1

        len_grid = len(GameState.Grid)
        all_moves = [(x,y) for x in range(len_grid) for y in range(len_grid)]

        for a_idx, a in enumerate(all_moves):
            if valid_move[a_idx]==1:
                if (s, a_idx) in self.Qsa:
                    u = self.Qsa[(s, a_idx)] + self.cpuct * self.Ps[s][a_idx] * np.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a_idx)])
                else:
                    u = self.cpuct * self.Ps[s][a_idx] * np.sqrt(self.Ns[s] + 1e-8) 

                if u>max_u:
                    max_u = u
                    best_a = a
                    best_a_index = a_idx


        a = best_a
        a_idx = best_a_index

        action_to_game = (a[0], a[1], GameState.Grid[a[0]][a[1]].index(None))
        GameState.playLegalMove(action_to_game)

        ## On continue de chercher le long de l'arbre

        v = self.search(GameState)

        ## On mets a jour après la rechreche les Q-values correspondantes
        ## On renvoie la nouvelle estimation de v après la recherche

        if (s, a_idx) in self.Qsa:
            self.Qsa[(s, a_idx)] = (self.Nsa[(s, a_idx)] * self.Qsa[(s, a_idx)] + v) / (self.Nsa[(s, a_idx)] + 1)
            self.Nsa[(s, a_idx)] += 1

        else:
            self.Qsa[(s, a_idx)] = v
            self.Nsa[(s, a_idx)] = 1

        self.Ns[s] += 1

        return -v
