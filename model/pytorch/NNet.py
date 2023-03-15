import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os


#####INFO
# Ce script implémente le réseau utilisé ainsi que les fonctions utiles associées
#####

# Classe utilisée pour plot les loss du réseau lors de l'entrainement
class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Fonction utilisée pour switch de joueur:
# En effet, le réseau joue toujours du point de vue du joueur 0 (plus simple de converger pour lui)
# Ainsi si le joueur actuel est 1 alors on inverse les pions
# Une transformation en tensor est également faites afin que le tableau soit lisible pour le réseau
def transform_board(boards):
    iterable = True
    if not isinstance(boards, tuple):
        iterable = False
        boards = [boards]
    boards_transformed = []
    for i in range(len(boards)):
        board_transformed = np.array(boards[i].Grid)
        board_transformed[board_transformed==1] = -1
        board_transformed[board_transformed==0] = 1

        board_transformed = board_transformed.astype('float32')
        np.nan_to_num(board_transformed, copy=False)

        if not boards[i].IsPlayerZeroTurn:
            board_transformed = -board_transformed

        boards_transformed.append(torch.tensor(board_transformed))
    
    if iterable:
        return torch.stack(boards_transformed)

    else: 
        return boards_transformed[0]
    

## Loss utilisée durant l'entraînement pour la policy (stratégie)
## L'output correspond à une Cross-entropy
def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]

## Loss utilisée durant l'entraînement pour la v value (estimation de la valeur du plateau)
## L'output correspond à une Mean square error
def loss_v(targets, outputs):
    return torch.sum((targets - outputs.squeeze()) ** 2) / targets.size()[0]
    

## Fonction d'entrainement du réseau
def train_nnet(examples, nnet, lr=0.001, epochs=10, batch_size=64):
    """
    examples: une liste d'exemples de la forme (gamesate, IsPlayerZeroTurn, pi,v)
              pi est le vecteur de stratégie (de probabilité d'action) obtenue par MCTS, v vaut +1 si
              le joueur a finalement gagné la partie, sinon -1.
    nnet: réseau de neurones utilisés
    lr : learning rate utilisé
    epochs: nombre d'epochs
    batch_size: batch size utilisé
    """
    optimizer = torch.optim.Adam(nnet.parameters(), lr=lr)

    for epoch in range(epochs):
        print('EPOCH ::: ' + str(epoch + 1))
        nnet.train()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()

        batch_count = int(len(examples) / batch_size)

        t = tqdm(range(batch_count), desc='Training Net')
        for _ in t:
            sample_ids = np.random.randint(len(examples), size=batch_size)
            boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
            boards = transform_board(boards)
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
            out_pi, out_v = nnet(boards)
            l_pi = loss_pi(target_pis, out_pi)
            l_v = loss_v(target_vs, out_v)
            total_loss = l_pi + l_v

            # print les loss
            pi_losses.update(l_pi.item(), boards.size(0))
            v_losses.update(l_v.item(), boards.size(0))
            t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

            # calcul du gradient + optimisation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    return nnet


## Fonction qui permet de faire une prédiction du réseau (policy+v value)

def predict_from_nnet(nnet, GameState):
    board = transform_board(GameState)
    nnet.eval()
    pi, v = nnet.forward(board)
    return torch.exp(pi).detach().numpy(), v.detach().numpy()

## Fonction permettant d'enregistrer les poids du réseau dans un fichier pth
def save_model(nnet, folder, filename):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        print("Checkpoint Directory does not exist! Making directory {}".format(folder))
        os.mkdir(folder)
    else:
        print("Checkpoint Directory exists! ")
    torch.save({
        'state_dict': nnet.state_dict(),
    }, filepath)

## Fonction permettant de charger les poids du réseau
def load_model(nnet, folder, filename):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise ("No model in path {}".format(filepath))
    checkpoint = torch.load(filepath)
    nnet.load_state_dict(checkpoint['state_dict'])

## Implémentation du réseau (ici une succession de Conv3d)
class NNet(nn.Module):
    def __init__(self, num_channels=50, len_board=4, dropout=0.3):
        super(NNet, self).__init__()
        self.conv1 = nn.Conv3d(1, num_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.conv6 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.conv7 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels)
        self.bn3 = nn.BatchNorm3d(num_channels)
        self.bn4 = nn.BatchNorm3d(num_channels)
        self.bn5 = nn.BatchNorm3d(num_channels)
        self.bn6 = nn.BatchNorm3d(num_channels)
        self.bn7 = nn.BatchNorm3d(num_channels)

        self.fc1 = nn.Linear((len_board**3)*num_channels, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fcpi = nn.Linear(512, len_board**2)
        self.fcv = nn.Linear(512, 1)

        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p=dropout)

        self.relu = nn.ReLU()

    def forward(self, x):

        batch = True
        if x.dim()==3:
            batch = False
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn6(x)

        x = self.conv7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn7(x)


        x = x.flatten(1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_bn1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_bn2(x)
        
        pi = self.fcpi(x)
        v = self.fcv(x)

        if not batch:
            pi = pi.squeeze()
            v = v.squeeze()
            return nn.functional.log_softmax(pi, dim=0), torch.tanh(v)
        
        return nn.functional.log_softmax(pi, dim=1), torch.tanh(v)









