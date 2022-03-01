import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset

class MLP(nn.Module):
    def __init__(self, neurons_per_layer, dropout_rate = 0.1, learning_rate = 1e-3, weight_decay = 0., batchsize=100, epochs = 50, use_biases = True):
        '''
        Implementation of a multi-layer perceptron, i.e., a feedforward neural network.
        Provides same methods as sklearn models.

        neurons_per_layer: list with number of neurons per layer, e.g., [29, 100, 100, 1]
        '''
        super(MLP, self).__init__()

        # parameters for setting up the neural network
        self.neurons_per_layer = neurons_per_layer
        self.use_biases = use_biases
        self.layers = nn.ModuleList()
        self.dropout_layer = nn.Dropout(p=dropout_rate)

        # create linear maps for each layer
        for i in range(1, len(neurons_per_layer)):
            self.layers.append(nn.Linear(self.neurons_per_layer[i-1], self.neurons_per_layer[i], bias = self.use_biases))

        # parameters for learning
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batchsize = batchsize
        self.epochs = epochs

    def forward(self, x):
        '''
        Build neural network.
        '''
        for act in self.layers[:-1]:
            x = act(x)
            x = F.leaky_relu(x)
            x = self.dropout_layer(x)
        x = self.layers[-1](x)

        return x.flatten()

    def fit(self, Xtrain, Ytrain, Xval, Yval):
        '''
        Train neural network. Takes training and validation data as input.

        Best model (on validation data) is saved.
        '''
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        lossf = nn.BCEWithLogitsLoss()

        # load tabular data as torch datasets
        dataset = TensorDataset(th.Tensor(Xtrain), th.Tensor(Ytrain))
        valset = TensorDataset(th.Tensor(Xval), th.Tensor(Yval))
        train_loader = th.utils.data.DataLoader(dataset, batch_size=self.batchsize, shuffle=True)
        eval_loader = th.utils.data.DataLoader(valset, batch_size=self.batchsize, shuffle=False)

        # best metric maximises tpr-fdr
        best_metric = 0
        for i in range(self.epochs):
            self.train()
            for (data, target) in tqdm(train_loader, leave=False):
                optimizer.zero_grad()
                output = self.forward(data).flatten()
                loss = lossf(output,target)
                loss.backward()
                optimizer.step()
            self.eval()
            outputs = []
            with th.no_grad():
                for data, target in eval_loader:
                    output = self.forward(data)
                    output = list((output > 0.5).detach().numpy().flatten())
                    outputs += output
            outputs = np.array(outputs).flatten()
            tpr, fdr = np.mean(outputs[Yval==1]==1),np.mean(Yval[outputs==1]==0)
            # save the model if it performs better as the last best model (on val. split)
            if tpr-fdr > best_metric:
                th.save({
                    'epoch': i,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tpr': tpr,
                    'fdr': fdr,
                    }, 'NeuralNet--best')
                best_metric = tpr-fdr
                print('NEW BEST!')
            print('EPOCH {}: tpr = {} and fdr = {}'.format(i, tpr, fdr))
        # load the best model for testing
        best_checkpoint = th.load('NeuralNet--best')
        self.load_state_dict(best_checkpoint['model_state_dict'])

    def predict(self, x):
        '''
        Predict classes.
        '''
        x = th.Tensor(x)
        return self.predict_proba(x)[:,1] > 0.5

    def predict_proba(self, x):
        '''
        Return class probabilities.
        '''
        self.eval()
        x = th.Tensor(x)
        probs = th.sigmoid(self.forward(x))
        probs = th.cat((1-probs.unsqueeze(-1), probs.unsqueeze(-1)), -1).detach().numpy()
        return probs