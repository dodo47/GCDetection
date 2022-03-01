from globsML.utils.eval import quick_eval_fdr
from tqdm.notebook import tqdm, trange
import numpy as np
import torch

def train_CNN(model, device, train_loader, optimizer):
    '''
    Train for one epoch (go through the whole training data once).
    
    model: CNN model
    device: device used for inference (CPU or GPU)
    train_loader: training data loader
    optimiser: torch optimiser used to update model parameters
    '''
    
    model.train()
    losses = []
    # binary cross entropy loss function
    lossf = torch.nn.BCELoss()

    for (data, target) in tqdm(train_loader, leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # predict
        output, xvec = model(data)
        # calculate loss
        loss = lossf(output.flatten(),target) 
        loss.backward()
        # update parameters
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

def test_CNN(model, device, loader, labels, epoch):
    '''
    Evaluate CNN model.
    
    model: CNN model
    device: device used for inference (CPU or GPU)
    loader: data loader
    labels: corresponding labels
    epoch: current training epoch
    '''
    model.eval()
    outputs = []
    with torch.no_grad():
        # go through the test data
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output, xvec = model(data)
            output = list((output > 0.5).cpu().detach().numpy().flatten())
            outputs += output
    # get true positive rate and false detection rate
    tpr, fdr = quick_eval_fdr(labels, np.array(outputs).flatten())
    
    return tpr, fdr