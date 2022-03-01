import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision
import numpy as np
from astropy.io import fits

def load_data(data, galaxies_to_be_used, path = '../data/ImageData'):
    '''
    Data for loading image data and assigning it the corresponding label (GC, non-GC).
    '''
    images, labels, probabilities, galaxies, IDs = [], [], [], [], []
    # for all galaxies
    for gal in galaxies_to_be_used:
        # load and append images
        if len(images) == 0:
            images = fits.getdata('{}/{}_images.fits'.format(path, gal))
        else:
            images = np.append(images, fits.getdata('{}/{}_images.fits'.format(path, gal)), axis=0)
        # append pGC, labels galaxy name and IDs from the tabular data
        probs =  data[data['galaxy']==gal]['pGC'].values
        labels += list(probs>=0.5)
        probabilities += list(probs)
        ID = list(data[data['galaxy']==gal].ID.values)
        galaxies += [gal]*len(ID)
        IDs += ID
    # set NaN pixel values to 0
    # (note: would be better to use offset from image preprocessing)
    images = np.nan_to_num(images, 0, nan=0)
    labels = list(np.array(labels, dtype=int))
    
    return images, np.array(labels), np.array(probabilities), np.array(galaxies), np.array(IDs)    

class CustomGCDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, images, labels, transform=None):
        self.images = torch.Tensor(images)
        self.labels = torch.Tensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]

        if self.transform:
            x = self.transform(x)

        y = self.labels[index]

        return x, y

    def __len__(self):
        return self.images.size(0)

class RandomShift(torch.nn.Module):
    """
    Randomly shift filters by a pixel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        f1 = shift_filter(tensor[0])
        f2 = shift_filter(tensor[1])
        final_f = torch.cat([f1,f2], 0)
        return final_f

def shift_filter(image):
    '''
    Helper function for RandomShift.
    '''
    p = np.random.random()
    if p <= 0.25:
        return image[1:,1:].unsqueeze(0)
    elif p > 0.25 and p <= 0.5:
        return image[:-1,1:].unsqueeze(0)
    elif p > 0.5 and p <= 0.75:
        return image[1:,:-1].unsqueeze(0)
    else:
        return image[:-1,:-1].unsqueeze(0)