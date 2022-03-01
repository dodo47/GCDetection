import matplotlib.pyplot as plt

def plot_source(source):
    '''
    Plots the g and z band of a source.
    Source is a 2xNxN image (array).
    '''
    fig,ax = plt.subplots(1,2, figsize=(10,10))
    ax[0].imshow(source[0])
    ax[1].imshow(source[1], cmap = 'magma')