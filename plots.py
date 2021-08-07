import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import os

#for loss and accuracy graphs
def plot_graph(graph_data, num_epochs, model):
    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.title('Acc vs Epoch [Model {}]'.format(model))
    plt.plot(range(1, num_epochs+1), graph_data['val_acc'], label='val_acc')

    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()

    plt.subplot(122)
    plt.title('Loss vs Epoch [Model {}]'.format(model))
    plt.plot(range(1, num_epochs+1), graph_data['train_loss'], label='train_loss')
    plt.plot(range(1, num_epochs+1), graph_data['val_loss'], label='val_loss')
    plt.xticks((np.asarray(np.arange(1, num_epochs+1, 1.0))))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    
    fname = 'model{}_graph.png'.format(model)
    
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    print('plot saved as {}\n\n'.format(fname))
    
def plot_bar(dataset_type):
    pass