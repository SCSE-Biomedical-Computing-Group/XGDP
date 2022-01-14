"""## utils.py"""

import os
import numpy as np
from math import sqrt
from scipy import stats

from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA

import torch
import matplotlib.pyplot as plt

import functions 

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='gdrive/MyDrive/FYP/Data/DRP/root_folder', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None,saliency_map=False, testing = False):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        self.dataset = dataset
        self.saliency_map = saliency_map
        self.testing  = testing

        if (self.testing):
            self.process(xd, xt, y,smile_graph)
        elif os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, smile_graph):
        ## xd : smile,
        ## cx : np array of 735 mutation values,
        ## y : IC50 value
        ## smile_graph : dictionary keys : smile, and values : 4_drug_outputs (num of mols, 72 features, edges, graph)


        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):

            if ((i%2000 == 0 or i+1 == data_len) and (not self.testing)):
                print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            c_size, features, edge_index, edge_features, this_graph = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:

            if (self.testing):
                ptr_F =torch.tensor([0, int(c_size)])
                batch_F = torch.zeros((int(c_size)), dtype = int)
                GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]), batch = batch_F, ptr = ptr_F)
            else:
                GCNData = DATA.Data(x=torch.Tensor(features),                                       ## rid_00
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),        ## rid_01
                                    edge_features=torch.Tensor(edge_features),
                                    y=torch.FloatTensor([labels]))                                  ## rid_02   tensor([0.6563])


            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target = torch.tensor([target], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])                                     ## rid_03

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if (self.testing):
            ptr_F =torch.tensor([0, int(c_size)])
            batch_F = torch.zeros((int(c_size)), dtype = int)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if (self.testing):
            return data_list  ## nico utils

        print('Graph construction done. Saving to file.')
        bts = sys.getsizeof(data_list)
        print(f"data_list: {bts} bytes")
        print(f"data_list: {bts/1000000} mb")
        print(f"data_list: {bts/1000000000} gb")
        print(f"len(data_list): {len(data_list)}")
        print(f"type(data_list[0]): {type(data_list[0])}")
        data, slices = self.collate(data_list)

        if (self.testing):
            return (data, slices)  ## nico utils

        print(" Saved to file.")
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
        print(" Complete.")

    def getXD(self):
        return self.xd

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(title+".png")

def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(title+".png")
