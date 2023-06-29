import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
from functools import reduce
from torch_geometric.data import InMemoryDataset
# from torch_geometric.loader import DataLoader     # for pyg >= 2.0
# pyg < 2, seems also works on pyg >= 2.0
from torch_geometric.data import DataLoader
from torch_geometric import data as DATA
import torch
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.nn.functional as F
from utils_preproc import save_gene_expr_matrix_X
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
from utils_data import *

def string2smiles_list(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i-1:i+1])
            i += 1
        else:
            char_list.append(string[i-1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list

def onehot_encode(char_list, smiles_string, length):
    encode_row = lambda char: list(map(int, [c == char for c in smiles_string]))
#     ans = np.array(list(map(encode_row, char_list)))
    ans = np.array([encode_row(c) for c in char_list])
#     print(ans)
    if ans.shape[1] < length:
        residual = np.zeros((len(char_list), length - ans.shape[1]), dtype=np.int8)
        ans = np.concatenate((ans, residual), axis=1)
    return ans

def smiles_to_onehot(smiles, c_chars, c_length):
    c_ndarray = np.ndarray(shape=(len(smiles), len(c_chars), c_length), dtype=np.float32)
    for i in range(len(smiles)):
        c_ndarray[i, ...] = onehot_encode(c_chars, smiles[i], c_length)
    return c_ndarray

def smiles_to_onehot_dict(smiles, c_chars, c_length):
    onehot_dict = {}
    for s in smiles:
        onehot_dict[s] = onehot_encode(c_chars, s, c_length)
        # print(onehot_dict[s].shape)
    return onehot_dict

def load_as_ndarray(folder='data/GDSC/'):
    reader = csv.reader(open(folder + "drug_smiles.csv"))
    next(reader, None)
    smiles = np.array(list(reader), dtype=np.str)
    return smiles

def charsets(smiles):
    union = lambda x, y: set(x) | set(y)
    c_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 2]))))
    i_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 3]))))
    return c_chars, i_chars


def get_drug_onehot_dict(folder='data/GDSC/'):
    smiles = load_as_ndarray()
    c_chars, i_chars = charsets(smiles)
    c_length = max(map(len, map(string2smiles_list, list(smiles[:, 2]))))
    i_length = max(map(len, map(string2smiles_list, list(smiles[:, 3]))))
    c_onehot = smiles_to_onehot_dict(smiles[:, 2], c_chars, c_length)
    i_onehot = smiles_to_onehot_dict(smiles[:, 3], i_chars, i_length)
    return c_onehot, i_onehot


def load_drug_smile_X(folder='data/GDSC/'):
    drug_dict = {}
    drug_smile = []

    reader = csv.reader(open(folder + "drug_smiles.csv"))  # From csv
    next(reader, None)  # From csv

    for cnt, item in enumerate(reader):  # From csv
        # From df3
        # print(item)
        name = item[0]
        smile = item[2]  # From csv
        # smile = item[-1]                                       ## From csv

        # skip the Cisplatin drug
        if (smile == "N.N.[Cl-].[Cl-].[Pt+2]"):
            print(f"name = {name}, smile = {smile}")
            continue
        # smile = item[1]                                                                           ## From df3

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
        if (smile == "N.N.[Cl-].[Cl-].[Pt+2]"):
            print(f"indx = {len(drug_smile)} , {drug_smile[-1]}")

    c_onehot_dict, _ = get_drug_onehot_dict(folder)

    return drug_dict, drug_smile, c_onehot_dict


class tCNNDataset(InMemoryDataset):
    def __init__(self, root='root_folder', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, onehot_dict=None, saliency_map=False, testing=False, dgl=None, cosl=None):

        super(tCNNDataset, self).__init__(root, transform, pre_transform)

        self.dataset = dataset
        self.saliency_map = saliency_map
        self.testing = testing

        if (self.testing):
            self.process(xd, xt, y, onehot_dict, dgl, cosl)
        elif os.path.isfile(self.processed_paths[0]):
            print(
                'Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(
                'Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, onehot_dict, dgl, cosl)
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

    def process(self, xd, xt, y, onehot_dict, dgl, cosl):
        # xd : smile,
        # cx : np array of 735 mutation values,
        # y : IC50 value
        # smile_graph : dictionary keys : smile, and values : 4_drug_outputs (num of mols, 72 features, edges, graph)

        assert (len(xd) == len(xt) and len(xt) == len(
            y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):

            if ((i % 2000 == 0 or i+1 == data_len) and (not self.testing)):
                print('Converting SMILES to one hot codes: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            dg_name = dgl[i]
            cos_name = cosl[i]

            onehot_arr = onehot_dict[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:

            if (self.testing):
                GCNData = DATA.Data(x=torch.Tensor(onehot_arr),
                                    y=torch.FloatTensor([labels]), smiles=smiles, drug_name=dg_name, cell_line_name=cos_name)
            else:
                GCNData = DATA.Data(x=torch.Tensor(onehot_arr),
                                    y=torch.FloatTensor([labels]), smiles=smiles, drug_name=dg_name, cell_line_name=cos_name)

            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target = torch.tensor(
                    [target], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])  # rid_03

            # GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            # if ((i % 2000 == 0 or i+1 == data_len) and (not self.testing)):
            #     print(GCNData)
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if (self.testing):
            return data_list  # nico utils

        print('Graph construction done. Saving to file.')
        bts = sys.getsizeof(data_list)
        print(f"data_list: {bts} bytes")
        print(f"data_list: {bts/1000000} mb")
        print(f"data_list: {bts/1000000000} gb")
        print(f"len(data_list): {len(data_list)}")
        print(f"type(data_list[0]): {type(data_list[0])}")
        data, slices = self.collate(data_list)

        if (self.testing):
            return (data, slices)  # nico utils

        print(" Saved to file.")
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
        print(" Complete.")

    def getXD(self):
        return self.xd
    

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval, return_attention_weights=False):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        x, x_cell_mut = data.x, data.target
        output = model(x, x_cell_mut)
        
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        # if batch_idx % log_interval == 0:
        #     print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
        #                                                                    batch_idx * len(data.x),
        #                                                                    len(train_loader.dataset),
        #                                                                    100. * batch_idx / len(train_loader),
        #                                                                    loss.item()))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader, return_attention_weights = False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            x, x_cell_mut = data.x, data.target
            output = model(x, x_cell_mut)
        
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    torch.cuda.empty_cache()  ## no grad
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def main_cv(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, br_fol, result_folder, model_folder, save_name, do_save = True):
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    model_st = modeling.__name__
    dataset = 'GDSC'
    
    print('\nrunning on ', model_st + '_' + dataset )

    processed_data_file_cv = br_fol + '/processed/' + dataset + '_cv_mix'+'.pt'
    processed_data_file_test = br_fol + '/processed/' + dataset + '_test_mix'+'.pt'
    assert os.path.isfile(processed_data_file_cv) and os.path.isfile(processed_data_file_test)

    cv_data = tCNNDataset(root=br_fol, dataset=dataset+'_cv_mix')
    test_data = tCNNDataset(root=br_fol, dataset=dataset+'_test_mix')
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

    kf = KFold(n_splits=3)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(device)
    best_model_id = 0
    best_model = None
    best_pearson_cv = 0
    ret_cv = []
    for i, (train_index, val_index) in enumerate(kf.split(cv_data)):
        print("CV: ", i)
        train_data = Subset(cv_data, train_index)
        # print(len(train_data))
        val_data = Subset(cv_data, val_index)
        # print(len(val_data))
        
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)

        print("CPU/GPU: ", torch.cuda.is_available())

        # training the model
        model = modeling().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_mse = 1000
        best_pearson = 0
        best_epoch = -1
        model_file_name = 'model_' + save_name + '_' + dataset + '_' + str(i) +  '.model'
        # result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        loss_fig_name = 'model_' + save_name + '_' + dataset + '_' + str(i) + '_loss'
        pearson_fig_name = 'model_' + save_name + '_' + dataset + '_' + str(i) + '_pearson'
        total_time = 0
        early_stop_tolerance = 30
        train_losses = []
        val_losses = []
        val_pearsons = []
        best_ret = []

        for epoch in tqdm(range(num_epoch)):
            # torch.cuda.empty_cache()
            start_time = time.time()
            print(f"epoch : {epoch+1}/{num_epoch} ")

            train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval)
            G,P = predicting(model, device, val_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),coeffi_determ(G,P)]

            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])

            if ret[1]<best_mse:
                if (do_save):
                    torch.save(model.state_dict(), model_folder + model_file_name)
                    
                best_epoch = epoch+1
                best_mse = ret[1]
                best_pearson = ret[2]
                best_ret = ret
                print(f"ret = {ret}")
                # print(f"ret_test = {ret_test}")
                print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
            else:
                print(f"ret = {ret}")
                # print(f"ret_test = {ret_test}")
                print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse, best_pearson, model_st, dataset)

            total_time += time.time() - start_time
            if (epoch - best_epoch) > early_stop_tolerance:
                print('early stop at epoch ', epoch)
                break
            # remaining_time = (num_epoch-epoch-1)*(total_time)/(epoch+1)
            # print(f"End of Epoch {epoch+1}; {int(remaining_time//3600)} hours, {int((remaining_time//60)%60)} minutes, and {int(remaining_time%60)} seconds remaining")

        draw_loss(train_losses, val_losses, result_folder + loss_fig_name)
        draw_pearson(val_pearsons, result_folder + pearson_fig_name)
        ret_cv.append(best_ret)

        if best_pearson > best_pearson_cv:
            best_pearson_cv = best_pearson
            best_model = model
            best_model_id = i
            print('best model changed to ', best_model_id)
    
    # test with the model with best validation performance
    G_test, P_test = predicting(best_model, device, test_loader)
        
    result_file_name = 'result_' + save_name + '_' + dataset + '_' +  '.csv'
    ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test),coeffi_determ(G_test,P_test)]
    if do_save:
        best_model_file_name = 'model_' + save_name + '_' + dataset + '_best' + str(best_model_id) +  '.model'
        torch.save(best_model.state_dict(), model_folder + best_model_file_name)
        ret_cv.append(ret_test)    # last line is for test
        ret_df = pd.DataFrame(ret_cv, columns = ['RMSE','MSE','Pearson','Spearman','R2'])
        ret_df.to_csv(result_folder + result_file_name)


class tCNN(torch.nn.Module):
    def __init__(self, n_input=28, n_output=1, batch_size=32, dropout=0.5):
        super(tCNN, self).__init__()
        self.n_input = n_input

        self.conv1 = torch.nn.Conv1d(n_input, 40, kernel_size=7)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = torch.nn.Conv1d(40, 80, kernel_size=7)
        self.pool2 = nn.MaxPool1d(3)
        self.conv3 = torch.nn.Conv1d(80, 60, kernel_size=7)
        self.pool3 = nn.MaxPool1d(3)

        self.conv_xd_1 = torch.nn.Conv1d(1, 40, kernel_size=7)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xd_2 = torch.nn.Conv1d(40, 80, kernel_size=7)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xd_3 = torch.nn.Conv1d(80, 60, kernel_size=7)
        self.pool_xt_3 = nn.MaxPool1d(3)

        self.fc1 = torch.nn.Linear(60*(4+32), 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.out = torch.nn.Linear(1024, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, xt):
        x = x.view(-1, self.n_input, x.shape[-1])    # dataloader of pyg will accumulate batch_size and n_input together
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        xt = self.pool_xt_1(F.relu(self.conv_xd_1(xt)))
        xt = self.pool_xt_2(F.relu(self.conv_xd_2(xt)))
        xt = self.pool_xt_3(F.relu(self.conv_xd_3(xt)))

        # print(x.shape, xt.shape)
        xc = torch.cat((x, xt), 2)
        xc = xc.view(-1, xc.shape[1]*xc.shape[2])
        xc = F.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = F.relu(self.fc2(xc))
        xc = self.dropout(xc)
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out
