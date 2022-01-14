"""## Functions"""


import pandas as pd
import pubchempy as pcp
from collections import Counter

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

import numpy as np
from rdkit.Chem import AllChem
from rdkit import DataStructs
from molvs import standardize_smiles
import random
import time
import networkx as nx
import csv
import math

import sys

import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import datetime
import argparse
import nvidia_smi



IPythonConsole.ipython_useSVG=True  #< set this to False if you want PNGs instead of SVGs
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def ctoint(_c):
  return int(str(_c)[9:-1])

def drugnameToSmiles(dn):
  return pcp.get_compounds(dn, 'name')[0].isomeric_smiles

def show_structure(sm, show_smiles = False):
  if (show_smiles):
    print(f"Smiles : {sm}")
  mol = Chem.MolFromSmiles(sm)
  return mol

def get_ecfp_sparsity(sml, fpl):
    tmp = ECFP6(sml).compute_ECFP6(fp_length = fpl, generate_df=False)
    return round(100*np.sum(tmp)/(tmp.shape[0]*tmp.shape[1]) , 2)

def norm_ic50(ic):
    return 1 / (1 + pow(math.exp(float(ic)), -0.1))

def denorm_ic50(ic):
    return -10*math.log((1-ic)/ic)

def predict_this(mdl, sml, cid, do_ECFP = False, fpl = None):
    cell_dict_X, cell_feature_X = save_cell_mut_matrix_XO()
    drug_dict_X, drug_smile_X, comp_smg = load_drug_smile_X(do_ECFP, fpl)

    mut_arr = cell_feature_X[cell_dict_X[str(cid)]]

    mut_arr = mut_arr.reshape(1, mut_arr.shape[0])
    sml_arr = np.array([sml])
    y_arr = np.array([0])
    smg = {sml: comp_smg[sml]}

    pr = TestbedDataset(root='', dataset="", xd=sml_arr, xt=mut_arr, y=y_arr, smile_graph=smg, testing = True).process(sml_arr, mut_arr, y_arr,smg)

    return denorm_ic50(float(mdl.forward(pr[0])[0][0][0]))

def list_difference(a, b):
    both = []
    in_a = []
    in_b = []
    if len(a) > len(b):
        for i in a:
            if i in b:
                both.append(i)
            else:
                in_a.append(i)
        for q in b:
            if q not in a:
                in_b.append(q)
    else:
        for i in b:
            if i in a:
                both.append(i)
            else:
                in_b.append(i)
        for q in a:
            if q not in b:
                in_a.append(q)
    return in_a, in_b, both

def atom_features_X(atom):
    return np.array(one_of_k_encoding_unk_X(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding_X(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk_X(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk_X(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding_X(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk_X(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_ecfp_identifiers(smiles, radius):
    """
        Returns the ECFP hashed value for a given smiles
        Input:
            smiles (str) : Smiles in string format
            radius (int) : Radius of ECFP, For ECFP6 Radius = 3
        Output:
            atomIndex_hash_1 (Dictionary) : Keys - Atom index, Values - List containing radius number of hash values,
    """
    mol = Chem.MolFromSmiles(smiles)
    bond_types = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    atomIndex_hash_1 = {}

    for atom in mol.GetAtoms():
        temp = (atom.GetDegree(), atom.GetTotalValence()-atom.GetTotalNumHs(), atom.GetAtomicNum(), int(atom.GetMass()), atom.GetFormalCharge(), atom.GetTotalNumHs(), int(atom.GetIsAromatic()))
        hs = hash(temp)
        atomIndex_hash_1[atom.GetIdx()] = [hs]
    for i in range(radius-1):
        l1 = []
        for atom_idx, hash_stack in atomIndex_hash_1.items():
            this_l = []
            hsh = hash_stack[-1]
            this_atom = mol.GetAtoms()[atom_idx]
            this_l.append(i+1)
            this_l.append(hsh)

            neighs_l = []
            for neigh in this_atom.GetNeighbors():
                neigh_idx = neigh.GetIdx()
                bd = [b for b in this_atom.GetBonds()]
                this_bond = mol.GetBondBetweenAtoms(this_atom.GetIdx(), (neigh.GetIdx()))
                this_bond_type = str(this_bond.GetBondType())
                this_bnum = bond_types.index(this_bond_type) + 1
                neighs_l.append((this_bnum, atomIndex_hash_1[neigh_idx][i]))
            neighs_l.sort(key = lambda x: x[1])
            for tup in neighs_l:
                this_l.append(tup[0])
                this_l.append(tup[1])
            l1.append(this_l)
            atomIndex_hash_1[atom_idx].append(hash(tuple(this_l)))
    return atomIndex_hash_1

def get_ecfp_node_features(smiles, radius, use_radius = None):
    """
        Returns the ECFP atom features for each atom in given smiles in form of a list of numpy arrays
        Input:
            smiles (str) : Smiles in string format
            radius (int) : Radius of ECFP, For ECFP6 Radius = 3
            use_radius (int) : only considers the ECFP value of fixed radius
        Output:
            features (list) : List of numpy arrays containing atom features
    """
    identifiers = get_ecfp_identifiers(smiles, radius)
    features = []
    for atomidx, ecfp_list in identifiers.items():
        if (use_radius != None):
            ecfp_list = [ecfp_list[use_radius-1]]

        bin_ecfp = ""
        for i, this_ecfp in enumerate(ecfp_list):
            bin_ecfp = bin_ecfp + bin(abs(this_ecfp)).replace("0b", "")
            bin_len = len(bin_ecfp)
            for bit in range(bin_len, int((i+1)*64)):
                bin_ecfp = bin_ecfp + "0"
        np_ecfp = np.array([int(char) for char in bin_ecfp]) ## .astype("uint8")
        features.append(np_ecfp)
    return features

def smile_to_graph_X(smile, do_mol_ecfp, fpl = None, do_edge_features = False, do_atom_ecfp = False, ecfp_radius = 3, use_radius = None):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    if (do_atom_ecfp):
        features = get_ecfp_node_features(smile, ecfp_radius, use_radius)
    else:
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features_X(atom)
            if (do_mol_ecfp):
                ecfp6_descriptor = ECFP6([smile])
                this_ecfp = ecfp6_descriptor.compute_ECFP6(fpl, generate_df=False)[0]
                feature = np.append(feature, this_ecfp, 0)

            # features.append( feature / sum(feature) ) ## Normalise
            features.append(sum(feature)*feature / sum(feature))

    edge_dict = {}
    edges = []
    bond_features = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])


        if (do_edge_features):
            this_feat = [0 for q in range(4)]
            q = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'].index(str(bond.GetBondType()))
            this_feat[q] = 1
            this_feat = np.array(this_feat)

            edge_dict[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = this_feat
            edge_dict[(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())] = this_feat

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        if (do_edge_features):
            bond_features.append(edge_dict[(e1, e2)])

    if (do_edge_features):
        return c_size, features, edge_index, bond_features, g
    else:
        return c_size, features, edge_index, g

def load_drug_smile_X(do_mol_ecfp = False, fpl = None, do_edge_features = False, do_atom_ecfp = False, ecfp_radius = None, use_radius = None):
    """
      Output :
        (dictionary) drug_dict : Keys - (str) name of drug, Values - (int) index/position of drug in drug_smile
        (list) drug_smile : List of all drug smiles
        (dictionary) smile_graph : Keys - (str) smiles of all drugs, Values - (tup) Five outputs of function smile_to_graph(smile)
    """
    drug_dict = {}
    drug_smile = []


    reader = csv.reader(open(folder + "drug_smiles.csv"))     ## From csv
    next(reader, None)                                        ## From csv

    for cnt, item in enumerate(reader):                       ## From csv
                                                                ## From df3
        name = item[0]
        smile = item[2]                                       ## From csv

        if (smile == "N.N.[Cl-].[Cl-].[Pt+2]"):
            print(f"name = {name}, smile = {smile}")
        # smile = item[1]                                                                           ## From df3

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
        if (smile == "N.N.[Cl-].[Cl-].[Pt+2]"):
            print(f"indx = {len(drug_smile)} , {drug_smile[-1]}")

    smile_graph = {}
    for smile in drug_smile:
        # g = smile_to_graph(smile)
        if (do_edge_features):
            gr = smile_to_graph_X(smile, do_mol_ecfp, fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius)
        else:
            gr = smile_to_graph_X(smile, do_mol_ecfp, fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius)
        smile_graph[smile] = gr

    return drug_dict, drug_smile, smile_graph

def save_cell_mut_matrix_X():
    """
    PANCANCER_Genetic_feature.csv
    0                1                 2           3          4         5                6
    cell_line_name	cosmic_sample_id	gdsc_desc1	gdsc_desc2	tcga_desc	genetic_feature	is_mutated

    Output :
        cell_dict :
        cell_feature : np array of shape (unique_cosmic_sample_id x 732)
        matrix_list
        mut_dict

    """
    f = open(folder + "PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1] ## cosmic_sample_id       1290730, 1290730, 1290730
        mut = item[5] ## genetic_feature            CDC27_mut, CDC73_mut, CDH1_mut
        is_mutated = int(item[6]) ## is_mutated     0, 0, 0

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))

    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    return cell_dict, cell_feature, matrix_list, mut_dict


def save_cell_mut_matrix_XO():
    """
 Output :
        (dictionary) cell_dict : Keys - (str) cosmic_sample_id, Values - (int) index/position of the key (cosmic_sample_id) in uniquely sorted list of cosmic_sample_id values
        (np array) cell_feature : Numpy array of shape (len(cell_dict), len(mut_dict)),
                                    1 if that (cosmic_sample_id, genetic_feature) pair has is_mutated = 1
                                    else 0
    """
    f = open(folder + "PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]           ## cosmic_sample_id
        mut = item[5]               ## genetic_feature
        is_mutated = int(item[6])   ## is_mutated

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))

    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)

    return cell_dict, cell_feature

def save_mix_drug_cell_matrix_X(do_mol_ecfp=False, fpl=None, do_edge_features=False, do_atom_ecfp=False, ecfp_radius=None, use_radius = None):
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature, qa, aq = save_cell_mut_matrix_X()
    drug_dict, drug_smile, smile_graph = load_drug_smile_X(do_mol_ecfp, fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius)

    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]    ## Drug name
        cell = item[3]    ## Cosmic sample Id
        ic50 = item[8]    ## IC50
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []
    random.shuffle(temp_data)
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            xd.append(drug_smile[drug_dict[drug]])        ## appending the smile of the drug into list xd
            xc.append(cell_feature[cell_dict[cell]])      ## appending numpy array of shape (len(mut_dict),) ie. (735,) to list xc
            y.append(ic50)                                ## appending (int) ic50 value of that smile to list y
            bExist[drug_dict[drug], cell_dict[cell]] = 1  ## (drug_name, Cosmic_sample_Id) pair used to index the numpy array and set to 1
            lst_drug.append(drug)                         ## appending (str) name of this drug to list lst_drug
            lst_cell.append(cell)                         ## appending (numeric str) this Cosmic sample Id to list lst_cell

    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)


    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)
    xd_train = xd[:size]
    xd_val = xd[size:size1]
    xd_test = xd[size1:]

    xc_train = xc[:size]
    xc_val = xc[size:size1]
    xc_test = xc[size1:]

    y_train = y[:size]
    y_val = y[size:size1]
    y_test = y[size1:]

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    return xd, xc, y

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    torch.cuda.empty_cache()  ## no grad
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, br_fol, save_folder, save_name, do_save = True):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    model_st = modeling.__name__
    dataset = 'GDSC'
    train_losses = []
    val_losses = []
    val_pearsons = []
    print('\nrunning on ', model_st + '_' + dataset )

    # processed_data_file_train = 'data/processed/' + dataset + '_train_mix'+'.pt'
    # processed_data_file_val = 'data/processed/' + dataset + '_val_mix'+'.pt'
    # processed_data_file_test = 'data/processed/' + dataset + '_test_mix'+'.pt'
    processed_data_file_train = br_fol + '/processed/' + dataset + '_train_mix'+'.pt'
    processed_data_file_val = br_fol + '/processed/' + dataset + '_val_mix'+'.pt'
    processed_data_file_test = br_fol + '/processed/' + dataset + '_test_mix'+'.pt'

    # root_folder+"root_001/processed/GDSC_train_mix.pt"

    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root=br_fol, dataset=dataset+'_train_mix')
        val_data = TestbedDataset(root=br_fol, dataset=dataset+'_val_mix')
        test_data = TestbedDataset(root=br_fol, dataset=dataset+'_test_mix')


        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        print("CPU/GPU: ", torch.cuda.is_available())

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print(device)
        model = modeling().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1
        model_file_name = 'model_' + save_name + '_' + dataset +  '.model'
        # result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        result_file_name = 'result_' + save_name + '_' + dataset +  '.csv'
        loss_fig_name = 'model_' + save_name + '_' + dataset + '_loss'
        pearson_fig_name = 'model_' + save_name + '_' + dataset + '_pearson'
        total_time = 0
        for epoch in range(num_epoch):
            start_time = time.time()
            print(f"epoch : {epoch+1}/{num_epoch} ")



            #
            nvidia_smi.nvmlInit()

            deviceCount = nvidia_smi.nvmlDeviceGetCount()
            for i in range(deviceCount):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

            nvidia_smi.nvmlShutdown()
            ######################





            train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval)
            G,P = predicting(model, device, val_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]

            G_test,P_test = predicting(model, device, test_loader)
            ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]

            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])

            if ret[1]<best_mse:
                if (do_save):
                    torch.save(model.state_dict(), save_folder + model_file_name)
                    with open(save_folder + "val_"+ result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                    with open(save_folder + "test_"+ result_file_name,'w') as f:
                        f.write(','.join(map(str,ret_test)))
                best_epoch = epoch+1
                best_mse = ret[1]
                best_pearson = ret[2]
                print(f"ret = {ret}")
                print(f"ret_test = {ret_test}")
                print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
            else:
                print(f"ret = {ret}")
                print(f"ret_test = {ret_test}")
                print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse, best_pearson, model_st, dataset)

            total_time += time.time() - start_time
            remaining_time = (num_epoch-epoch-1)*(total_time)/(epoch+1)
            print(f"End of Epoch {epoch+1}; {int(remaining_time//3600)} hours, {int((remaining_time//60)%60)} minutes, and {int(remaining_time%60)} seconds remaining")

        draw_loss(train_losses, val_losses, loss_fig_name)
        draw_pearson(val_pearsons, pearson_fig_name)
