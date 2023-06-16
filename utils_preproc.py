"""## Functions"""


import pandas as pd
import pickle
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
import sys
import os
from random import shuffle
import torch
import torch.nn as nn
import datetime
import argparse
import nvidia_smi

# from dataframes import ECFP6
from utils_data import *
# from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader     # for pyg >= 2.0
# from torch_geometric.data import DataLoader         # pyg < 2, seems also works on pyg >= 2.0


# < set this to False if you want PNGs instead of SVGs
IPythonConsole.ipython_useSVG = True


class ECFP6:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles

    def mol2fp(self, mol, fp_length, radius=3):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=fp_length)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def compute_ECFP6(self, fp_length, name=None, generate_df=True):
        bit_headers = ['bit' + str(i) for i in range(fp_length)]
        arr = np.empty((0, fp_length), int).astype(int)
        for i in self.mols:
            fp = self.mol2fp(i, fp_length)
            arr = np.vstack((arr, fp))
        if (not generate_df):
            return np.asarray(arr).astype(int)
        df_ecfp6 = pd.DataFrame(np.asarray(
            arr).astype(int), columns=bit_headers)
        df_ecfp6.insert(loc=0, column='smiles', value=self.smiles)
        if name != None:
            df_ecfp6.to_csv(name[:-4]+'_ECFP6.csv', index=False)

        return df_ecfp6


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def ctoint(_c):
    return int(str(_c)[9:-1])


def drugnameToSmiles(dn):
    return pcp.get_compounds(dn, 'name')[0].isomeric_smiles


def show_structure(sm, show_smiles=False):
    if (show_smiles):
        print(f"Smiles : {sm}")
    mol = Chem.MolFromSmiles(sm)
    return mol


def get_ecfp_sparsity(sml, fpl):
    tmp = ECFP6(sml).compute_ECFP6(fp_length=fpl, generate_df=False)
    return round(100*np.sum(tmp)/(tmp.shape[0]*tmp.shape[1]), 2)


def norm_ic50(ic):
    return 1 / (1 + pow(math.exp(float(ic)), -0.1))


def denorm_ic50(ic):
    return -10*math.log((1-ic)/ic)


def predict_this(mdl, sml, cid, do_ECFP=False, fpl=None):
    cell_dict_X, cell_feature_X = save_cell_mut_matrix_XO()
    drug_dict_X, drug_smile_X, comp_smg = load_drug_smile_X(do_ECFP, fpl)

    mut_arr = cell_feature_X[cell_dict_X[str(cid)]]

    mut_arr = mut_arr.reshape(1, mut_arr.shape[0])
    sml_arr = np.array([sml])
    y_arr = np.array([0])
    smg = {sml: comp_smg[sml]}

    pr = TestbedDataset(root='', dataset="", xd=sml_arr, xt=mut_arr, y=y_arr,
                        smile_graph=smg, testing=True).process(sml_arr, mut_arr, y_arr, smg)

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
    return np.array(one_of_k_encoding_unk_X(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding_X(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk_X(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk_X(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding_X(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
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
        temp = (atom.GetDegree(), atom.GetTotalValence()-atom.GetTotalNumHs(), atom.GetAtomicNum(),
                int(atom.GetMass()), atom.GetFormalCharge(), atom.GetTotalNumHs(), int(atom.GetIsAromatic()))
        hs = hash(temp)
        atomIndex_hash_1[atom.GetIdx()] = [hs]
    for i in range(radius-1):       # should not minus 1??
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
                this_bond = mol.GetBondBetweenAtoms(
                    this_atom.GetIdx(), (neigh.GetIdx()))
                this_bond_type = str(this_bond.GetBondType())
                this_bnum = bond_types.index(this_bond_type) + 1
                neighs_l.append((this_bnum, atomIndex_hash_1[neigh_idx][i]))
            neighs_l.sort(key=lambda x: x[1])
            for tup in neighs_l:
                this_l.append(tup[0])
                this_l.append(tup[1])
            l1.append(this_l)
            atomIndex_hash_1[atom_idx].append(hash(tuple(this_l)))
    return atomIndex_hash_1


def get_ecfp_node_features(smiles, radius, use_radius=None, do_ordinary_atom_feat=False):
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
        feature = np.array([int(char)
                           for char in bin_ecfp])  # .astype("uint8")

        if do_ordinary_atom_feat:
            mol = Chem.MolFromSmiles(smiles)
            atom = mol.GetAtoms()[atomidx]
            ordinary_feature = atom_features_X(atom)
            # print(feature)
            # print(ordinary_feature)
            feature = np.concatenate((feature, ordinary_feature), axis=0)
            # print(feature)
        features.append(feature)
    return features


def smile_to_graph_X(smile, do_ordinary_atom_feat, do_mol_ecfp, fpl=None, do_edge_features=False, do_atom_ecfp=False, ecfp_radius=3, use_radius=None, use_relational_edge=False):
    '''
        Inputs:
            smile: SMILES vector of drug
            do_mol_ecfp: molecular level ecfp (all the atoms have same features)
            fpl: length of bit vectors of ecfp fingerprints (mol-level)
            do_edge_features: chemical bond type will be edge features
            do_atom_ecfp: atom level ecfp
            ecfp_radius (int) : Radius of ECFP, For ECFP6 Radius = 3
            use_radius (int) : only considers the ECFP value of fixed radius
    '''
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    if (do_atom_ecfp):
        # atom level ecfp features
        features = get_ecfp_node_features(
            smile, ecfp_radius, use_radius, do_ordinary_atom_feat)
    else:
        # benchmark atom features (symbol, degrees, ...)
        # if do_mol_ecfp, append the mol-level ecfp as well
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features_X(atom)
            if (do_mol_ecfp):
                ecfp6_descriptor = ECFP6([smile])
                this_ecfp = ecfp6_descriptor.compute_ECFP6(
                    fpl, generate_df=False)[0]
                feature = np.append(feature, this_ecfp, 0)

            # features.append( feature / sum(feature) ) ## Normalise
            features.append(sum(feature)*feature / sum(feature))

    edge_dict = {}
    edges = []
    bond_features = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

        if (do_edge_features):
            temp_feat = [0 for q in range(4)]
            q = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'].index(
                str(bond.GetBondType()))
            # print(q)
            temp_feat[q] = 1
            # print(temp_feat)

            if use_relational_edge:
                # print('creating relational mol graph')
                this_feat = np.array([q])
            else:
                # print('creating non-relational mol graph')
                this_feat = np.array(temp_feat)

            edge_dict[(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx())] = this_feat
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


def load_drug_smile_X(do_ordinary_atom_feat=False, do_mol_ecfp=False, fpl=None, do_edge_features=False, do_atom_ecfp=False, ecfp_radius=None, use_radius=None, use_relational_edge=False, folder="data/GDSC/"):
    """
      Output :
        (dictionary) drug_dict : Keys - (str) name of drug, Values - (int) index/position of drug in drug_smile
        (list) drug_smile : List of all drug smiles
        (dictionary) smile_graph : Keys - (str) smiles of all drugs, Values - (tup) Five outputs of function smile_to_graph(smile)
    """
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

    smile_graph = {}
    # print(drug_smile)
    for smile in drug_smile:
        # g = smile_to_graph(smile)
        # print(smile)
        if (do_edge_features):
            gr = smile_to_graph_X(smile, do_ordinary_atom_feat, do_mol_ecfp, fpl,
                                  do_edge_features, do_atom_ecfp, ecfp_radius, use_radius, use_relational_edge)
        else:
            gr = smile_to_graph_X(smile, do_ordinary_atom_feat, do_mol_ecfp,
                                  fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius)
        smile_graph[smile] = gr

    return drug_dict, drug_smile, smile_graph


def save_cell_mut_matrix_X(folder='data/GDSC/'):
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
        cell_id = item[1]  # cosmic_sample_id       1290730, 1290730, 1290730
        # genetic_feature            CDC27_mut, CDC73_mut, CDH1_mut
        mut = item[5]
        is_mutated = int(item[6])  # is_mutated     0, 0, 0

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


def save_cell_mut_matrix_XO(folder='data/GDSC/'):
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
        cell_id = item[1]  # cosmic_sample_id
        mut = item[5]  # genetic_feature
        is_mutated = int(item[6])  # is_mutated

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


def save_mix_drug_cell_matrix_X(do_ordinary_atom_feat=False, do_mol_ecfp=False, fpl=None, do_edge_features=False, do_atom_ecfp=False, ecfp_radius=None, use_radius=None, return_names=True, use_relational_edge=False, folder='data/GDSC/'):
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature, qa, aq = save_cell_mut_matrix_X()
    drug_dict, drug_smile, smile_graph = load_drug_smile_X(
        do_ordinary_atom_feat, do_mol_ecfp, fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius, use_relational_edge)

    print('drug number:', len(drug_dict))
    print('cell line number:', len(cell_dict))

    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]  # Drug name
        cell = item[3]  # Cosmic sample Id
        ic50 = item[8]  # IC50
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    print('total length of drug-cellline pair:', len(temp_data))

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []

    # TODO: remove this shuffle operation. (finished)
    # for mixed test, shuffle will be done in load_data.py, controlling by the random seed
    # for blind test, no shuffle is needed
    # random.shuffle(temp_data)

    n_missing = 0

    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            # appending the smile of the drug into list xd
            xd.append(drug_smile[drug_dict[drug]])
            # appending numpy array of shape (len(mut_dict),) ie. (735,) to list xc
            xc.append(cell_feature[cell_dict[cell]])
            # appending (int) ic50 value of that smile to list y
            y.append(ic50)
            # (drug_name, Cosmic_sample_Id) pair used to index the numpy array and set to 1
            bExist[drug_dict[drug], cell_dict[cell]] = 1
            # appending (str) name of this drug to list lst_drug
            lst_drug.append(drug)
            # appending (numeric str) this Cosmic sample Id to list lst_cell
            lst_cell.append(cell)

        else:
            # if drug not in drug_dict:
            #     print('unrecognized drug:', drug)
            # if cell not in cell_dict:
            #     print('unrecognized cell line:', cell)

            n_missing += 1

    print('missing pairs:', n_missing)

    if (return_names):
        xd, xc, y, dglist, coslist = np.asarray(xd), np.asarray(
            xc), np.asarray(y), np.asarray(lst_drug), np.asarray(lst_cell)
    else:
        xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    if (return_names):
        return xd, xc, y, dglist, coslist
    else:
        return xd, xc, y


# functions to use gene expression data from CCLE
def preproc_gene_expr(ccle_expr, meta_data, top_n=1000, filter_by_l1000 = False):
    if filter_by_l1000:
        l1000_gene_df = pd.read_csv('data/landmark_genes.txt', sep='\t', header=0)
        landmark_genes = l1000_gene_df['Symbol'].values
        ccle_genes = ccle_expr.columns.values
        selected_genes = np.intersect1d(landmark_genes, ccle_genes)
        filtered_expr = ccle_expr[selected_genes]
    else:
        # remove genes with low expression levels and select top n (default=1000) genes according to variance
        ccle_expr = ccle_expr.loc[:,
                                (ccle_expr == 0).sum() < ccle_expr.shape[0]*0.1]
        expr_var = ccle_expr.var()
        expr_var_arr = np.array(expr_var)
        gene_rnk = np.flip(np.argsort(expr_var_arr))
        filtered_expr = ccle_expr.iloc[:, gene_rnk[:top_n]]

    meta_data = meta_data[meta_data['COSMICID'].notna()]
    expr_data = filtered_expr.merge(
        meta_data, left_index=True, right_on='DepMap_ID')
    expr_data.drop('DepMap_ID', axis=1, inplace=True)

    expr_data['COSMICID'] = expr_data['COSMICID'].astype(int).astype(str)
    expr_data.set_index('COSMICID', inplace=True)

    return expr_data


def save_gene_expr_matrix_X(top_n=1000, folder='data/CCLE/', filter_by_l1000=False):
    df = pd.read_csv(folder + 'CCLE_expression.csv', index_col=0, header=0)
    meta_df = pd.read_csv(folder + 'sample_info.csv',
                          header=0, usecols=['DepMap_ID', 'COSMICID'])
    processed_df = preproc_gene_expr(df, meta_df, top_n, filter_by_l1000)

    cells = processed_df.index.values
    cell_dict = dict()
    for c in cells:
        idx = np.where(cells == c)[0]
        cell_dict[c] = idx

    cell_feature = processed_df.values
    gene_list = processed_df.columns

    return cell_dict, cell_feature, gene_list


def save_mix_drug_geneexpr_matrix_X(do_ordinary_atom_feat=True, do_mol_ecfp=False, fpl=None, do_edge_features=False, do_atom_ecfp=False, ecfp_radius=None, use_radius=None, use_relational_edge=False, return_names=True, top_n=1000, filter_by_l1000=False, folder='data/GDSC/'):
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

#     cell_dict, cell_feature, qa, aq = save_cell_mut_matrix_X()
    cell_dict, cell_feature, _ = save_gene_expr_matrix_X(top_n=top_n, filter_by_l1000=filter_by_l1000)
    drug_dict, drug_smile, smile_graph = load_drug_smile_X(
        do_ordinary_atom_feat, do_mol_ecfp, fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius, use_relational_edge)

    print('drug number:', len(drug_dict))
    print('cell line number:', len(cell_dict))

    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]  # Drug name
        cell = item[3]  # Cosmic sample Id
        ic50 = item[8]  # IC50
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    print('total length of drug-cellline pair:', len(temp_data))

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []

    # TODO: remove this shuffle operation. (finished)
    # for mixed test, shuffle will be done in load_data.py, controlling by the random seed
    # for blind test, no shuffle is needed
    # random.shuffle(temp_data)

    n_missing = 0

    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            # appending the smile of the drug into list xd
            xd.append(drug_smile[drug_dict[drug]])
            # appending numpy array of shape (len(mut_dict),) ie. (735,) to list xc
            xc.append(cell_feature[cell_dict[cell]])
            # appending (int) ic50 value of that smile to list y
            y.append(ic50)
            # (drug_name, Cosmic_sample_Id) pair used to index the numpy array and set to 1
            bExist[drug_dict[drug], cell_dict[cell]] += 1
            # appending (str) name of this drug to list lst_drug
            lst_drug.append(drug)
            # appending (numeric str) this Cosmic sample Id to list lst_cell
            lst_cell.append(cell)

        else:
            # if drug not in drug_dict:
            #     print('unrecognized drug:', drug)
            # if cell not in cell_dict:
            #     print('unrecognized cell line:', cell)

            n_missing += 1

    print('missing pairs:', n_missing)

    if (return_names):
        xd, xc, y, dglist, coslist = np.asarray(xd), np.asarray(
            xc), np.asarray(y), np.asarray(lst_drug), np.asarray(lst_cell)
    else:
        xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    if (return_names):
        return xd, xc, y, dglist, coslist, bExist
    else:
        return xd, xc, y, bExist
