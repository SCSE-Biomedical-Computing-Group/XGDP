from nis import match
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import os, argparse
from collections import defaultdict
from torch_geometric.loader import DataLoader
import torch
from utils_data import TestbedDataset
from models import GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet, GINNet, GINENet
from rdkit_heatmaps.molmapping import mapvalues2mol
from rdkit_heatmaps.utils import transform2png

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, default=0, help="model type: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE")
# parser.add_argument("-o", "--object", type=int, default=1, help="decoding object: 0:atoms, 1:bonds, 2:cell_line")
# parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, default='001', help="branch")
parser.add_argument("-a", "--annotation", type=int, default=2, help="annotation type: 0:numbers, 1:heatmap, 2:both")

args = parser.parse_args()
model_type = args.model
# decoding_object = args.object
# gpu = args.gpu
b = args.branch
annotation = args.annotation

model_name = ['GCN', 'GAT', 'GAT_Edge', 'GATv2', 'SAGE', 'GIN', 'GINE'][model_type]
branch_folder = "root_folder/root_" + b

dataset = 'GDSC'
test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
test_batch = 1
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

d_path = branch_folder + '/Saliency/GNNExplainer/Bonds/' + model_name
d_save_path = branch_folder + '/SaliencyMap/GNNExplainer/Bonds/' + model_name

if annotation == 0:
    d_save_path = d_save_path + '/saliency_score'
elif annotation == 1:
    d_save_path = d_save_path + '/heatmap'
elif annotation == 2:
    d_save_path = d_save_path + '/ss+heatmap'
else:
    print('wrong annotation type!')
    exit
        
os.makedirs(d_save_path, exist_ok=True)

# for idx, data in enumerate(test_loader):
#     print("progress: ", idx, "/", len(test_data))


def make_drug_dict(dir):
    drug_dict = dict()
    drug_sal_dict = dict()
    for filename in os.listdir(dir):
        drug_name = filename.split('_')[1]
        one = np.load(os.path.join(dir, filename))
        if drug_name not in drug_dict.keys() and drug_name not in drug_sal_dict.keys():
            drug_dict[drug_name] = 1
            drug_sal_dict[drug_name] = one
        else:
            drug_dict[drug_name] += 1
            drug_sal_dict[drug_name] = np.add(drug_sal_dict[drug_name], one)
    
    for k, v in drug_sal_dict.items():
        drug_sal_dict[k] = v/drug_dict[k]
    
    return drug_dict, drug_sal_dict


def make_edge_dict(loader):
    smiles_dict = dict()
    edge_index_dict = dict()
    for data in loader:
        drug_name = data.drug_name[0]
        smiles = data.smiles[0]
        edge_index = data.edge_index.numpy()
        
        if drug_name not in smiles_dict.keys() and drug_name not in edge_index_dict.keys():
            smiles_dict[drug_name] = smiles
            edge_index_dict[drug_name] = edge_index
    
    return smiles_dict, edge_index_dict


def draw_mol_saliency_scores(drug_sal_dict, smiles_dict, edge_index_dict, save_path, annotation_type):
    for k, v in drug_sal_dict.items():
        print('working on ', k)
        edge_index = edge_index_dict[k]
        # norm_v = (v - v.min()) /(v.max() - v.min())

        edge_ss_dict = defaultdict(float)
        counts = defaultdict(int)

        # for val, x, y in zip(norm_v, *edge_index):
        for val, x, y in zip(v, *edge_index):
            if x > y:
                x, y = y, x
            edge_ss_dict[(x, y)] += val
            counts[(x, y)] += 1
            
        for edge, count in counts.items():
            edge_ss_dict[edge] /= count
            # edge_ss_dict[edge] = edge_ss_dict[edge].round(2)
            
        min_ss = min(edge_ss_dict.values())
        max_ss = max(edge_ss_dict.values())
        # print(min_ss, max_ss)

        for edge, value in edge_ss_dict.items():
            edge_ss_dict[edge] = (value - min_ss)/(max_ss - min_ss)
            edge_ss_dict[edge] = edge_ss_dict[edge].round(2)

        smiles = smiles_dict[k]
        mol = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol)
        AllChem.Compute2DCoords(mol)
        bond_weights = []
        for i, bond in enumerate(mol.GetBonds()):
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u > v:
                u, v = v, u
            if annotation_type == 0:
                bond.SetProp('bondNote',str(edge_ss_dict[(u, v)]))
            elif annotation_type == 1:
                bond_weights.append(edge_ss_dict[(u, v)])
            elif annotation_type == 2:
                bond.SetProp('bondNote',str(edge_ss_dict[(u, v)]))
                bond_weights.append(edge_ss_dict[(u, v)])
        
        if annotation_type == 0:
            Chem.Draw.MolToImageFile(mol, os.path.join(save_path, k + '.png'), size = (1000, 1000))
        elif annotation_type == 1 or annotation_type == 2:
            canvas = mapvalues2mol(mol, bond_weights = bond_weights)
            img = transform2png(canvas.GetDrawingText())
            img.save(os.path.join(save_path, k + '.png'))
        

_, sal_dict = make_drug_dict(d_path)
print('all drugs: ', sal_dict.keys())
smiles_dict, edge_idx_dict = make_edge_dict(test_loader)
draw_mol_saliency_scores(sal_dict, smiles_dict, edge_idx_dict, d_save_path, annotation)