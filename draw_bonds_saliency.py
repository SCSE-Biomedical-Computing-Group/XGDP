import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader
import torch
from captum.attr import Saliency, IntegratedGradients
import numpy as np
import pandas as pd
import argparse, os

from utils_data import TestbedDataset
from models import GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet, GINNet, GINENet


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, default=0, help="model type: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE")
parser.add_argument("-o", "--object", type=int, default=1, help="decoding object: 0:atoms, 1:bonds, 2:cell_line")
parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, default='001', help="branch")

args = parser.parse_args()
model_type = args.model
decoding_object = args.object
gpu = args.gpu
b = args.branch

decoding_type = ['Atoms', 'Bonds', 'CellLine'][decoding_object]
model_name = ['GCN', 'GAT', 'GAT_Edge', 'GATv2', 'SAGE', 'GIN', 'GINE'][model_type]

branch_folder = "root_folder/root_" + b

dataset = 'GDSC'
test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
test_batch = 1
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

d_path = branch_folder + '/Saliency/' + decoding_type + '/' + model_name + '/'
d_save_path = branch_folder + '/SaliencyMap/' + decoding_type + '/' + model_name + '/'
os.makedirs(d_save_path, exist_ok=True)

for idx, data in enumerate(test_loader):
    print(idx)
    drug = data.drug_name[0]
    cell_line = data.cell_line_name[0]
    postflix = drug + '_' + cell_line +'.npy'
    filename = d_path + str(idx) + '_' + drug + '_' + cell_line + '.npy'
    
    saliency_score = np.load(filename)
    agg_ss = np.zeros(int(saliency_score.shape[0]/2))
    for i in range(agg_ss.shape[0]):
        agg_ss[i] = saliency_score[2*i] + saliency_score[2*i+1]
    
    norm_ss = (agg_ss - agg_ss.min()) /(agg_ss.max() - agg_ss.min())
    norm_ss = norm_ss.round(2)
    
    smiles = data.smiles[0]
    mol = Chem.MolFromSmiles(smiles)
    for i, bond in enumerate(mol.GetBonds()):
        bond.SetProp('bondNote',str(norm_ss[i]))

    Chem.Draw.MolToImageFile(mol, d_save_path + drug + '_' + cell_line + '.png', size = (1000, 1000))
#     f = [filename for filename in os.listdir(d_path) if filename.match('*_' + drug + '_' + cell_line +'.npy')]
#     filename =  glob.glob(d_path + '*' + postflix)
#     print(filename)
    