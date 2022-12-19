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
from utils_decoding import make_ss_dict, make_edge_dict, draw_mol_saliency_scores

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
    exit()
        
os.makedirs(d_save_path, exist_ok=True)

_, sal_dict = make_ss_dict(d_path)
print('all drugs: ', sal_dict.keys())
smiles_dict, edge_idx_dict = make_edge_dict(test_loader)
draw_mol_saliency_scores(sal_dict, smiles_dict, edge_idx_dict, d_save_path, annotation)