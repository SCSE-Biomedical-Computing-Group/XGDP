from rdkit import Chem
from rdkit.Chem import AllChem
import os, argparse
from collections import defaultdict
from torch_geometric.loader import DataLoader
import torch
from utils_data import TestbedDataset
# from models import GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet, GINNet, GINENet, WIRGATNet, ARGATNet, RGCNNet
from rdkit_heatmaps.molmapping import mapvalues2mol
from rdkit_heatmaps.utils import transform2png
from utils_decoding import make_ss_dict, make_edge_dict, draw_mol_saliency_scores, draw_saliency_scores
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, default=0, help="model type: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE, 7:WIRGAT, 8:ARGAT, 9:RGCN")
# parser.add_argument("-o", "--object", type=int, default=1, help="decoding object: 0:atoms, 1:bonds, 2:cell_line")
# parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, default='001', help="branch")
parser.add_argument("-a", "--annotation", type=int, default=3, help="annotation type: 0:numbers, 1:heatmap, 2:both, 3:functional group heatmap")
parser.add_argument("-e", "--explain_type", type=int,
                    default=1, help="explain type: 0:model, 1:phenomenon")

args = parser.parse_args()
model_type = args.model
# decoding_object = args.object
# gpu = args.gpu
b = args.branch
annotation = args.annotation
exp = args.explain_type

model_name = ['GCN', 'GAT', 'GAT_Edge', 'GATv2', 'SAGE', 'GIN', 'GINE', 'WIRGAT', 'ARGAT', 'RGCN'][model_type]
branch_folder = "root_folder/root_" + b
explanation_type = ['model', 'phenomenon'][exp]

dataset = 'GDSC'
test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
test_batch = 1
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

d_edge_path = branch_folder + '/Saliency/GNNExplainer/Bonds/' + explanation_type + '/' + model_name
d_node_path = branch_folder + '/Saliency/GNNExplainer/Atoms/' + explanation_type + '/' + model_name
d_save_path = branch_folder + '/SaliencyMap/GNNExplainer/' + explanation_type + '/' + model_name

if annotation == 0:
    d_save_path = d_save_path + '/saliency_score'
elif annotation == 1:
    d_save_path = d_save_path + '/heatmap'
elif annotation == 2:
    d_save_path = d_save_path + '/atom_heatmap'
elif annotation == 3:
    d_save_path = d_save_path + '/fg_heatmap'
else:
    print('wrong annotation type!')
    exit()
        
os.makedirs(d_save_path, exist_ok=True)

_, node_sal_dict, edge_sal_dict = make_ss_dict(d_node_path, d_edge_path)
print('all drugs: ', edge_sal_dict.keys())
smiles_dict, edge_idx_dict = make_edge_dict(test_loader)
if annotation == 3:
    with open('data/GDSC/decoding_vocabulary.pkl', 'rb') as file:
        decoding_voc = pickle.load(file)
else:
    decoding_voc = None
draw_saliency_scores(decoding_voc, node_sal_dict, edge_sal_dict, smiles_dict, edge_idx_dict, d_save_path, annotation)
# if annotation == 3:
#     with open('data/GDSC/decoding_vocabulary.pkl', 'rb') as file:
#         decoding_voc = pickle.load(file)
#     # print('decoding vocabulary: ', decoding_voc)
#     draw_fg_saliency_scores(decoding_voc, node_sal_dict, edge_sal_dict, smiles_dict, edge_idx_dict, d_save_path, annotation)
# else:
#     draw_mol_saliency_scores(node_sal_dict, edge_sal_dict, smiles_dict, edge_idx_dict, d_save_path, annotation)