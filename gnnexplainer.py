from torch_geometric.loader import DataLoader
import torch
from torch_geometric.nn import GNNExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from utils_data import TestbedDataset
from models import GATNet_E, GATNet, GCNNet, GATv2Net, GINNet, GINENet, SAGENet, WIRGATNet, RGCNNet
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, default=0, help="model type: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE, 7:WIRGAT, 8:ARGAT, 9:RGCN")
# parser.add_argument("-o", "--object", type=int, default=0, help="decoding object: 0:node features, 1:edge importance")
parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, default='001', help="branch")

args = parser.parse_args()
model_type = args.model
# decoding_object = args.object
gpu = args.gpu
b = args.branch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# device = torch.device("cpu")
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

# model = GATNet_E()
# model_path = 'root_folder/root_002/results/model_GAT_Edge-EP300-SW801010_GDSC.model'

model = [GCNNet(), GATNet(), GATNet_E(), GATv2Net(), SAGENet(), GINNet(), GINENet(), WIRGATNet(), None, RGCNNet()][model_type]
model_name = ['GCN', 'GAT', 'GAT_Edge', 'GATv2', 'SAGE', 'GIN', 'GINE', 'WIRGAT', 'ARGAT', 'RGCN'][model_type]

branch_folder = "root_folder/root_" + b
model_path = os.path.join(branch_folder, 'models/model_' + model_name + '-EP300-SW801010_GDSC.model')

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# decoding_type = ['AtomFeatures', 'Bonds'][decoding_object]

dataset = 'GDSC'
test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
test_batch = 1
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)


def explain(data, device):
    data = data.to(device)
    explainer = GNNExplainer(model, lr=1e-4, epochs=300, return_type='regression')
    node_feat_mask, edge_mask = explainer.explain_graph(x = data.x, edge_index = data.edge_index, x_cell_mut = data.target, edge_feat = data.edge_features)
    node_feat_mask = node_feat_mask.cpu().detach().numpy()
    edge_mask = edge_mask.cpu().detach().numpy()
    
    return node_feat_mask, edge_mask


save_path_node_feat = os.path.join(branch_folder, 'Saliency/GNNExplainer/AtomFeatures/' + model_name + '/')
save_path_edge = os.path.join(branch_folder, 'Saliency/GNNExplainer/Bonds/' + model_name + '/')
os.makedirs(save_path_node_feat, exist_ok=True)
os.makedirs(save_path_edge, exist_ok=True)

for idx, data in enumerate(test_loader):
    print(idx)
    drug_name = data.drug_name[0]
    cell_line_name = data.cell_line_name[0]
    print('drug name: ', drug_name)
    print('cell_line name: ', cell_line_name)
#     print(type(data))
    # print(data.edge_index)
    # print(data.edge_features)
    data = data.to(device)
    node_feat_mask, edge_mask = explain(data, device)
    # mask_cell = explain_cell_line(data, device)
    # print(mask_drug)
    # print(mask_cell)
    np.save(save_path_node_feat + str(idx) + '_' + drug_name + '_' + cell_line_name + '.npy', node_feat_mask)
    np.save(save_path_edge + str(idx) + '_' + drug_name + '_' + cell_line_name + '.npy', edge_mask)
    # np.save(save_path_cell + str(idx) + '.npy', mask_cell)
    
    del data
    torch.cuda.empty_cache()