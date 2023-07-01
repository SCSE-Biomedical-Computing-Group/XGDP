from torch_geometric.loader import DataLoader
import torch
# from torch_geometric.nn import GNNExplainer
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from utils_data import TestbedDataset
from models import GATNet_E, GATNet, GCNNet, GATv2Net, GINNet, GINENet, SAGENet, WIRGATNet, ARGATNet, RGCNNet
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, default=0,
                    help="model type: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE, 7:WIRGAT, 8:ARGAT, 9:RGCN")
# parser.add_argument("-o", "--object", type=int, default=0, help="decoding object: 0:node features, 1:edge importance")
parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, default='001', help="branch")
parser.add_argument("-e", "--explain_type", type=int,
                    default=1, help="explain type: 0:model, 1:phenomenon")
parser.add_argument("-a", "--do_attn", action="store_true", default=False, help="add this flag to combine features with attn layer")

args = parser.parse_args()
model_type = args.model
# decoding_object = args.object
gpu = args.gpu
b = args.branch
exp = args.explain_type
do_attn = args.do_attn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# device = torch.device("cpu")
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

# model = GATNet_E()
# model_path = 'root_folder/root_002/results/model_GAT_Edge-EP300-SW801010_GDSC.model'

model_class = [GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet, GINNet,
         GINENet, WIRGATNet, ARGATNet, RGCNNet][model_type]
model = model_class(use_attn=do_attn)
model_name = ['GCN', 'GAT', 'GAT_Edge', 'GATv2', 'SAGE',
              'GIN', 'GINE', 'WIRGAT', 'ARGAT', 'RGCN'][model_type]

explanation_type = ['model', 'phenomenon'][exp]

branch_folder = "root_folder/root_" + b
model_path = os.path.join(
    branch_folder, 'models/model_' + model_name + '-EP300-SW801010_GDSC_best.model')

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# decoding_type = ['AtomFeatures', 'Bonds'][decoding_object]

dataset = 'GDSC'
test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
test_batch = 1
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)


def explain(data, device):
    data = data.to(device)
    y = data.y.view(test_batch, data.y.size(0))
    # explainer = GNNExplainer(model, lr=1e-4, epochs=300, return_type='regression')
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=300),
        explanation_type=explanation_type,    # 'phenomenon'
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )
    # node_feat_mask, edge_mask = explainer(x = data.x, edge_index = data.edge_index, x_cell_mut = data.target, edge_feat = data.edge_features)
    # node_feat_mask = node_feat_mask.cpu().detach().numpy()
    # edge_mask = edge_mask.cpu().detach().numpy()

    explanation = explainer(x=data.x, edge_index=data.edge_index, target=y,
                            batch=data.batch, x_cell_mut=data.target, edge_feat=data.edge_features)
    node_mask = explanation.node_mask.cpu().detach().numpy()
    edge_mask = explanation.edge_mask.cpu().detach().numpy()

    return node_mask, edge_mask


# save_path_node_feat = os.path.join(branch_folder, 'Saliency/GNNExplainer/AtomFeatures/' + model_name + '/')
save_path_node = os.path.join(
    branch_folder, 'Saliency/GNNExplainer/Atoms/' + explanation_type + '/' + model_name + '/')
save_path_edge = os.path.join(
    branch_folder, 'Saliency/GNNExplainer/Bonds/' + explanation_type + '/' + model_name + '/')
os.makedirs(save_path_node, exist_ok=True)
os.makedirs(save_path_edge, exist_ok=True)

for idx, data in enumerate(tqdm(test_loader)):
    # print(idx)
    drug_name = data.drug_name[0]
    cell_line_name = data.cell_line_name[0]
    # print('drug name: ', drug_name)
    # print('cell_line name: ', cell_line_name)
#     print(type(data))
    # print(data.edge_index)
    # print(data.edge_features)
    data = data.to(device)
    node_mask, edge_mask = explain(data, device)
    # mask_cell = explain_cell_line(data, device)
    # print(mask_drug)
    # print(mask_cell)
    np.save(save_path_node + str(idx) + '_' + drug_name +
            '_' + cell_line_name + '.npy', node_mask)
    np.save(save_path_edge + str(idx) + '_' + drug_name +
            '_' + cell_line_name + '.npy', edge_mask)
    # np.save(save_path_cell + str(idx) + '.npy', mask_cell)

    del data
    torch.cuda.empty_cache()
