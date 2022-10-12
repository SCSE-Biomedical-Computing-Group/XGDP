from torch_geometric.loader import DataLoader
import torch
from captum.attr import Saliency, IntegratedGradients
import numpy as np
import pandas as pd
import argparse, os

from utils_data import TestbedDataset
from models import GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet, GINNet, GINENet, WIRGATNet, RGCNNet

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, default=0, help="model type: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE, 7:WIRGAT, 8:ARGAT, 9:RGCN")
parser.add_argument("-o", "--object", type=int, default=2, help="decoding object: 0:drug, 1:drug edges, 2:cell_line")
parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, default='001', help="branch")

args = parser.parse_args()
model_type = args.model
decoding_object = args.object
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

decoding_type = ['Atoms', 'Bonds', 'CellLine'][decoding_object]

dataset = 'GDSC'
test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
test_batch = 1
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)


def model_forward(input_mask, data):
    # batch = data.batch
    if decoding_type == 'Atom':
        output = model(input_mask, data.edge_index, data.batch, data.target, data.edge_features)
    elif decoding_type == 'Bonds':  # this is only supported by GCN
        output = model(data.x, data.edge_index, data.batch, data.target, data.edge_features, input_mask)
    elif decoding_type == 'CellLine':
        output = model(data.x, data.edge_index, data.batch, input_mask, data.edge_features)
    else:
        print('wrong decoding type!')
        exit    

    return output


def explain(data, device, decoding_type):
    data = data.to(device)
    if decoding_type == 'Atom':
        # input_mask = torch.ones(data.x.shape[0], data.x.shape[1]).requires_grad_(True).to(device)
        input_mask = data.x
        internal_bs = data.x.shape[0]
    elif decoding_type == 'Bonds':
        input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
        internal_bs = data.edge_index.shape[1]
    elif decoding_type == 'CellLine':
        # cnv features have 2 dim, gene expr features have 3 dim
        # input_mask = torch.ones(data.target.shape[0], data.target.shape[1]).requires_grad_(True).to(device)
        # input_mask = torch.ones(data.target.shape[0], data.target.shape[1], data.target.shape[2]).requires_grad_(True).to(device)
        input_mask = data.target
        internal_bs = data.target.shape[0]
    else:
        print('wrong decoding type!')
        exit  

#     print(input_mask_drug.shape)
#     print(data.edge_index.shape)
    ig = IntegratedGradients(model_forward)
    mask = ig.attribute(input_mask, 
                             target = 0,    # target is the interested output dim for decoding, in our case it's 0 (the only dim), but in multi-class classification task it could be other values
                             additional_forward_args = (data,)
#                                           additional_forward_args = (data.edge_index, data.batch),
                             , internal_batch_size=internal_bs
                            )
    
    mask = np.abs(mask.cpu().detach().numpy())
#     mask_cell = np.abs(mask_cell.cpu().detach().numpy())
    
    if mask.max() > 0:
        mask = mask / mask.max()
        
#     if mask_cell.max() > 0:
#         mask_cell = mask_drug / mask_cell.max()
    
    return mask


'''
def model_forward_cell_line(input_mask, data):
    output, _ = model(data.x, input_mask, data.edge_index, data.batch, data.edge_features)
    return output


def explain_cell_line(data, device):
    data= data.to(device)
    # input_mask_drug = torch.ones(data.x.shape[0], data.x.shape[1]).requires_grad_(True).to(device)
    input_mask_cell = torch.ones(data.target.shape[0], data.target.shape[1]).requires_grad_(True).to(device)

#     print(input_mask_drug.shape)
#     print(data.edge_index.shape)
    ig = IntegratedGradients(model_forward_cell_line)
    mask_drug = ig.attribute(input_mask_cell, 
                             target = 0,
                             additional_forward_args = (data,)
#                                           additional_forward_args = (data.edge_index, data.batch),
                             , internal_batch_size=data.target.shape[0]
                            )
    
    mask_drug = np.abs(mask_drug.cpu().detach().numpy())
#     mask_cell = np.abs(mask_cell.cpu().detach().numpy())
    
    if mask_drug.max() > 0:
        mask_drug = mask_drug / mask_drug.max()
        
#     if mask_cell.max() > 0:
#         mask_cell = mask_drug / mask_cell.max()
    
    return mask_drug
'''

save_path = os.path.join(branch_folder, 'Saliency/IG/' + decoding_type + '/' + model_name + '/')
# save_path_drug = os.path.join(branch_folder, 'Saliency/Drug/' + model_name + '/')
# save_path_cell = os.path.join(branch_folder, 'Saliency/CellLine/' + model_name + '/')
os.makedirs(save_path, exist_ok=True)

for idx, data in enumerate(test_loader):
    print(idx)
    drug_name = data.drug_name[0]
    cell_line_name = data.cell_line_name[0]
    print('drug name: ', drug_name)
    print('cell_line name: ', cell_line_name)
#     print(type(data))
#     print(data.batch)
    data = data.to(device)
    mask_drug = explain(data, device, decoding_type)
    # mask_cell = explain_cell_line(data, device)
    # print(mask_drug)
    # print(mask_cell)
    np.save(save_path + str(idx) + '_' + drug_name + '_' + cell_line_name + '.npy', mask_drug)
    # np.save(save_path_cell + str(idx) + '.npy', mask_cell)
    
    # del data
    # torch.cuda.empty_cache()
    