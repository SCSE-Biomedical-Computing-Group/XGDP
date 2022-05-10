from torch_geometric.loader import DataLoader
import torch
from captum.attr import Saliency, IntegratedGradients
import numpy as np
import pandas as pd

from utils import TestbedDataset
from models import GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet

device = torch.device("cpu")

# model = GATNet_E()
# model_path = 'root_folder/root_002/results/model_GAT_Edge-EP300-SW801010_GDSC.model'
model = GCNNet()
model_path = 'root_folder/root_002/results/model_GCN-EP300-SW801010_GDSC.model'
model.load_state_dict(torch.load(model_path, map_location=device))

branch_folder = "root_folder/root_002"
dataset = 'GDSC'
test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
test_batch = 1
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)


def model_forward(input_mask, data):
    batch = data.batch
    output, _ = model(input_mask, data.target, data.edge_index, batch, data.edge_features)

    return output


def explain_drug(data, device):
    data= data.to(device)
    input_mask_drug = torch.ones(data.x.shape[0], data.x.shape[1]).requires_grad_(True).to(device)
#     input_mask_cell = torch.ones(data.target.shape[0], data.target.shape[1]).requires_grad_(True).to(device)

#     print(input_mask_drug.shape)
#     print(data.edge_index.shape)
    ig = IntegratedGradients(model_forward)
    mask_drug = ig.attribute(input_mask_drug, 
                             target = 0,
                             additional_forward_args = (data,)
#                                           additional_forward_args = (data.edge_index, data.batch),
                             , internal_batch_size=data.x.shape[0]
                            )
    
    mask_drug = np.abs(mask_drug.cpu().detach().numpy())
#     mask_cell = np.abs(mask_cell.cpu().detach().numpy())
    
    if mask_drug.max() > 0:
        mask_drug = mask_drug / mask_drug.max()
        
#     if mask_cell.max() > 0:
#         mask_cell = mask_drug / mask_cell.max()
    
    return mask_drug


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

save_path_drug = 'root_folder/root_002/Saliency/Drug/GCNNet/'
save_path_cell = 'root_folder/root_002/Saliency/CellLine/GCNNet/'
for idx, data in enumerate(test_loader):
    print(idx)
#     print(type(data))
#     print(data.batch)
    data = data.to(device)
    mask_drug = explain_drug(data, device)
    # mask_cell = explain_cell_line(data, device)
    # print(mask_drug)
    # print(mask_cell)
    np.save(save_path_drug + str(idx) + '.npy', mask_drug)
    # np.save(save_path_cell + str(idx) + '.npy', mask_cell)
    
    del data
    torch.cuda.empty_cache()
    