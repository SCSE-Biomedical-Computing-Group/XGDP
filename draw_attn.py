# this is only applicable to bonds decoding

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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# device = torch.device("cpu")
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

model = [GCNNet(), GATNet(), GATNet_E(), GATv2Net(), SAGENet(), GINNet(), GINENet()][model_type]
model_name = ['GCN', 'GAT', 'GAT_Edge', 'GATv2', 'SAGE', 'GIN', 'GINE'][model_type]

branch_folder = "root_folder/root_" + b
model_path = os.path.join(branch_folder, 'models/model_' + model_name + '-EP300-SW801010_GDSC.model')

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

decoding_type = ['Atoms', 'Bonds', 'CellLine'][decoding_object]

dataset = 'GDSC'
test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
test_batch = 1
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

d_save_path = 'root_folder/root_' + b + '/SaliencyMap/' + decoding_type + '/' + model_name + '/'
os.makedirs(d_save_path, exist_ok=True)

for idx, data in enumerate(test_loader):
    data = data.to(device)
    drug = data.drug_name[0]
    cell_line = data.cell_line_name[0]
    filename = drug + '_' + cell_line +'.png'
    print(idx, drug, cell_line)
    
    out, _, attn_weights = model(data.x, data.edge_index, data.target, data.batch, data.edge_features, return_attention_weights=True)    
    attn_weights = attn_weights[1].to('cpu').detach().numpy()
    attn_weights = attn_weights[:-data.x.shape[0], ]
    
    agg_attn_weights = np.zeros(int(attn_weights.shape[0]/2))
    for i in range(0, agg_attn_weights.shape[0]):
        agg_attn_weights[i] = attn_weights[2*i] + attn_weights[2*i+1]
    norm_attn_weights = (agg_attn_weights - agg_attn_weights.min()) /(agg_attn_weights.max() - agg_attn_weights.min())
    norm_attn_weights = norm_attn_weights.round(2)
    
    mol = Chem.MolFromSmiles(data.smiles[0])
    for i, bond in enumerate(mol.GetBonds()):
        bond.SetProp('bondNote',str(norm_attn_weights[i]))
        
    Chem.Draw.MolToImageFile(mol, d_save_path + filename, size = (1000, 1000))