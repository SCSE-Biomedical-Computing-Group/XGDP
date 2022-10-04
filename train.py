# Training
from email import parser
from pyexpat import model
import pandas as pd
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
import sys, os
from random import shuffle
from sympy import arg
import torch
import torch.nn as nn
import datetime
import argparse
import nvidia_smi

import utils_preproc
import utils_data
import models_deprecated
# import load_data

from utils_train import main
# from functions import main
# from models_deprecated import GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet
from models import GCNNet, GATNet, GATNet_E, GATv2Net, GINENet, GINNet, SAGENet, WIRGATNet

parser = argparse.ArgumentParser()
# parser.add_argument("-m", "--model", type=str, default="GCN", help="model type: GCN, GAT, GAT_Edge, GATv2, SAGE, GIN, GINE")
parser.add_argument("-m", "--model", type=int, default=0, help="model type: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE, 7:WIRGAT")
parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, required=True, help="branch")

args = parser.parse_args()
model_type = args.model
gpu = args.gpu
b = args.branch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# save_name = "GCN-EP300-SW801010"
model_name = ['GCN', 'GAT', 'GAT_Edge', 'GATv2', 'SAGE', 'GIN', 'GINE', 'WIRGAT'][model_type]
save_name = model_name + "-EP300-SW801010"
# branch_folder = "gdrive/MyDrive/FYP/Data/DRP/root_folder/root_028"
branch = 'root_' + b
branch_folder = "root_folder/" + branch

# if 'GAT' in model_name:
#     return_attention_weights = True
# else:
#     return_attention_weights = False

# save_folder = "gdrive/MyDrive/FYP/Saves/GCN/"
result_folder = branch_folder + "/results/"
model_folder = branch_folder + '/models/'
os.makedirs(result_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)
# if return_attention_weights:
#     os.makedirs(branch_folder + '/Saliency/AttnWeight', exist_ok=True)

# assert(model_type in ["GCN", "GAT", "GAT_Edge", "GATv2", "SAGE", "GIN", "GINE"])
# if model_type == "GCN":
#     modeling = GCNNet
# elif model_type == "GAT":
#     modeling = GATNet
# elif model_type == "GAT_Edge":
#     modeling = GATNet_E
# elif model_type == "GATv2":
#     modeling = GATv2Net
# elif model_type == "SAGE":
#     modeling = SAGENet
# else:
#     print("wrong model type!")
#     exit
modeling = [GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet, GINNet, GINENet, WIRGATNet][model_type]
    
# train_batch = 1024
# val_batch = 1024
# test_batch = 1024
train_batch = 512
val_batch = 512
test_batch = 512

lr = 1e-4
num_epoch = 300
log_interval = 20
cuda_name = gpu
print(f"branch_folder = {branch_folder}")
main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, br_fol = branch_folder, result_folder = result_folder, model_folder = model_folder, save_name = save_name, return_attention_weights=False, do_save = True)
