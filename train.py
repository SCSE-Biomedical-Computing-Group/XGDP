# Training
from email import parser
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

import functions
import utils
import models_deprecated
import load_data

from functions import main
# from models_deprecated import GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet
from models import GCNNet, GATNet, GATNet_E, GATv2Net, SAGENet

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="GCN", help="model type: GCN, GAT, GAT_Edge, GATv2, SAGE")
parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")

args = parser.parse_args()
model_type = args.model
gpu = args.gpu

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# save_folder = "gdrive/MyDrive/FYP/Saves/GCN/"
save_folder = "root_folder/root_002/results/"
os.makedirs(save_folder, exist_ok=True)

# save_name = "GCN-EP300-SW801010"
save_name = model_type + "-EP300-SW801010"
# branch_folder = "gdrive/MyDrive/FYP/Data/DRP/root_folder/root_028"
branch_folder = "root_folder/root_002"

assert(model_type in ["GCN", "GAT", "GAT_Edge", "GATv2", "SAGE"])
if model_type == "GCN":
    modeling = GCNNet
elif model_type == "GAT":
    modeling = GATNet
elif model_type == "GAT_Edge":
    modeling = GATNet_E
elif model_type == "GATv2":
    modeling = GATv2Net
elif model_type == "SAGE":
    modeling = SAGENet
else:
    print("wrong model type!")
    exit
train_batch = 1024
val_batch = 1024
test_batch = 1024
lr = 1e-4
num_epoch = 100
log_interval = 20
cuda_name = gpu
print(f"branch_folder = {branch_folder}")
main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, br_fol = branch_folder, save_folder = save_folder, save_name = save_name, do_save = True)
