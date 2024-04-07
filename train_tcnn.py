# Training
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
import sys
import os
from random import shuffle
import torch
import torch.nn as nn
import datetime
import argparse
import nvidia_smi

from utils_tcnn import *

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, default='tcnn', help="branch")
parser.add_argument("-c", "--do_cv", action="store_true", default=False, help="add this flag to do cross validation")

args = parser.parse_args()
gpu = args.gpu
b = args.branch
do_cv = args.do_cv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# save_name = "GCN-EP300-SW801010"
model_name = 'tCNN'
save_name = model_name + "-EP300-SW801010"
# branch_folder = "gdrive/MyDrive/FYP/Data/DRP/root_folder/root_028"
branch = 'root_' + b
branch_folder = "root_folder/" + branch


# save_folder = "gdrive/MyDrive/FYP/Saves/GCN/"
result_folder = branch_folder + "/results/"
model_folder = branch_folder + '/models/'
os.makedirs(result_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

modeling = tCNN

# train_batch = 1024
# val_batch = 1024
# test_batch = 1024

# train_batch = 512
# val_batch = 512
# test_batch = 512

train_batch = 32
val_batch = 32
test_batch = 32

lr = 1e-4
num_epoch = 300
log_interval = 20
cuda_name = gpu
print(f"branch_folder = {branch_folder}")
main_cv(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, br_fol=branch_folder,
     result_folder=result_folder, model_folder=model_folder, save_name=save_name, do_save=True)
