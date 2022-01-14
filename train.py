"""## Training
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
import torch
import torch.nn as nn
import datetime
import argparse
import nvidia_smi

import functions
import utils
import models
import load_data

save_folder = "gdrive/MyDrive/FYP/Saves/GCN/"
save_name = "GCN-EP300-SW801010"
branch_folder = "gdrive/MyDrive/FYP/Data/DRP/root_folder/root_028"

modeling = GCNNet
train_batch = 1024
val_batch = 1024
test_batch = 1024
lr = 1e-4
num_epoch = 300
log_interval = 20
cuda_name = 0
print(f"branch_folder = {branch_folder}")
main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, br_fol = branch_folder, save_folder = save_folder, save_name = save_name, do_save = True)
