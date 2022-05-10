import pandas as pd
import pubchempy as pcp
from collections import Counter

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
# import pandasql

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
# from models.gat import GATNet
# from models.gat_gcn import GAT_GCN
# from models.gcn import GCNNet
# from models.ginconv import GINConvNet
# from utils import *
import datetime
import argparse
import nvidia_smi


"""## Data Setup"""

# folder = "gdrive/MyDrive/FYP/Data/DRP/"  ## aka folder
# root_folder = "gdrive/MyDrive/FYP/Data/DRP/root_folder/"
# data_folder = "gdrive/MyDrive/FYP/Data/"
# sage_folder = "gdrive/MyDrive/FYP/Data/SAGEData/"

folder = "data/"  ## aka folder
root_folder = "root_folder/"
data_folder = "data/"
# sage_folder = "gdrive/MyDrive/FYP/Data/SAGEData/"

# df0 = pd.read_csv('gdrive/MyDrive/FYP/Data/DRP/drug_smiles.csv') 
df0 = pd.read_csv(data_folder+'/drug_smiles.csv')
df0.rename(columns = {'name':'drug_name'}, inplace = True)
print(f"df0 Length = {len(df0)}")
print()
unique_drug_name_df0 = list(Counter(df0["drug_name"].tolist()).keys())
print(f"unique_drug_name_df0 = {len(unique_drug_name_df0)}")
unique_cid_df0 = list(Counter(df0["CID"].tolist()).keys())
print(f"unique_cid_df0 = {len(unique_cid_df0)}")
unique_CanonicalSMILES_df0 = list(Counter(df0["CanonicalSMILES"].tolist()).keys())
print(f"unique_CanonicalSMILES_df0 = {len(unique_CanonicalSMILES_df0)}")
unique_IsomericSMILES_df0 = list(Counter(df0["IsomericSMILES"].tolist()).keys())
print(f"unique_IsomericSMILES_df0 = {len(unique_IsomericSMILES_df0)}")
df0.head()

# df1 = pd.read_csv(data_folder = 'PANCANCER_IC.csv') ## count = 135,242 PANCANCER_IC PANCANCER_IC.csv
df1 = pd.read_csv(data_folder + 'PANCANCER_IC.csv')  ## count = 224,510 correct
df1.rename(columns = {'Drug name':'drug_name'}, inplace = True)
df1.rename(columns = {'Drug Id':'drug_id'}, inplace = True)
print(f"df1 Length = {len(df1)}")
df1.head()

df2 = pd.read_csv(data_folder + 'drugs_smile.csv', sep="\t")
print("df2")
print()
unique_drug_id_df2 = list(Counter(df2["drug_id"].tolist()).keys())
print(f"unique_drug_id_df2 = {len(unique_drug_id_df2)}")
unique_smiles_df2 = list(Counter(df2["SMILES"].tolist()).keys())
print(f"unique_smiles_df2 = {len(unique_smiles_df2)}")
print()
print(f"df2 Length = {len(df2)}")
df2.head()

df3 = pd.merge(df1, df2, how ='inner', on =['drug_id'])
df3.rename(columns = {'Cosmic sample Id':'cosmic_sample_id'}, inplace = True)
df4 = pd.read_csv('gdrive/MyDrive/FYP/Data/PANCANCER_Genetic_feature.csv')
unique_genetic_feature = list(dict.fromkeys(list(df4.genetic_feature)))
df4 = pd.read_csv('gdrive/MyDrive/FYP/Data/PANCANCER_Genetic_feature.csv')

df4

print("df3")
print()
unique_drug_name_df3 = list(Counter(df3["drug_name"].tolist()).keys())
print(f"unique_drug_name_df3 = {len(unique_drug_name_df3)}")
unique_drug_name_df3 = list(Counter(df3["drug_id"].tolist()).keys())
print(f"unique_drug_name_df3 = {len(unique_drug_name_df3)}")
unique_cell_line_name_df3 = list(Counter(df3["Cell line name"].tolist()).keys())
print(f"unique_cell_line_name_df3 = {len(unique_cell_line_name_df3)}")
unique_cosmic_sample_id_df3 = list(Counter(df3["cosmic_sample_id"].tolist()).keys())
print(f"unique_cosmic_sample_id_df3 = {len(unique_cosmic_sample_id_df3)}")
unique_tcga_classification_df3= list(Counter(df3["TCGA classification"].tolist()).keys())
print(f"unique_tcga_classification_df3 = {len(unique_tcga_classification_df3)}")
unique_tissue_df3= list(Counter(df3["Tissue"].tolist()).keys())
print(f"unique_tissue_df3 = {len(unique_tissue_df3)}")
unique_smiles_df3 = list(Counter(df3["SMILES"].tolist()).keys())
print(f"unique_smiles_df3 = {len(unique_smiles_df3)}")
print()
print(f"df3 Length = {len(df3)}")
df3.head()

print("df4")
print()
unique_cell_line_name_df4 = list(Counter(df4["cell_line_name"].tolist()).keys())
print(f"unique_cell_line_name_df4 = {len(unique_cell_line_name_df4)}")
unique_cosmic_sample_id_df4 = list(Counter(df4["cosmic_sample_id"].tolist()).keys())
print(f"unique_cosmic_sample_id_df4 = {len(unique_cosmic_sample_id_df4)}")
unique_gdsc_desc1_df4 = list(Counter(df4["gdsc_desc1"].tolist()).keys())
print(f"unique_gdsc_desc1_df4 = {len(unique_gdsc_desc1_df4)}")
unique_gdsc_desc2_df4 = list(Counter(df4["gdsc_desc2"].tolist()).keys())
print(f"unique_gdsc_desc2_df4 = {len(unique_gdsc_desc2_df4)}")
unique_tcga_desc_df4 = list(Counter(df4["tcga_desc"].tolist()).keys())
print(f"unique_tcga_desc_df4 = {len(unique_tcga_desc_df4)}")
unique_genetic_feature_df4= list(Counter(df4["genetic_feature"].tolist()).keys())
print(f"unique_genetic_feature_df4 = {len(unique_genetic_feature_df4)}")
unique_is_mutated_df4= list(Counter(df4["is_mutated"].tolist()).keys())
print(f"unique_is_mutated_df4 = {len(unique_is_mutated_df4)}")
print()
print(f"df4 Length = {len(df4)}")
df4.head()

df5 = pd.merge(df1, df0, how ='inner', on =['drug_name'])
df5 = df5.drop(['IsomericSMILES'], axis = 1)
df5.rename(columns = {'Cosmic sample Id':'cosmic_sample_id'}, inplace = True)
df5.rename(columns = {'CanonicalSMILES':'SMILES'}, inplace = True)

unique_cosmic_sample_id_df5 = list(Counter(df5["cosmic_sample_id"].tolist()).keys())
unique_name_diff = list(set(unique_cosmic_sample_id_df4) - set(unique_cosmic_sample_id_df5)) + list(set(unique_cosmic_sample_id_df5) - set(unique_cosmic_sample_id_df4))  
in_both = []
for cd in unique_cosmic_sample_id_df5:
    if (cd in unique_cosmic_sample_id_df4):
        in_both.append(cd)
all_csi = df5["cosmic_sample_id"].tolist()
all_i = [i for i in range(len(all_csi))]
remove_these_i = []
for i, c in zip(all_i, all_csi):
    if (c not in in_both):
        remove_these_i.append(i)
df5 = df5.drop(remove_these_i)
df5.reset_index(inplace = True)
df5 = df5.drop(['index'], axis = 1)

print("df5")
print()
unique_drug_name = list(Counter(df5["drug_name"].tolist()).keys())
print(f"unique_drug_name = {len(unique_drug_name)}")

unique_drug_id = list(Counter(df5["drug_id"].tolist()).keys())
print(f"unique_drug_id = {len(unique_drug_id)}")

unique_cell_line_name = list(Counter(df5["Cell line name"].tolist()).keys())
print(f"unique_cell_line_name = {len(unique_cell_line_name)}")

unique_cosmic_sample_id = list(Counter(df5["cosmic_sample_id"].tolist()).keys())
print(f"unique_cosmic_sample_id = {len(unique_cosmic_sample_id)}")

unique_tcga_classification= list(Counter(df5["TCGA classification"].tolist()).keys())
print(f"unique_tcga_classification = {len(unique_tcga_classification)}")

unique_tissue= list(Counter(df5["Tissue"].tolist()).keys())
print(f"unique_tissue = {len(unique_tissue)}")

unique_tissue_sub_type= list(Counter(df5["Tissue sub-type"].tolist()).keys())
print(f"unique_tissue_sub_type = {len(unique_tissue_sub_type)}")

unique_CID= list(Counter(df5["CID"].tolist()).keys())
print(f"unique_CID = {len(unique_CID)}")

unique_smiles = list(Counter(df5["SMILES"].tolist()).keys())
print(f"unique_smiles = {len(unique_smiles)}")

print()
print(f"df5 Length = {len(df5)}")

df5.head()

this_smiles = list(df3.iloc[[23]].SMILES)[0]
show_structure(this_smiles, show_smiles =True)

this_smiles = list(df3.iloc[[1213]].SMILES)[0]
show_structure(this_smiles, show_smiles =True)  ## ewq

class ECFP6:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles

    def mol2fp(self, mol, fp_length, radius = 3):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits = fp_length)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def compute_ECFP6(self, fp_length, name = None, generate_df = True):
        bit_headers = ['bit' + str(i) for i in range(fp_length)]
        arr = np.empty((0,fp_length), int).astype(int)
        for i in self.mols:
            fp = self.mol2fp(i, fp_length)
            arr = np.vstack((arr, fp))
        if (not generate_df):
            return np.asarray(arr).astype(int)
        df_ecfp6 = pd.DataFrame(np.asarray(arr).astype(int),columns=bit_headers)
        df_ecfp6.insert(loc=0, column='smiles', value=self.smiles)
        if name != None:
            df_ecfp6.to_csv(name[:-4]+'_ECFP6.csv', index=False)
        
        return df_ecfp6
smiles = [standardize_smiles(i) for i in unique_smiles] 
convert_to_nonstd = dict(zip(smiles, unique_smiles)) 


ecfp6_descriptor = ECFP6(smiles)        
df6 = ecfp6_descriptor.compute_ECFP6(fp_length = 1024)

smiles = [standardize_smiles(i) for i in [unique_smiles[0]]] 
convert_to_nonstd = dict(zip(smiles, [unique_smiles[0]])) 

ecfp6_descriptor = ECFP6(smiles)        
df_X = ecfp6_descriptor.compute_ECFP6(fp_length = 1024) 
df_X

print("df6")
print()
unique_smiles_df6 = list(Counter(df6["smiles"].tolist()).keys())
print(f"unique_smiles_df6 = {len(unique_smiles_df6)}")
print()
print(f"df6 Length = {len(df6)}")
df5.head()

smiles = [standardize_smiles(i) for i in unique_smiles] 
convert_to_nonstd = dict(zip(smiles, unique_smiles)) 

ecfp6_descriptor = ECFP6(smiles)     
df7 = ecfp6_descriptor.compute_ECFP6(fp_length = 2048//2)

print("df7")
print()
unique_smiles_df7 = list(Counter(df7["smiles"].tolist()).keys())
print(f"unique_smiles = {len(unique_smiles)}")
print()
print(f"df7 Length = {len(df7)}")
df6.head()